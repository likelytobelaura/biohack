/*-------------------------------------------------------------------------
 * mnist: train an MLP 64-256-256-16 on one released MNIST draw and label its test set, in one launch.
 * The method is ~/work/mnist-et/harness/oracle.py, which is this kernel's oracle.
 *
 * Data parallel. Shire s (s-th of the launched shires) takes batch rows s*32 .. s*32+31, one per minion;
 * hart 0 of each minion works, hart 1 returns at once. Every shire keeps its own copy of the weights (W, and
 * the transposed W2T and W3T the backward pass reads) in its L2 scratchpad. A training step is:
 *
 *   R1  row:  x -> A0 (row scaled to norm sqrt(60))             G1  A0 . W1  -> Z
 *   R2  row:  A1 = relu(Z - mean)                                G2  A1 . W2  -> Z
 *   R3  row:  A2 = relu(Z - mean)                                G3  A2 . W3  -> LP (eight K slices)
 *   R4  row:  softmax, G3 = (p - onehot) / B                     G4  G3 . W3T -> Z
 *   R5  row:  D2 = centre(Z * (A2 > 0))                          G5  D2 . W2T -> Z
 *   R6  row:  D1 = centre(Z * (A1 > 0))                          G6  GP = A0'D1 | A1'D2 | A2'G3
 *   chip barrier
 *   U1  each owned 16x16 block: pull every shire's GP block (TensorLoadL2Scp), GS = their sum
 *   U2  each owned block: SGD-Nesterov on W and V, and the transposed block into W2T / W3T
 *   chip barrier
 *   U3  every shire pulls the blocks it does not own from their owners (TensorLoadL2Scp)
 *
 * with a shire barrier after every phase. Inference runs R1 G1 R2 G2 R3 G3 and an argmax over chunks of the
 * test set.
 *
 * Coherence (the L1s are not coherent and write back whole lines). Every write of shared data is an fswg.ps,
 * to an explicit shire address (the L2-local fswl.ps/flwl.ps pair trips sys_emu's coherency checker on scratchpad
 * lines, so it is not used). Shared data in this shire's scratchpad is read through the L1 (fbc.ps, flw.ps,
 * scalar loads) after the reader has evicted those lines from its own L1 (flwg.ps there costs ~1000 cycles).
 * Private staging (MN_S_PRIV, 4 KB per minion) is touched only by its minion. Data crosses shires only by
 * TensorLoadL2Scp pulls, which bypass every cache, after a chip barrier.
 *
 * No fdiv, fsqrt, double or 64-bit int<->float anywhere: the reciprocals and the square root are Newton steps,
 * exp a polynomial. Erratum 1.29 is handled in the asm (mk.inc); the scalar C is the compiler's.
 *-------------------------------------------------------------------------*/

#include <stdbool.h>
#include <stdint.h>
#include "etsoc/isa/esr_defines.h"
#include "etsoc/isa/hart.h"
#include "mnist_args.h"

int64_t entry_point(const struct MnArgs* args);

#define INLINE static inline __attribute__((always_inline))
#define NOINLINE static __attribute__((noinline))
#define FREGS                                                                                      \
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", \
        "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", \
        "f28", "f29", "f30", "f31"

#include "mk.inc"

// ---------------------------------------------------------------------------------------------- primitives

// hpmcounter3 reads 128 short when its low 7 bits are 0-10 (docs/et-soc1-notes.md). Hart 0 only.
INLINE uint32_t rdcyc(void)
{
    uint64_t v;
    __asm__ __volatile__("csrr %0, hpmcounter3" : "=r"(v));
    return (uint32_t)(v + ((uint64_t)((v & 0x7F) < 11) << 7));
}

INLINE void fence(void) { __asm__ __volatile__("fence" ::: "memory"); }

// Evict n lines from this hart's L1, leaving them in L2 (evict_va, level 1; one op covers up to 16 lines).
INLINE void evict_lines(uint64_t addr, uint64_t n)
{
    while (n) {
        const uint64_t c = n > 16 ? 16 : n;
        const uint64_t enc = (1ull << 58) | (addr & 0xFFFFFFFFFFC0ull) | (c - 1);
        register uint64_t x31 __asm__("x31") = 64;
        __asm__ __volatile__("csrw 0x89f, %1" : : "r"(x31), "r"(enc) : "memory");
        addr += c * 64;
        n -= c;
    }
}

// Before a cacheop, fence; after it and before touching the lines, wait for the cacheops (tensor_wait 6).
INLINE void evict_begin(void) { fence(); }
INLINE void evict_end(void) { __asm__ __volatile__("csrwi 0x830, 6" ::: "memory"); }

// n lines from this shire's scratchpad to private dst, through this hart's L1 (evicted from it first).
INLINE void lines_in(float* dst, uint64_t src, uint64_t n)
{
    evict_begin();
    evict_lines(src, n);
    evict_end();
    for (uint64_t i = 0; i < n; ++i, src += 64, dst += 16) {
        __asm__ __volatile__("flw.ps f28, 0(%0)\n flw.ps f29, 32(%0)\n"
                             "fmv.x.w x0, f28\n fmv.x.w x0, f29\n nop\n"
                             "fsw.ps f28, 0(%1)\n fsw.ps f29, 32(%1)\n"
                             : : "r"(src), "r"(dst) : "t0", "memory", "f28", "f29");
    }
}

// n lines from private src to dst (fswg.ps).
#define LINES_OUT(NAME, ST)                                                                              \
    INLINE void NAME(uint64_t dst, const float* src, uint64_t n)                                         \
    {                                                                                                    \
        for (uint64_t i = 0; i < n; ++i, dst += 64, src += 16) {                                         \
            __asm__ __volatile__("flw.ps f28, 0(%1)\n flw.ps f29, 32(%1)\n"                              \
                                 "fmv.x.w x0, f28\n fmv.x.w x0, f29\n nop\n"                            \
                                 ST " f28, (%0)\n addi t0, %0, 32\n " ST " f29, (t0)\n"                \
                                 : : "r"(dst), "r"(src) : "t0", "memory", "f28", "f29");                \
        }                                                                                                \
    }
LINES_OUT(lines_out, "fswg.ps")  // fswl.ps here trips sys_emu's coherency checker (see NOTES.md)
LINES_OUT(lines_out_g, "fswg.ps")

/* TensorLoadL2Scp: `lines` (1-16) lines from addr, `stride` bytes apart, into this shire's scratchpad from line
   dst (offset / 64), bypassing every cache. Waited for with tensor_wait 2 + id. */
INLINE void pull(uint64_t dst_off, uint64_t addr, uint64_t lines, uint64_t stride, uint64_t id)
{
    const uint64_t dst = dst_off / 64;
    const uint64_t v = ((dst & 0x1FFFCull) << 46) | ((dst & 3) << 4) | (addr & 0xFFFFFFFFFFC0ull) | ((lines - 1) & 0xF);
    register uint64_t x31 __asm__("x31") = (stride & 0xFFFFFFFFFFC0ull) | (id & 1);
    __asm__ __volatile__("csrw 0x85f, %1" : : "r"(x31), "r"(v) : "memory");
}
INLINE void pull_wait(uint64_t id) { __asm__ __volatile__("csrw 0x830, %0" : : "r"(2 + id) : "memory"); }

// Copy four lines, each from its own src to its own dst, the eight loads in flight together.
INLINE void copy4(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3, uint64_t d0, uint64_t d1, uint64_t d2,
                  uint64_t d3)
{
    __asm__ __volatile__(
        "flwg.ps f20, (%0)\n addi t0, %0, 32\n flwg.ps f21, (t0)\n"
        "flwg.ps f22, (%1)\n addi t0, %1, 32\n flwg.ps f23, (t0)\n"
        "flwg.ps f24, (%2)\n addi t0, %2, 32\n flwg.ps f25, (t0)\n"
        "flwg.ps f26, (%3)\n addi t0, %3, 32\n flwg.ps f27, (t0)\n"
        "fmv.x.w x0, f20\n fmv.x.w x0, f21\n fmv.x.w x0, f22\n fmv.x.w x0, f23\n"
        "fmv.x.w x0, f24\n fmv.x.w x0, f25\n fmv.x.w x0, f26\n fmv.x.w x0, f27\n"
        "fswg.ps f20, (%4)\n addi t0, %4, 32\n fswg.ps f21, (t0)\n"
        "fswg.ps f22, (%5)\n addi t0, %5, 32\n fswg.ps f23, (t0)\n"
        "fswg.ps f24, (%6)\n addi t0, %6, 32\n fswg.ps f25, (t0)\n"
        "fswg.ps f26, (%7)\n addi t0, %7, 32\n fswg.ps f27, (t0)\n"
        : : "r"(s0), "r"(s1), "r"(s2), "r"(s3), "r"(d0), "r"(d1), "r"(d2), "r"(d3)
        : "t0", "memory", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27");
}

/* SGD-Nesterov on one line (16 weights; the caller has evicted gs, w and v from its L1): g = clip(gs + wd*w); v = mu*v + g; w -= lr*(g + mu*v). Writes w and
   v back (fswg.ps) and the new w to private p. c[] = {wd, -clip, clip, mu, lr} as float bits. */
INLINE void update_line(uint64_t gs, uint64_t w, uint64_t v, float* p, const uint32_t* c)
{
    __asm__ __volatile__(
        "flw.ps f0, 0(%0)\n flw.ps f1, 32(%0)\n"
        "flw.ps f2, 0(%1)\n flw.ps f3, 32(%1)\n"
        "flw.ps f4, 0(%2)\n flw.ps f5, 32(%2)\n"
        "lw t0, 0(%4)\n fbcx.ps f10, t0\n lw t0, 4(%4)\n fbcx.ps f11, t0\n lw t0, 8(%4)\n fbcx.ps f12, t0\n"
        "lw t0, 12(%4)\n fbcx.ps f13, t0\n lw t0, 16(%4)\n fbcx.ps f14, t0\n"
        "fmv.x.w x0, f0\n fmv.x.w x0, f1\n fmv.x.w x0, f2\n fmv.x.w x0, f3\n fmv.x.w x0, f4\n fmv.x.w x0, f5\n"
        "fmv.x.w x0, f10\n fmv.x.w x0, f11\n fmv.x.w x0, f12\n fmv.x.w x0, f13\n fmv.x.w x0, f14\n"
        "fmadd.ps f0, f10, f2, f0\n fmadd.ps f1, f10, f3, f1\n fmv.x.w x0, f0\n fmv.x.w x0, f1\n"   // g + wd*w
        "fmax.ps f0, f0, f11\n fmax.ps f1, f1, f11\n fmv.x.w x0, f0\n fmv.x.w x0, f1\n"
        "fmin.ps f0, f0, f12\n fmin.ps f1, f1, f12\n fmv.x.w x0, f0\n fmv.x.w x0, f1\n"
        "fmadd.ps f4, f13, f4, f0\n fmadd.ps f5, f13, f5, f1\n fmv.x.w x0, f4\n fmv.x.w x0, f5\n"   // v
        "fmadd.ps f6, f13, f4, f0\n fmadd.ps f7, f13, f5, f1\n fmv.x.w x0, f6\n fmv.x.w x0, f7\n"   // g + mu*v
        "fnmsub.ps f2, f14, f6, f2\n fnmsub.ps f3, f14, f7, f3\n fmv.x.w x0, f2\n fmv.x.w x0, f3\n" // w - lr*()
        "nop\n"
        "fswg.ps f2, (%1)\n addi t0, %1, 32\n fswg.ps f3, (t0)\n"
        "fswg.ps f4, (%2)\n addi t0, %2, 32\n fswg.ps f5, (t0)\n"
        "fsw.ps f2, 0(%3)\n fsw.ps f3, 32(%3)\n"
        : : "r"(gs), "r"(w), "r"(v), "r"(p), "r"(c)
        : "t0", "memory", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f10", "f11", "f12", "f13", "f14");
}

// ---------------------------------------------------------------------------------------------- scalar maths

INLINE float bits_f(uint32_t u) { float f; __builtin_memcpy(&f, &u, 4); return f; }
INLINE uint32_t f_bits(float f) { uint32_t u; __builtin_memcpy(&u, &f, 4); return u; }

// 1/sqrt(x), x > 0: a bit-level guess and four Newton steps.
INLINE float rsqrt_nr(float x)
{
    float y = bits_f(0x5F3759DFu - (f_bits(x) >> 1));
    for (int i = 0; i < 4; ++i)
        y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

// 1/x, x > 0: a bit-level guess and five Newton steps.
INLINE float rcp_nr(float x)
{
    float y = bits_f(0x7EF311C3u - f_bits(x));
    for (int i = 0; i < 5; ++i)
        y = y * (2.0f - x * y);
    return y;
}

// exp(x) for x <= 0: 2^n * 2^f, 2^f by its Taylor series to degree 7 on f in [0, 1).
INLINE float exp_neg(float x)
{
    if (x < -87.0f)
        return 0.0f;
    const float t = x * 1.44269504f;
    int32_t n = (int32_t)t;  // toward zero
    if ((float)n > t)
        --n;
    const float f = (t - (float)n) * 0.693147181f;
    float p = 1.0f + f * (1.0f + f * (0.5f + f * (0.166666667f + f * (0.0416666667f + f * (0.00833333333f +
              f * (0.00138888889f + f * 0.000198412698f))))));
    return p * bits_f((uint32_t)(n + 127) << 23);
}

// The oracle's hash_uniform: u in [0, 1) for element idx of a tensor seeded with seed.
INLINE float hash_u(uint32_t seed, uint32_t idx)
{
    uint32_t h = idx * 0x9E3779B1u + seed;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return (float)(int32_t)(h >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------------------------------- context

struct Ctx {
    const struct MnArgs* a;
    uint64_t shire, sidx, nsh, j;
    uint64_t gen;      // chip barriers passed
    float* priv;       // this minion's 4 KB of scratchpad, its own
    uint32_t error;
    uint32_t phase[MN_P_COUNT];
    uint32_t dbg[8];   // G2 only: [0] evict cycles, [1] mk_f cycles, [2] mk_f calls, [3] barrier wait
};

INLINE uint64_t scp(const struct Ctx* c, uint64_t off) { return MN_SCP(c->shire, off); }

// ---------------------------------------------------------------------------------------------- barriers
// Copied from workloads/onchip: credit release, polled with a limit so a lost credit fails the run, never the card.

static bool take_credit1(uint64_t limit)
{
    for (uint64_t i = 0; i < limit; ++i) {
        uint64_t v;
        __asm__ __volatile__("csrr %0, 0xcc0" : "=r"(v) : : "memory");
        if ((v >> 16) & 0xFFFF) {
            __asm__ __volatile__("csrw 0x821, %0" : : "r"(1ull) : "memory");
            return true;
        }
    }
    return false;
}

// The 32 minions of this shire: meet in fast local barrier 30, the last one gives everyone a credit.
static bool shire_barrier(struct Ctx* c)
{
    fence();
    uint64_t last;
    __asm__ __volatile__("csrrw %0, 0x820, %1" : "=r"(last) : "r"((31ull << 5) | 30) : "memory");
    if (last)
        *((volatile uint64_t*)ESR_SHIRE(c->shire, FCC_CREDINC_0) + 1) = 0xFFFFFFFFull;
    if (!take_credit1(c->a->poll_limit)) {
        c->error = 1;
        return false;
    }
    return true;
}

static inline uint32_t amoadd_g(volatile uint32_t* p, uint32_t v)
{
    uint32_t old;
    __asm__ __volatile__("amoaddg.w %0, %1, (%2)" : "=r"(old) : "r"(v), "r"(p) : "memory");
    return old;
}

// Every launched minion: one global atomic per shire, and the shire that completes the count releases all.
static bool chip_barrier(struct Ctx* c)
{
    fence();
    uint64_t last;
    __asm__ __volatile__("csrrw %0, 0x820, %1" : "=r"(last) : "r"((31ull << 5) | 29) : "memory");
    if (last) {
        const uint32_t seen = amoadd_g((volatile uint32_t*)c->a->counter, 1) + 1;
        if (seen == (uint32_t)((c->gen + 1) * c->nsh)) {
            for (uint64_t s = 0; s < 32; ++s)
                if ((c->a->shire_mask >> s) & 1)
                    *((volatile uint64_t*)ESR_SHIRE(s, FCC_CREDINC_0) + 1) = 0xFFFFFFFFull;
        }
    }
    c->gen++;
    if (!take_credit1(c->a->poll_limit)) {
        c->error = 2;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- GEMM phases

/* out (32 x n) = A (32 x k) . B (k x n), A rows lda bytes apart, B in column panels `panel` bytes apart (each a
   k x 16 strip, rows 64 B apart), out rows ldo bytes apart; n a multiple of 16.
   Tiles of 8 rows x 16 columns, dealt round the minions. */
static void gemm_rows(struct Ctx* c, uint64_t a_off, uint64_t lda, uint64_t b, uint64_t panel, uint64_t k,
                      uint64_t o_off, uint64_t ldo, uint64_t n)
{
    const uint64_t A = scp(c, a_off), O = scp(c, o_off), ncb = n / 16;
    const bool dbg = k == MN_H && b == scp(c, MN_S_W + MN_OFF_W2 * 4);
    const uint32_t t0 = rdcyc();
    evict_begin();
    evict_lines(A, MN_ROWS * lda / 64);
    for (uint64_t t = c->j; t < 4 * ncb; t += 32)
        evict_lines(b + (t % ncb) * panel, k + 1);  // the B panel, and the row mk_f reads past it
    evict_end();
    const uint32_t t1 = rdcyc();
    for (uint64_t t = c->j; t < 4 * ncb; t += 32) {
        const uint64_t rt = t / ncb, cb = t % ncb, p = A + rt * 8 * lda;
        mk_f(p, p + lda, p + 2 * lda, p + 3 * lda, p + 4 * lda, p + 5 * lda, p + 6 * lda, p + 7 * lda,
             b + cb * panel, 64, k, O + rt * 8 * ldo + cb * 64, ldo);
        if (dbg)
            c->dbg[2]++;
    }
    if (dbg) {
        const uint32_t t2 = rdcyc();
        c->dbg[0] += t1 - t0;
        c->dbg[1] += t2 - t1;
    }
}

// LP[ks][row] = A2[row][ks*32 .. +31] . W3[ks*32 .. +31][:], 4 row tiles x 8 K slices, one tile per minion.
static void gemm_logits(struct Ctx* c)
{
    const uint64_t A = scp(c, MN_S_A2), lda = MN_H * 4;
    evict_begin();
    evict_lines(A, MN_ROWS * lda / 64);
    evict_lines(scp(c, MN_S_W + (MN_OFF_W3 + (c->j % 8) * 32 * MN_C) * 4), 33);
    evict_end();
    for (uint64_t t = c->j; t < 32; t += 32) {
        const uint64_t rt = t / 8, ks = t % 8, p = A + rt * 8 * lda + ks * 32 * 4;
        mk_f(p, p + lda, p + 2 * lda, p + 3 * lda, p + 4 * lda, p + 5 * lda, p + 6 * lda, p + 7 * lda,
             scp(c, MN_S_W + (MN_OFF_W3 + ks * 32 * MN_C) * 4), MN_C * 4, 32,
             scp(c, MN_S_LP + (ks * MN_ROWS + rt * 8) * MN_C * 4), MN_C * 4);
    }
}

// GP = A0' D1 | A1' D2 | A2' G3, each over this shire's 32 rows: 8 x 16 tiles of the weight gradient.
static void gemm_grads(struct Ctx* c)
{
    evict_begin();
    evict_lines(scp(c, MN_S_A0), MN_ROWS * MN_D * 4 / 64);
    evict_lines(scp(c, MN_S_A1), MN_ROWS * MN_H * 4 / 64);
    evict_lines(scp(c, MN_S_A2), MN_ROWS * MN_H * 4 / 64);
    evict_lines(scp(c, MN_S_D1), MN_ROWS * MN_H * 4 / 64 + 16);
    evict_lines(scp(c, MN_S_D2), MN_ROWS * MN_H * 4 / 64 + 16);
    evict_lines(scp(c, MN_S_G3), MN_ROWS + 1);
    evict_end();
    // layer: rows, column blocks, activation (and its row bytes), delta (and its row bytes), offset, panel floats
    static const uint32_t L[3][8] = {
        {MN_D, 16, MN_S_A0, MN_D * 4, MN_S_D1, MN_H * 4, MN_OFF_W1, MN_D * 16},
        {MN_H, 16, MN_S_A1, MN_H * 4, MN_S_D2, MN_H * 4, MN_OFF_W2, MN_H * 16},
        {MN_H, 1, MN_S_A2, MN_H * 4, MN_S_G3, MN_C * 4, MN_OFF_W3, MN_H * 16},
    };
    uint64_t t0 = 0;
    for (int l = 0; l < 3; ++l) {
        const uint64_t ncb = L[l][1], tiles = (L[l][0] / 8) * ncb;
        for (uint64_t t = (c->j + 32 - t0 % 32) % 32; t < tiles; t += 32) {
            const uint64_t rt = t / ncb, cb = t % ncb;
            mk_g(scp(c, L[l][2]) + rt * 8 * 4, L[l][3], scp(c, L[l][4]) + cb * 64, L[l][5], MN_ROWS,
                 scp(c, MN_S_GP + (L[l][6] + cb * L[l][7] + rt * 8 * 16) * 4), 64);
        }
        t0 += tiles;
    }
}

// ---------------------------------------------------------------------------------------------- row phases

// R1: this minion's row of A0: x (60 floats, read-only input) padded to 64 and scaled to norm sqrt(60).
static void row_input(struct Ctx* c, const float* x, bool active)
{
    float* r = c->priv;
    float ss = 0.0f;
    for (int i = 0; i < MN_D; ++i) {
        r[i] = (active && i < MN_DIN) ? x[i] : 0.0f;
        ss += r[i] * r[i];
    }
    const float s = 7.74596669f * rsqrt_nr(ss > 1e-20f ? ss : 1e-20f);  // sqrt(60 / ss)
    for (int i = 0; i < MN_D; ++i)
        r[i] *= s;
    lines_out(scp(c, MN_S_A0 + c->j * MN_D * 4), r, MN_D / 16);
}

// R2, R3: dst row = relu(Z row - its mean).
static void row_act(struct Ctx* c, uint64_t dst_off)
{
    float* z = c->priv;
    lines_in(z, scp(c, MN_S_Z + c->j * MN_H * 4), MN_H / 16);
    float m = 0.0f;
    for (int i = 0; i < MN_H; ++i)
        m += z[i];
    m *= 1.0f / MN_H;
    for (int i = 0; i < MN_H; ++i) {
        const float v = z[i] - m;
        z[i] = v > 0.0f ? v : 0.0f;
    }
    lines_out(scp(c, dst_off + c->j * MN_H * 4), z, MN_H / 16);
}

// The row's logits: the eight K-slice partials summed.
static void row_logits(struct Ctx* c, float* lg)
{
    float* p = c->priv + 512;
    lines_in(p, scp(c, MN_S_LP + c->j * MN_C * 4), 1);
    for (int i = 0; i < MN_C; ++i)
        lg[i] = p[i];
    for (int ks = 1; ks < 8; ++ks) {
        lines_in(p, scp(c, MN_S_LP + (ks * MN_ROWS + c->j) * MN_C * 4), 1);
        for (int i = 0; i < MN_C; ++i)
            lg[i] += p[i];
    }
}

// R4 (training): G3 row = (softmax(logits) - onehot(y)) / B, or zeros for a row past the batch.
static void row_softmax(struct Ctx* c, int32_t y, bool active, float inv_b)
{
    float lg[MN_C];
    row_logits(c, lg);
    float* g = c->priv;
    for (int i = 0; i < MN_C; ++i)
        g[i] = 0.0f;
    if (active) {
        float m = lg[0];
        for (int i = 1; i < MN_NC; ++i)
            m = lg[i] > m ? lg[i] : m;
        float s = 0.0f;
        for (int i = 0; i < MN_NC; ++i) {
            g[i] = exp_neg(lg[i] - m);
            s += g[i];
        }
        const float r = rcp_nr(s);
        for (int i = 0; i < MN_NC; ++i)
            g[i] = (g[i] * r - (i == y ? 1.0f : 0.0f)) * inv_b;
    }
    lines_out(scp(c, MN_S_G3 + c->j * MN_C * 4), g, 1);
}

// R5, R6: dst row = centre(Z row * (act row > 0)).
static void row_delta(struct Ctx* c, uint64_t act_off, uint64_t dst_off)
{
    float* z = c->priv;
    float* a = c->priv + MN_H;
    lines_in(z, scp(c, MN_S_Z + c->j * MN_H * 4), MN_H / 16);
    lines_in(a, scp(c, act_off + c->j * MN_H * 4), MN_H / 16);
    float m = 0.0f;
    for (int i = 0; i < MN_H; ++i) {
        z[i] = a[i] > 0.0f ? z[i] : 0.0f;
        m += z[i];
    }
    m *= 1.0f / MN_H;
    for (int i = 0; i < MN_H; ++i)
        z[i] -= m;
    lines_out(scp(c, dst_off + c->j * MN_H * 4), z, MN_H / 16);
}

// ---------------------------------------------------------------------------------------------- ownership

INLINE uint64_t owner_sidx(const struct Ctx* c, uint32_t blk) { return blk % c->nsh; }

/* Weights are stored in column panels: every 16 columns of a matrix are one contiguous K x 16 strip, so a 16 x 16
   block is 1 KB of consecutive lines. Block b's float offset in W (W1 blocks (br, bc) = b, W2 blocks 64 + br*16 + bc,
   W3 blocks 320 + br): */
INLINE uint32_t block_off(uint32_t b)
{
    if (b < MN_BLK_W2)
        return MN_OFF_W1 + (b % 16) * MN_D * 16 + (b / 16) * 256;
    if (b < MN_BLK_W3)
        return MN_OFF_W2 + ((b - MN_BLK_W2) % 16) * MN_H * 16 + ((b - MN_BLK_W2) / 16) * 256;
    return MN_OFF_W3 + (b - MN_BLK_W3) * 256;
}

// The float offset in T of the transpose of W2 or W3 block b: W2 (br, bc) -> W2T (bc, br); W3 br -> W3T (0, br).
INLINE uint32_t tblock_off(uint32_t b)
{
    if (b < MN_BLK_W3)
        return ((b - MN_BLK_W2) / 16) * MN_H * 16 + ((b - MN_BLK_W2) % 16) * 256;
    return MN_OFF_W3T + (b - MN_BLK_W3) * 256;
}

/* U1a: pull every shire's partial gradient of every block this shire owns into staging slot
   (k * nsh + p), 1 KB each, k the block's rank among this shire's blocks. One TensorLoadL2Scp per (block, shire). */
static void pull_grads(struct Ctx* c)
{
    uint32_t k = 0, n = 0;
    for (uint32_t b = (uint32_t)c->sidx; b < MN_BLOCKS; b += (uint32_t)c->nsh, ++k) {
        const uint32_t off = block_off(b);
        for (uint32_t p = 0; p < c->nsh; ++p) {
            const uint32_t pair = k * (uint32_t)c->nsh + p;
            if (pair % 32 != c->j)
                continue;
            pull(MN_S_STAGE + pair * 1024ull, MN_SCP(p, MN_S_GP + off * 4ull), 16, 64, n & 1);
            if (++n % 2 == 0) {
                pull_wait(0);
                pull_wait(1);
            }
        }
    }
    pull_wait(0);
    pull_wait(1);
}

// U1b: GS line = sum of the nsh staged copies of that line.
static void sum_grads(struct Ctx* c)
{
    uint32_t k = 0, rank = 0;
    for (uint32_t b = (uint32_t)c->sidx; b < MN_BLOCKS; b += (uint32_t)c->nsh, ++k) {
        const uint32_t off = block_off(b);
        const uint64_t stage = scp(c, MN_S_STAGE + k * c->nsh * 1024ull);
        bool evicted = false;
        for (uint32_t r = 0; r < 16; ++r) {
            if (rank++ % 32 != c->j)
                continue;
            if (!evicted) {
                evict_begin();
                evict_lines(stage, c->nsh * 16);
                evict_end();
                evicted = true;
            }
            sum_line(scp(c, MN_S_GS + off * 4ull + r * 64), scp(c, MN_S_STAGE + k * c->nsh * 1024ull + r * 64),
                     1024, c->nsh);
        }
    }
}

// U2: the update of this minion's blocks, and their transposes.
static void update_blocks(struct Ctx* c)
{
    const struct MnArgs* a = c->a;
    const uint32_t k[5] = {f_bits(a->wd), f_bits(-a->clip), f_bits(a->clip), f_bits(a->mu), f_bits(a->lr)};
    float* nw = c->priv;           // the new block, 16 x 16
    float* tw = c->priv + 256;     // its transpose
    for (uint32_t b = 0; b < MN_BLOCKS; ++b) {
        if (owner_sidx(c, b) != c->sidx || (b / c->nsh) % 32 != c->j)
            continue;
        const uint32_t off = block_off(b);
        evict_begin();
        evict_lines(scp(c, MN_S_GS + off * 4ull), 16);
        evict_lines(scp(c, MN_S_W + off * 4ull), 16);
        evict_lines(scp(c, MN_S_V + off * 4ull), 16);
        evict_end();
        for (uint32_t r = 0; r < 16; ++r) {
            const uint64_t e = (off + r * 16) * 4ull;
            update_line(scp(c, MN_S_GS + e), scp(c, MN_S_W + e), scp(c, MN_S_V + e), nw + r * 16, k);
        }
        if (b < MN_BLK_W2)
            continue;
        for (uint32_t r = 0; r < 16; ++r)
            for (uint32_t q = 0; q < 16; ++q)
                tw[q * 16 + r] = nw[r * 16 + q];
        lines_out_g(scp(c, MN_S_T + tblock_off(b) * 4ull), tw, 16);
    }
}

// U3: pull every block this shire does not own (and its transpose) from its owner, one TensorLoadL2Scp each.
static void gather_weights(struct Ctx* c)
{
    uint32_t rank = 0, n = 0;
    for (uint32_t b = 0; b < MN_BLOCKS; ++b) {
        const uint64_t o = owner_sidx(c, b);
        if (o == c->sidx)
            continue;
        for (int t = 0; t < (b < MN_BLK_W2 ? 1 : 2); ++t) {
            if (rank++ % 32 != c->j)
                continue;
            const uint64_t src = t ? MN_S_T + tblock_off(b) * 4ull : MN_S_W + block_off(b) * 4ull;
            pull(src, MN_SCP(o, src), 16, 64, n & 1);
            if (++n % 2 == 0) {
                pull_wait(0);
                pull_wait(1);
            }
        }
    }
    pull_wait(0);
    pull_wait(1);
}

// Every shire builds the same initial weights from the hash (indexed row-major, as the oracle), and zero momentum.
static void init_weights(struct Ctx* c)
{
    const struct MnArgs* a = c->a;
    float* line = c->priv;
    for (uint32_t l = c->j; l < MN_W_LINES; l += 32) {
        // line l -> matrix row r, columns q0 .. q0+15
        uint32_t r, q0, which;
        if (l < MN_OFF_W2 / 16) {
            which = 1; q0 = (l / MN_D) * 16; r = l % MN_D;
        } else if (l < MN_OFF_W3 / 16) {
            which = 2; q0 = ((l - MN_OFF_W2 / 16) / MN_H) * 16; r = (l - MN_OFF_W2 / 16) % MN_H;
        } else {
            which = 3; q0 = 0; r = l - MN_OFF_W3 / 16;
        }
        for (uint32_t q = 0; q < 16; ++q) {
            float v = 0.0f;
            if (which == 1 && r < MN_DIN)
                v = (2.0f * hash_u(a->seed1, r * MN_H + q0 + q) - 1.0f) * a->bound1;
            else if (which == 2)
                v = (2.0f * hash_u(a->seed2, r * MN_H + q0 + q) - 1.0f) * a->bound2;
            line[q] = v;
        }
        lines_out(scp(c, MN_S_W + l * 64), line, 1);
        for (uint32_t q = 0; q < 16; ++q)
            line[q] = 0.0f;
        lines_out(scp(c, MN_S_V + l * 64), line, 1);
    }
    for (uint32_t l = c->j; l < MN_T_LINES; l += 32) {
        for (uint32_t q = 0; q < 16; ++q) {
            float v = 0.0f;
            if (l < MN_OFF_W3T / 16) {  // W2T row r (= W2 column), column q0 + q (= W2 row)
                const uint32_t r = l % MN_H, q0 = (l / MN_H) * 16;
                v = (2.0f * hash_u(a->seed2, (q0 + q) * MN_H + r) - 1.0f) * a->bound2;
            }
            line[q] = v;
        }
        lines_out(scp(c, MN_S_T + l * 64), line, 1);
    }
}

// ---------------------------------------------------------------------------------------------- the run

#define PHASE(c, p, call)                        \
    do {                                         \
        const uint32_t _t = rdcyc();             \
        call;                                    \
        if (!shire_barrier(c))                   \
            return false;                        \
        (c)->phase[p] += rdcyc() - _t;           \
    } while (0)

// The forward pass up to the logit partials, for rows off .. off+rows-1 of x (n x 60).
static bool forward(struct Ctx* c, const float* x, uint32_t off, uint32_t rows)
{
    const uint32_t b = (uint32_t)(c->sidx * MN_ROWS + c->j);
    const bool active = b < rows;
    PHASE(c, MN_P_R1, row_input(c, x + (uint64_t)(off + (active ? b : 0)) * MN_DIN, active));
    PHASE(c, MN_P_G1, gemm_rows(c, MN_S_A0, MN_D * 4, scp(c, MN_S_W + MN_OFF_W1 * 4), MN_D * 64, MN_D, MN_S_Z,
                                MN_H * 4, MN_H));
    PHASE(c, MN_P_R2, row_act(c, MN_S_A1));
    PHASE(c, MN_P_G2, gemm_rows(c, MN_S_A1, MN_H * 4, scp(c, MN_S_W + MN_OFF_W2 * 4), MN_H * 64, MN_H, MN_S_Z,
                                MN_H * 4, MN_H));
    PHASE(c, MN_P_R3, row_act(c, MN_S_A2));
    PHASE(c, MN_P_G3, gemm_logits(c));
    return true;
}

static bool train_step(struct Ctx* c, uint32_t t)
{
    const struct MnArgs* a = c->a;
    const uint32_t off = (t * a->batch) % a->n_train;
    const uint32_t rows = a->n_train - off < a->batch ? a->n_train - off : a->batch;
    const uint32_t b = (uint32_t)(c->sidx * MN_ROWS + c->j);
    const bool active = b < rows;
    const int32_t y = active ? ((const int32_t*)a->train_y)[off + b] : -1;
    if (!forward(c, (const float*)a->train_x, off, rows))
        return false;
    PHASE(c, MN_P_R4, row_softmax(c, y, active, rcp_nr((float)(int32_t)rows)));
    PHASE(c, MN_P_G4, gemm_rows(c, MN_S_G3, MN_C * 4, scp(c, MN_S_T + MN_OFF_W3T * 4), MN_C * 64, MN_C, MN_S_Z,
                                MN_H * 4, MN_H));
    PHASE(c, MN_P_R5, row_delta(c, MN_S_A2, MN_S_D2));
    PHASE(c, MN_P_G5, gemm_rows(c, MN_S_D2, MN_H * 4, scp(c, MN_S_T), MN_H * 64, MN_H, MN_S_Z, MN_H * 4, MN_H));
    PHASE(c, MN_P_R6, row_delta(c, MN_S_A1, MN_S_D1));
    {
        const uint32_t _t = rdcyc();
        gemm_grads(c);
        if (!chip_barrier(c))
            return false;
        c->phase[MN_P_G6] += rdcyc() - _t;
    }
    PHASE(c, MN_P_U1, pull_grads(c));
    PHASE(c, MN_P_U1, sum_grads(c));
    {
        const uint32_t _t = rdcyc();
        update_blocks(c);
        if (!chip_barrier(c))
            return false;
        c->phase[MN_P_U2] += rdcyc() - _t;
    }
    PHASE(c, MN_P_U3, gather_weights(c));
    return true;
}

static bool infer(struct Ctx* c)
{
    const struct MnArgs* a = c->a;
    const uint32_t chunk = (uint32_t)(c->nsh * MN_ROWS);
    for (uint32_t off = 0; off < a->n_test; off += chunk) {
        const uint32_t rows = a->n_test - off < chunk ? a->n_test - off : chunk;
        const uint32_t b = (uint32_t)(c->sidx * MN_ROWS + c->j);
        if (!forward(c, (const float*)a->test_x, off, rows))
            return false;
        const uint32_t t0 = rdcyc();
        if (b < rows) {
            float lg[MN_C];
            row_logits(c, lg);
            int32_t best = 0;
            for (int i = 1; i < MN_NC; ++i)
                best = lg[i] > lg[best] ? i : best;
            int32_t* rec = (int32_t*)c->priv;
            for (int i = 0; i < 16; ++i)
                rec[i] = i == 0 ? best : 0;
            lines_out_g(a->labels + (uint64_t)(off + b) * 64, (const float*)rec, 1);
        }
        if (!shire_barrier(c))
            return false;
        c->phase[MN_P_INF] += rdcyc() - t0;
    }
    return true;
}

static void put_stats(const struct Ctx* c, uint32_t steps_done, uint32_t total, uint32_t train)
{
    struct MnStats* s = (struct MnStats*)(c->priv + 768);
    __builtin_memset(s, 0, sizeof(*s));
    s->magic = MN_MAGIC;
    s->error = c->error;
    s->minion = (uint32_t)(c->shire * 32 + c->j);
    s->steps_done = steps_done;
    s->total_cycles = total;
    s->train_cycles = train;
    for (int i = 0; i < MN_P_COUNT; ++i)
        s->phase[i] = c->phase[i];
    for (int i = 0; i < 8; ++i)
        s->pad[i] = c->dbg[i];
    lines_out_g(c->a->stats + (c->shire * 32 + c->j) * sizeof(struct MnStats), (const float*)s, 2);
}

int64_t entry_point(const struct MnArgs* args)
{
    const uint64_t hart = get_hart_id();
    const uint64_t shire = hart >> 6;
    if ((hart & 1) || shire >= 32 || !((args->shire_mask >> shire) & 1))
        return 0;
    __asm__ __volatile__("mov.m.x m0, zero, 0xff" ::: "memory");  // every vector lane on

    struct Ctx c;
    __builtin_memset(&c, 0, sizeof(c));
    c.a = args;
    c.shire = shire;
    c.j = (hart >> 1) & 31;
    c.nsh = (uint64_t)__builtin_popcountll(args->shire_mask);
    c.sidx = (uint64_t)__builtin_popcountll(args->shire_mask & ((1ull << shire) - 1));
    c.priv = (float*)MN_SCP(shire, MN_S_PRIV + c.j * 4096);

    const uint32_t t_start = rdcyc();
    uint32_t t = 0;
    init_weights(&c);
    bool ok = shire_barrier(&c);
    for (; ok && t < args->steps; ++t)
        ok = train_step(&c, t);
    const uint32_t t_train = rdcyc();
    if (ok)
        ok = infer(&c);
    if (ok && args->wout && c.sidx == 0) {
        for (uint32_t l = c.j; l < MN_W_LINES; l += 32)
            copy4(scp(&c, MN_S_W + l * 64), scp(&c, MN_S_W + l * 64), scp(&c, MN_S_W + l * 64),
                  scp(&c, MN_S_W + l * 64), args->wout + l * 64, args->wout + l * 64, args->wout + l * 64,
                  args->wout + l * 64);
    }
    const uint32_t t_end = rdcyc();
    put_stats(&c, t, t_end - t_start, t_train - t_start);
    fence();
    return 0;
}
