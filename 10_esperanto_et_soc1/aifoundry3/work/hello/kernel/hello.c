/*-------------------------------------------------------------------------
 * hello: every hart writes one 64-byte record about itself, the host checks
 * that every hart of the launched shires reported in.
 *
 * Each record is a whole cache line, so the non-coherent L1s never collide.
 * Thread 0 of each minion also times a chain of dependent adds with
 * hpmcounter3; only one thread per minion reads it (erratum 1.23).
 *-------------------------------------------------------------------------*/

#include <stdint.h>
#include "etsoc/isa/hart.h"
#include "hello_args.h"

// hpmcounter3 reads 128 short when its low 7 bits are 0-10 (docs/et-soc1-notes.md).
static inline uint64_t fixcyc(uint64_t v)
{
    return v + ((uint64_t)((v & 0x7F) < 11) << 7);
}

static inline uint64_t rdcyc(void)
{
    uint64_t v;
    __asm__ volatile("csrr %0, hpmcounter3" : "=r"(v));
    return fixcyc(v);
}

int64_t entry_point(const struct HelloArgs* args);

int64_t entry_point(const struct HelloArgs* args)
{
    const uint64_t hart = get_hart_id();
    const uint64_t shire = hart >> 6;
    if (shire >= 32 || !((args->shire_mask >> shire) & 1))
        return 0;
    const uint32_t minion = (uint32_t)(hart >> 1) & 31;
    const uint32_t thread = (uint32_t)hart & 1;

    uint32_t loop_cycles = 0;
    uint64_t x = hart;
    if (thread == 0) {
        const uint64_t t0 = rdcyc();
        for (int i = 0; i < HELLO_LOOP; ++i)
            __asm__ volatile("addi %0, %0, 1" : "+r"(x));
        const uint64_t t1 = rdcyc();
        loop_cycles = (uint32_t)(t1 - t0);
    }

    volatile uint32_t* rec = (volatile uint32_t*)args->out + hart * HELLO_WORDS;
    rec[0] = HELLO_MAGIC;
    rec[1] = (uint32_t)hart;
    rec[2] = (uint32_t)shire;
    rec[3] = minion;
    rec[4] = thread;
    rec[5] = minion >> 3;  // neighbourhood
    rec[6] = args->launch;
    rec[7] = loop_cycles;
    rec[8] = (uint32_t)(x - hart);  // must equal HELLO_LOOP for thread 0
    for (int i = 9; i < HELLO_WORDS; ++i)
        rec[i] = 0;
    return 0;
}
