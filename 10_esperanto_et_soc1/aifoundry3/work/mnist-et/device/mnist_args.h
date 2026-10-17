// mnist: an MLP 64-256-256-16 trained end to end on the ET-SoC-1, one launch per call.
// Shared by the device kernel (riscv64 gcc) and the host (g++). Plain fixed-width types only.
#pragma once
#include <stdint.h>

#define MN_DIN 60    // input features as released
#define MN_D 64      // padded input width
#define MN_H 256     // hidden width
#define MN_C 16      // padded output width (10 classes)
#define MN_NC 10
#define MN_ROWS 32   // batch rows per shire, one per minion

// Weights, in floats, in one buffer, each matrix ([in][out]) stored as column panels: columns 16c .. 16c+15 of all
// rows are one contiguous strip. The same layout in W, V, GP (partial gradient) and GS (summed gradient).
#define MN_OFF_W1 0                                // 64 x 256
#define MN_OFF_W2 (MN_D * MN_H)                    // 256 x 256
#define MN_OFF_W3 (MN_OFF_W2 + MN_H * MN_H)        // 256 x 16
#define MN_W_FLOATS (MN_OFF_W3 + MN_H * MN_C)      // 86016
#define MN_W_LINES (MN_W_FLOATS / 16)              // 5376
// Transposed copies the backward pass reads: W2T (256 x 256) and W3T (16 x 256), one buffer.
#define MN_OFF_W3T (MN_H * MN_H)
#define MN_T_FLOATS (MN_OFF_W3T + MN_C * MN_H)     // 69632
#define MN_T_LINES (MN_T_FLOATS / 16)
// Update ownership is by 16x16 block: W1 4x16 blocks, W2 16x16, W3 16x1.
#define MN_BLK_W2 64
#define MN_BLK_W3 (64 + 256)
#define MN_BLOCKS (64 + 256 + 16)                  // 336

// Scratchpad layout of every compute shire (byte offsets; Format 0 address 0x80000000 + (shire << 23) + off).
// Offset 0 faults, so everything starts 256 KB in. Each B operand has a readable line after it.
#define MN_SCP(shire, off) (0x80000000ull + ((uint64_t)(shire) << 23) + (uint64_t)(off))
#define MN_S_W 0x40000ull                                  // W, this shire's copy
#define MN_S_T (MN_S_W + MN_W_FLOATS * 4 + 1024)           // W2T, W3T
#define MN_S_V (MN_S_T + MN_T_FLOATS * 4 + 1024)           // momentum (owned blocks only)
#define MN_S_GP (MN_S_V + MN_W_FLOATS * 4)                 // this shire's partial gradient
#define MN_S_GS (MN_S_GP + MN_W_FLOATS * 4 + 1024)         // summed gradient (owned blocks only)
#define MN_S_A0 (MN_S_GS + MN_W_FLOATS * 4)                // 32 x 64   inputs
#define MN_S_A1 (MN_S_A0 + MN_ROWS * MN_D * 4)             // 32 x 256  layer 1 activations
#define MN_S_A2 (MN_S_A1 + MN_ROWS * MN_H * 4)             // 32 x 256  layer 2 activations
#define MN_S_G3 (MN_S_A2 + MN_ROWS * MN_H * 4)             // 32 x 16   output delta
#define MN_S_D2 (MN_S_G3 + MN_ROWS * MN_C * 4 + 1024)      // 32 x 256  layer 2 delta
#define MN_S_D1 (MN_S_D2 + MN_ROWS * MN_H * 4 + 1024)      // 32 x 256  layer 1 delta
#define MN_S_Z (MN_S_D1 + MN_ROWS * MN_H * 4 + 1024)       // 32 x 256  every GEMM's output
#define MN_S_LP (MN_S_Z + MN_ROWS * MN_H * 4)              // 8 x 32 x 16 partial logits
#define MN_S_PRIV (MN_S_LP + 8 * MN_ROWS * MN_C * 4)       // 4 KB per minion, its own
#define MN_S_STAGE (MN_S_PRIV + MN_ROWS * 4096)       // gradient pulls: owned blocks x shires, 1 KB each
#define MN_S_END (MN_S_STAGE + 368 * 1024)
#define MN_SCP_BYTES 0x280000ull

#define MN_MAGIC 0x4D4E4554u  // "MNET"
#define MN_MAX_MINIONS 1024

// Phases, for the per-minion cycle counts.
enum { MN_P_R1, MN_P_G1, MN_P_R2, MN_P_G2, MN_P_R3, MN_P_G3, MN_P_R4, MN_P_G4, MN_P_R5, MN_P_G5, MN_P_R6,
       MN_P_G6, MN_P_U1, MN_P_U2, MN_P_U3, MN_P_INF, MN_P_COUNT };

struct MnArgs {
  uint64_t train_x;     // n_train x 60 f32, as released
  uint64_t train_y;     // n_train i32
  uint64_t test_x;      // n_test x 60 f32
  uint64_t labels;      // out: n_test x 16 i32, one line per row, [0] the label
  uint64_t wout;        // out, if nonzero: shire 0's final W (MN_W_FLOATS f32)
  uint64_t stats;       // out: MnStats[MN_MAX_MINIONS], by minion
  uint64_t counter;     // a zeroed DRAM line for the chip barrier's atomic
  uint64_t shire_mask;
  uint64_t poll_limit;  // barrier give-up count, so a lost credit fails the run and not the card
  uint32_t n_train, n_test, steps, batch;
  float lr, mu, wd, clip;
  float bound1, bound2;  // initial weights: uniform(-b, b)
  uint32_t seed1, seed2;
};

// One cache line per minion... and more: 64 B header line plus the phase cycle sums.
struct MnStats {
  uint32_t magic, error, minion, steps_done;
  uint32_t total_cycles, train_cycles, pad0, pad1;
  uint32_t phase[MN_P_COUNT];  // cycles summed over the run, hart 0
  uint32_t pad[8];
};

#ifdef __cplusplus
static_assert(sizeof(MnStats) == 128, "MnStats is two lines");
static_assert(MN_S_END <= MN_SCP_BYTES, "scratchpad layout overflows 2.5 MB");
#endif
