// Kernel arguments shared by the device kernel (riscv64 gcc) and the host (g++).
#pragma once
#include <stdint.h>

struct AatArgs {
  uint64_t a;           // device address of A[m][k], row-major fp32
  uint64_t b;           // device address of B[k][n], row-major fp32 (the host passes B = A^T)
  uint64_t c;           // device address of C[m][n] (output)
  uint64_t shire_mask;  // shires the kernel was launched on (to number the harts)
  uint32_t m;           // rows of A and C
  uint32_t n;           // columns of B and C, multiple of 16
  uint32_t k;           // columns of A, rows of B
  uint32_t pad;
};
