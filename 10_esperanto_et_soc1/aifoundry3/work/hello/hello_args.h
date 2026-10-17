// Kernel arguments shared by the device kernel (riscv64 gcc) and the host (g++).
#pragma once
#include <stdint.h>

#define HELLO_MAGIC 0x48454C4Fu  // "HELO"
#define HELLO_WORDS 16           // one 64-byte line per hart
#define HELLO_LOOP 1000          // dependent adds timed by thread 0 of each minion

struct HelloArgs {
  uint64_t out;         // device address of 2048 x 64-byte records
  uint64_t shire_mask;  // shires the kernel was launched on
  uint32_t launch;      // launch index, echoed back
  uint32_t pad;
};
