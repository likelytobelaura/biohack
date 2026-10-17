# hello on aifoundry3 card 0, 16 Oct 2026 (PDT)
Code: ~/work/hello (kernel/hello.c, host/main.cpp, hello_args.h; build files copied from workloads/sgemm).
Each hart writes one 64 B record (magic, hart, shire, minion, thread, neighbourhood, launch, loop cycles, loop result);
thread 0 of each minion times 1000 iterations of addi/addiw/bnez with hpmcounter3 (fixcyc applied).

sysemu/ : --shires 0x3 --reps 2: 128/128 harts, PASS; loop 3007/3135 cycles (simulator: not timing-accurate). Boot 32 s.
card/   : telemetry before (24.0-24.5 W, 53 C, 600 MHz), manifest, et-who/et-usage/dmesg before and after.
  all 32 shires, 3 launches: 2048/2048 harts each, 0 bad, PASS; 0.206-0.208 ms per launch; loop 14,065-14,393 cycles, median 14,233.
  shire 0 only, 3 launches: 64/64 each, PASS; 0.56 / 0.16 / 0.56 ms per launch (runtime's ~500 us idle poll); loop median 14,231-14,238.
  device open about 185-190 ms per process. No new ET kernel messages; et-who empty after.
Reading: ~14.2 cycles per iteration on silicon vs ~3 in sysemu. Energy manual 3.1: taken branch issues 0.058/hart/cycle
(both harts busy), adds 0.37, so the taken branch dominates the loop. Not yet tested directly (e.g. by unrolling).
