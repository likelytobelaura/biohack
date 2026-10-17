# MNIST end-to-end on the ET-SoC-1 (port of cybertronai/sutro-problems mnist-a100)
Started 16 Oct 2026 on aifoundry3 card 0. Person said "go and do it all".
Reference: ~/work/ref/sutro-mnist-a100/mnist.py (rules in its docstring), README.md (leaderboard).

Plan
1. Harness (no card): download MNIST/Fashion/KMNIST to ~/data/sutro-mnist; draws made with mnist.py's own
   pipeline (split halves, PCA-whiten to 60, Haar rotation, label permutation); 11 MNIST + 4 hold-out calls;
   score with mnist.py's judge rules. NumPy reference method to fix the algorithm.
2. Device MLP 60-256-256-10, full-batch training; simulator first, then card 0 under lock, timeout 10.
3. Speed: scratchpad, tensor unit, one launch.
4. Scored run, time (with/without PCIe copy) and energy above idle (ettelem).
Timing convention: device open + kernel load in the untimed warm-up (like compile/graph capture on A100).

Log
- 13:40 Harness: harness/release.py (mnist.py's draw code, copied; make/score), data in ~/data/sutro-mnist (53 MB),
  run dirs ~/data/mnist-et/runs/devN. NumPy sweeps (harness/reference.py, explore.py): plain full-batch Adam ~5.6% at 60
  steps; record tricks (row norm, centred ReLU, SGD-Nesterov lr .4 mu .95 wd .001, clip .03, batch 1024) work.
- Method fixed = harness/oracle.py: MLP 64-256-256-16(10), 100 steps, hash-uniform init. dev1 4.67% (worst 5.33),
  dev2 4.58% (worst 4.92). Difficulty 1 band 5.40%. ~52 GFLOP/call.
- Device design (v1): one launch per call, hart 0 only. Data parallel: shire = 32 batch rows (1/minion); each shire
  holds W (+W2T) in its scratchpad. Per step: row-wise phases (scalar) + col-sliced vector GEMMs (fbc.ps/flwl.ps,
  8x16 tiles) with shire barriers (FLB+credit), partial grads -> chip barrier -> line-level sum across shires
  (flwg.ps) -> block owner updates (16x16 blocks, also writes W2T) -> chip barrier -> all-gather (flwg.ps->fswl.ps).
  Coherence rule: shared writes only fswl.ps; shared reads flwl/flwg, or L1-path after evict_va (level L2).
  Barriers copied from workloads/onchip (credit release, polled with a limit).
- 14:05 v1 (row-major, flwg/fswg everywhere, remote flwg exchange) on card 0: 1 call 425 ms (train 418 ms), error 4.54%
  (oracle 4.51%), 99.12% label agreement. 2 steps: rel diff 2e-4 (W2) vs oracle; card bit-reproducible over 3 runs.
  Profile: U3 gather 50%, U1 sum 25%, GEMMs ~17%.
- 14:12 v2: weights in column panels (16x16 block = 1 KB contiguous), exchange by TensorLoadL2Scp (1 pull / block),
  gradient staging MN_S_STAGE. Tried flwl/fswl for local data: sys_emu Mem-Checker FATAL "Coherency Read Hazard" on
  flwl.ps of W lines written by other minions' fswl.ps -> reverted to flwg/fswg for all shared vector data.
- 14:30 v3 (evict-then-L1 reads): no faster (113 ms). Instrumented G2: inner vector loop ~5.8 cycles/instr, 0.58 MAC/cycle;
  evicts negligible. Tensor unit (~7.7 MAC/cycle) is the next step if speed matters; person said speed is not the goal.
- 32-shire sysemu run of v1, 2 steps: weights bit-identical to the card's. Card result = simulator result.
- 14:40 SCORED RUN score1 (seed 20261016, hold-out kmnist): difficulty 1 PASS. MNIST 95.36% (4.64% error, worst draw
  5.24%), kmnist 91.84%, 111.3 ms/call end to end (kernel 109 ms, copy in 1.5, out 0.8); device open 185 ms untimed.
  Files: ~/runs/mnist/score1 (run.log, score.txt, manifest), labels/times in ~/data/mnist-et/runs/score1/dev.
- 14:33 Person asked to release the card: et-who empty, no jobs of ours running. They post "released" in #community-lab.
