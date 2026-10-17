# First hour, aifoundry3, 16 Oct 2026 (PDT)
- Card chosen: aifoundry3 card 0. Link 16 GT/s, et-who --check exit 0, no queues/other card programs, et-usage 30m shows only the live monitor.
- Clone: ~/et-soc1-prototyping @ f0a4087 (sparse, 38 MB).
- Step 3 simulator smoke (it_test_code_loading --mode=sysemu): PASSED 3 tests, 106 s.
- Step 5 sgemm sysemu -n 64: mismatches 0/4096, PASS.
- NEXT: steps 4, 6, 7 on card 0 need the person's yes first (not yet given).
- Person said yes to card runs (16 Oct ~13:15 PDT).
- Step 4 card smoke (--mode=pcie): PASSED 3 tests, 703 ms. Log smoke-pcie.log.
- Step 6 sgemm n=512 reps 3 on card 0: 2.34/2.37/2.40 ms per launch (~113 GFLOP/s), mismatches 0/262144, PASS. out.txt, manifest.txt, when.txt.
- Step 7 ettelem idle 5 s: idle.jsonl (50 samples). Card idle, lock released.
