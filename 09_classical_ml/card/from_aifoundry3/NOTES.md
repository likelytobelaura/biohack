# S = A·Aᵀ for the CITE-seq matrix on aifoundry3 card 0, 17 Oct 2026

Input `cite_A.f32` (8656 x 400 fp32, SHA-256 502eae5c…a173ef, checked). Output `cite_S_card.f32`, 8656 x 8656 fp32
row-major, 299,705,344 bytes, SHA-256 6dd20841c7e6c86ba2956a1997aa9486ef946248ff0a013447ba2a637e05d890.

## Result (card run 14:53 PDT, 32 shires / 2048 harts, 3 launches)

| | |
|---|---|
| time per launch | 373.8, 373.2, 373.3 ms (kernelLaunch → waitForStream, host-timed, launch overhead included) |
| GFLOP/s | 160.3, 160.6, 160.6 (2·8656²·400 = 59.94 GFLOP per launch) |
| copy-in (A, Aᵀ, NaN-filled S) / copy-out (S) | 76 ms / 72 ms (about 4 GB/s) |
| device open / held in total | 0.18 s / 1.45 s |
| max abs error vs float64 host reference | 4.05e-7 (at (7970,7970), a diagonal entry ≈ 1) |
| mean abs error | 1.3e-8 |
| entries off by more than 1e-4 | 0 of 74,926,336 |
| non-finite entries | 0 (S was pre-filled with NaN on the card, so every entry was written) |
| symmetry, max abs(S − Sᵀ) | 0 (bit-exact) |

The diagonal is 1.0 for the 8649 real cells and 0 for the 7 zero-padded rows, as expected for row-normalised data.

## Staged checks before it

- Simulator (`--sysemu`, 1 shire), first 64 rows x all 400 columns: max abs error 2.5e-7, 0 entries > 1e-4. (`aat/sim/`)
- Card, first 2048 rows: 26.2 ms per launch, 128 GFLOP/s, max abs error 3.6e-7, 0 entries > 1e-4. (`runs/card-2048/`)

## How

- Code: `~/osteo/aat`, copied from `et-soc1-prototyping/workloads/sgemm`. The kernel is sgemm's scalar 1x16-strip loop with
  the square n split into m, n, k. The host reads A, builds B = Aᵀ (400 x 8656) on the host so the kernel's B reads stay
  contiguous, and writes S. `registerRuntimeLogLevels()` is kept first in `main`. `--dry` does everything but open the
  device.
- Check: `aat/check.py` (numpy, A in float64, A·Aᵀ in float64, compared with the card's fp32 S).
- Every card run: link checked, `et-who --check` for card 0, `ET_DEVICES=0 flock -n /run/lock/etsoc-shire0.lock
  timeout 10 …`, `et-who` afterwards (empty each time). Manifests and outputs are in `runs/card-*/`.
- Hashes: aat.c 5532a892…, main.cpp e6a27512…, aat.elf 13183f79…, aat_host d2a9f812….

## Notes for later

- The full size beats sgemm's README figure (127 GFLOP/s at n = 1024) because the larger size spreads the launch cost.
- It computes both triangles. Since S is symmetric, computing only j ≥ i would halve the work. 8-wide `.ps` SIMD or the
  tensor unit would be much faster again. At 0.37 s none of this matters for one product.
