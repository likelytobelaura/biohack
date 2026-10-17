"""Check S = A A^T (fp32 file) against a float64 host reference.
   python3 -I check.py A.f32 M K S.f32"""
import sys
import numpy as np

a_path, m, k, s_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
A = np.fromfile(a_path, dtype=np.float32).reshape(m, k).astype(np.float64)
S = np.fromfile(s_path, dtype=np.float32).reshape(m, m)
ref = A @ A.T
err = np.abs(S.astype(np.float64) - ref)
nonfinite = int((~np.isfinite(S)).sum())
err[~np.isfinite(err)] = np.inf
print(f"m={m} k={k} entries={m * m}")
print(f"non-finite entries {nonfinite}")
print(f"max abs error {np.max(err):.3e} at {np.unravel_index(np.argmax(err), err.shape)}")
print(f"mean abs error {np.mean(err[np.isfinite(err)]):.3e}")
print(f"entries off by more than 1e-4: {int((err > 1e-4).sum())}")
print(f"symmetry: max |S - S^T| {np.max(np.abs(S - S.T)):.3e}")
print(f"ref diag range {ref.diagonal().min():.4f}..{ref.diagonal().max():.4f}, |ref| max {np.abs(ref).max():.4f}")
