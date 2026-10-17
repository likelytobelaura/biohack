# Task: one matrix product on aifoundry3 card 0, for Laura's osteosarcoma imputation work

Read ~/lab-start.md first and follow its card rules exactly (free check, card lock, every device-opening process
capped at 10 s, card 0 only, nothing that changes the card's state).

**Card:** aifoundry3 card 0. Laura posted "using aifoundry3 card 0" in #community-lab at about 14:40 PDT on
17 October, and authorises the card runs this task needs (several capped launches are fine). No reset, retrain,
clock, power or firmware changes. No download or install over 1 GB.

**Input:** `~/osteo/cite_A.f32`: a row-major fp32 matrix A, 8656 x 400, zero-padded. SHA-256 502eae5c2bc369401750c623122939273e2bbab7ebf750df63026ba069a173ef.
It is CITE-seq data, column-centred and row-normalised, so S = A·Aᵀ is the cell-by-cell cosine similarity.

**Goal:** compute S = A·Aᵀ (8656 x 8656, fp32) on card 0, and write it row-major to `~/osteo/cite_S_card.f32`.
That's about 60 GFLOP. Start from `~/et-soc1-prototyping/workloads/sgemm`, the square random-input sgemm that
already runs on this card, or from the MNIST harness in `~/work/mnist-et`. Add file input/output and the
rectangular shape (M = N = 8656, K = 400; Bᵀ is A itself). Check it in the simulator at a small size first.

**Check and report**, in `~/osteo/NOTES.md`:
- the time per launch and GFLOP/s;
- a host-side check against a float64 host reference: max abs error, and the count of entries off by more than 1e-4.

When finished (or blocked), write a one-paragraph summary to `~/osteo/DONE`. Work only in `~/osteo` and build
directories you create; leave the existing `claude` tmux session and its unsent input alone.
