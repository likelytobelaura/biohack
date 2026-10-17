# Osteosarc held-out gold labels

Gold answers for the 20% random mask of Sid Sijbrandij's osteosarcoma data
(the eval prompts in Modal volume `osteo-data:prompts/baseline.jsonl`).
15,964 prompts, 24,634 masked items, every item labelled.

Values were rebuilt from the public bucket `s3://sid-sijbrandij-osteosarc-dataset`.
A source file was accepted for an entity only if it reproduced ≥99% of that
entity's known training values; `validated_files.json` lists which file each entity came from.

- `golden.jsonl` — one row per prompt: `prompt_id`, `data_type`, `entity`, `messages` (the eval prompt),
  `items` (number → item name), `gold` (number → true value), and `completion`
  (gold as an assistant message, same shape as `train/sid_full.jsonl`).
- `labels.csv` — the same labels flat: one row per (prompt_id, item number).
- `build_golden.py` — rebuilds both: `python3 build_golden.py <baseline.jsonl> <baseline_truth.jsonl> <out_dir>`.

Caveats: 194 CITE-seq targets (tumor_T1, tumor_T3_CD45neg) have no training cells to validate
against; their values come from files that passed for the other tumour samples. Some fully held-out
entities (all eval cells, 47 DIA regions) were matched by name inside a validated file.
Gold values keep the source's full precision; training completions were rounded to ~4 significant figures.
