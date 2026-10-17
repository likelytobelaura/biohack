# Prediction files

One JSONL per classical method (`{"prompt_id", "text": "{item: value}"}`), in the format `score_compare.py` reads.

There are no files for `col_mode` or `knn_entity_vote`, the two HLA-only methods. They abstain on all 127 HLA
targets: every masked HLA feature belongs to a gene with no observed allele anywhere (neither copy), so a method
that learns only from this data has nothing to vote with. An earlier run wrote both as all-blank files, which
were byte-identical; they were removed on 17 Oct 2026.
