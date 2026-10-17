#!/bin/zsh
# refresh the inventory every 10 min while the osteo-lomo sweep runs, then once more when hold-out scores appear
M=~/sutro-problems/.venv/bin/modal; A=${1:-ap-AIzYiRcj5xsmQdDteujfMa}
running() { $M app list --json 2>/dev/null | python3 -c "import json,sys;a=[x for x in json.load(sys.stdin) if x['app_id']=='$A'];print('y' if a and a[0]['stopped_at'] is None else 'n')"; }
while [[ $(running) == y ]]; do python3 -I ~/biohack/inventory.py > /dev/null && echo "$(date +%H:%M) refreshed"; sleep 600; done
python3 -I ~/biohack/inventory.py > /dev/null; echo "$(date +%H:%M) sweep stopped; final pull done"
for i in {1..12}; do [[ -f ~/osteo-results/lomo_scores.json ]] && break; sleep 300; done
python3 -I ~/biohack/inventory.py | sed -n '/## 6/,/## Related/p'
