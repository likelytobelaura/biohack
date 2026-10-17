#!/bin/zsh
# refresh the inventory every 10 min until the LOMO sweep app has stopped and the jobs producing osteo data are done (max 10h)
I=${0:A:h}/inventory.py; M=~/sutro-problems/.venv/bin/modal; A=${1:-ap-AIzYiRcj5xsmQdDteujfMa}; JOBS=(6c5c300f 7d63451c c76b4182)
app_up() { $M app list --json 2>/dev/null | python3 -c "import json,sys;a=[x for x in json.load(sys.stdin) if x['app_id']=='$A'];print('y' if a and a[0]['stopped_at'] is None else 'n')"; }
jobs_busy() { for j in $JOBS; do python3 -c "import json;print(json.load(open('$HOME/.claude/jobs/$j/state.json'))['state'])" 2>/dev/null; done | grep -c working; }
for i in {1..60}; do
  python3 -I $I > /dev/null && echo "$(date +%H:%M) refreshed (app up: $(app_up), jobs working: $(jobs_busy))"
  [[ $(app_up) == n && $(jobs_busy) == 0 ]] && break
  sleep 600
done
python3 -I $I | sed -n '/## 6/,/## Links/p'
