#!/bin/zsh
# commit + push the refreshed inventory and newly mirrored small files from ~/biohack to origin (fast-forward only).
# Commits nothing if the staged diff looks like it holds a credential, or if any staged file is over 50 MB.
cd ~/biohack || exit 1
git pull -q --ff-only || { echo "git pull failed; not pushing"; exit 1; }
git add -A
git diff --cached --quiet && { echo "nothing new"; exit 0; }
if git diff --cached | grep -qE 'hf_[A-Za-z0-9]{30,}|gh[opsu]_[A-Za-z0-9]{30,}|nvapi-[A-Za-z0-9_-]{20,}|sk-ant-|AKIA[A-Z0-9]{16}|tskey-|PRIVATE KEY'; then
  git reset -q; echo "possible credential in staged changes; nothing committed"; exit 1
fi
big=$(git diff --cached --name-only -z | xargs -0 -I{} find {} -size +50M 2>/dev/null)
[[ -n $big ]] && { git reset -q; echo "file over 50 MB staged: $big; nothing committed"; exit 1; }
n=$(git diff --cached --name-only | wc -l | tr -d ' ')
git commit -qm "Auto-refresh inventory: $n files ($(date '+%Y-%m-%d %H:%M'))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "pushed $(git rev-parse --short HEAD) ($n files)"
