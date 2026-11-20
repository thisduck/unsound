"""Compare two result runs transcript by transcript.

  python -I compare.py <results_dir> <engine_a> <engine_b> [sets...]

Prints, per set, how many hypotheses are byte-identical, and the differing
pairs (first few). Used to check that an engine built against a different
ggml, or in a process shared with other engines, produces the same text.
"""
import glob, json, os, sys

res, a, b = sys.argv[1:4]
sets = sys.argv[4:] or sorted({os.path.basename(f)[len(a) + 1:-6] for f in glob.glob(os.path.join(res, f"{a}.*.jsonl"))})
tot = same = 0
for s in sets:
    fa, fb = os.path.join(res, f"{a}.{s}.jsonl"), os.path.join(res, f"{b}.{s}.jsonl")
    if not (os.path.exists(fa) and os.path.exists(fb)):
        continue
    ha = {r["id"]: r["hyp"] for r in map(json.loads, open(fa)) if "id" in r}
    hb = {r["id"]: r["hyp"] for r in map(json.loads, open(fb)) if "id" in r}
    ids = [i for i in ha if i in hb]
    diff = [i for i in ids if ha[i] != hb[i]]
    tot += len(ids); same += len(ids) - len(diff)
    print(f"{s:<20} {len(ids) - len(diff)}/{len(ids)} identical")
    for i in diff[:3]:
        print(f"   {a}: {ha[i][:110]}\n   {b}: {hb[i][:110]}")
print(f"TOTAL {same}/{tot} identical")
