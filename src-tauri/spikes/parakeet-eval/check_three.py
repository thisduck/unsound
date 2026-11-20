"""Check a three_engines run against standalone runs.

  python -I check_three.py <three.json> <results_dir> <whisper_engine> <parakeet_engine> [other_three.json]

- Whisper / Parakeet text in the shared process vs each engine's standalone
  eval output (results_dir/<engine>.<set>.jsonl, any set), per clip.
- Round 0 vs round 1 (reverse call order) for every engine, including the LLM.
- Optionally the LLM text vs another three_engines run (same prompts).
"""
import glob, json, os, sys

three = json.load(open(sys.argv[1]))
res, we, pe = sys.argv[2:5]


def standalone(engine):
    out = {}
    for f in glob.glob(os.path.join(res, f"{engine}.*.jsonl")):
        for r in map(json.loads, open(f)):
            if "id" in r:
                out[r["id"]] = r["hyp"]
    return out


sw, sp = standalone(we), standalone(pe)
r0 = [r for r in three if r["round"] == 0]
r1 = {r["id"]: r for r in three if r["round"] == 1}
ok = {"whisper": 0, "parakeet": 0, "w_rounds": 0, "p_rounds": 0, "llm_rounds": 0, "empty": 0}
for r in r0:
    for key, ref, k in (("whisper", sw, "whisper"), ("parakeet", sp, "parakeet")):
        if ref.get(r["id"], "").strip() == r[k]:
            ok[key] += 1
        else:
            print(f"DIFF {key} {r['id']}\n  shared:     {r[k][:120]}\n  standalone: {ref.get(r['id'], '<missing>')[:120]}")
    ok["w_rounds"] += r["whisper"] == r1[r["id"]]["whisper"]
    ok["p_rounds"] += r["parakeet"] == r1[r["id"]]["parakeet"]
    ok["llm_rounds"] += r["llm"] == r1[r["id"]]["llm"]
    ok["empty"] += not (r["whisper"] and r["parakeet"] and r["llm"])
n = len(r0)
print(f"clips {n}: whisper==standalone {ok['whisper']}, parakeet==standalone {ok['parakeet']}, "
      f"round-stable whisper {ok['w_rounds']} parakeet {ok['p_rounds']} llm {ok['llm_rounds']}, any-empty {ok['empty']}")
if len(sys.argv) > 5:
    other = {r["id"]: r for r in json.load(open(sys.argv[5])) if r["round"] == 0}
    same = sum(r["llm"] == other[r["id"]]["llm"] for r in r0 if r["id"] in other)
    print(f"llm text identical to {os.path.basename(sys.argv[5])}: {same}/{n}")
print("sample llm:", r0[0]["llm"][:100])
