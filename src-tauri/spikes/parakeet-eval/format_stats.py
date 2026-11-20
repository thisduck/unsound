"""Punctuation/capitalisation/number-format statistics per engine.

  python -I format_stats.py <data_dir> <results_dir> <set> <engine>...

Per set: % of non-empty hypotheses starting with a capital, % ending in
. ? !, commas and sentence ends per 100 words, % containing a numeral; the
reference row shows the same for the (orthographic) references.
"""
import json, os, re, sys

data, res, sset, engines = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]


def stats(txts):
    txts = [t.strip() for t in txts if t.strip()]
    w = sum(len(t.split()) for t in txts)
    return (100 * sum(t[0].isupper() for t in txts) / len(txts),
            100 * sum(t[-1] in ".?!" for t in txts) / len(txts),
            100 * sum(t.count(",") for t in txts) / w,
            100 * sum(len(re.findall(r"[.?!](\s|$)", t)) for t in txts) / w,
            100 * sum(bool(re.search(r"[0-9]", t)) for t in txts) / len(txts))


man = [json.loads(l) for l in open(os.path.join(data, sset, "manifest.jsonl"))]
print(f"{sset:<16} {'cap%':>6} {'end%':>6} {',/100w':>7} {'./100w':>7} {'num%':>6}")
print(f"{'reference':<16} " + " ".join(f"{x:6.1f}" for x in stats([m["ref"] for m in man])))
for e in engines:
    hy = [r["hyp"] for r in map(json.loads, open(os.path.join(res, f"{e}.{sset}.jsonl"))) if "hyp" in r]
    print(f"{e:<16} " + " ".join(f"{x:6.1f}" for x in stats(hy)))
