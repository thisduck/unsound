"""Latency summary from eval outputs: python -I latency.py <results_dir>

Per engine: model load and first-run (cold) from each run's meta line
(median over runs), warm wall time for 2-10 s clips (English speech sets) and
for 20-25 s chunks (ami_sdm_chunks), and peak memory footprint from the
matching `/usr/bin/time -l` log.
"""
import glob, json, os, re, statistics, sys

res = sys.argv[1]
eng = {}
for f in glob.glob(os.path.join(res, "*.jsonl")):
    base = os.path.basename(f)[:-6]
    if "." not in base:
        continue
    e, sset = base.split(".", 1)
    d = eng.setdefault(e, dict(load=[], first=[], short=[], chunk=[], mem=[]))
    for l in open(f):
        r = json.loads(l)
        if r.get("meta"):
            d["load"].append(r["load_ms"]); d["first"].append(r["first_run_ms"])
        elif sset in ("librispeech_clean", "ami_ihm", "ami_sdm", "earnings22") and 2 <= r["dur"] <= 10:
            d["short"].append((r["dur"], r["ms"]))
        elif sset == "ami_sdm_chunks":
            d["chunk"].append((r["dur"], r["ms"]))
    log = f[:-6] + ".log"
    if os.path.exists(log):
        m = re.search(r"(\d+)\s+peak memory footprint", open(log, errors="replace").read())
        if m:
            d["mem"].append(int(m.group(1)))


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


print(f"{'engine':<16}{'load ms':>9}{'first ms':>9}{'2-10s n':>8}{'p50 ms':>8}{'p90 ms':>8}{'RTFx':>7}"
      f"{'20-25s n':>9}{'p50 ms':>8}{'max ms':>8}{'RTFx':>7}{'peak MB':>9}")
for e, d in sorted(eng.items()):
    s = [ms for _, ms in d["short"]]; c = [ms for _, ms in d["chunk"]]
    rs = sum(x for x, _ in d["short"]) / max(1e-9, sum(s) / 1000) if s else 0
    rc = sum(x for x, _ in d["chunk"]) / max(1e-9, sum(c) / 1000) if c else 0
    print(f"{e:<16}{statistics.median(d['load']):>9.0f}{statistics.median(d['first']):>9.0f}"
          f"{len(s):>8}{q(s, .5) if s else 0:>8.0f}{q(s, .9) if s else 0:>8.0f}{rs:>7.0f}"
          f"{len(c):>9}{q(c, .5) if c else 0:>8.0f}{max(c) if c else 0:>8.0f}{rc:>7.0f}"
          f"{(max(d['mem']) / 2**20 if d['mem'] else 0):>9.0f}")
