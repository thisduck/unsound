"""Fetch small, seeded subsets of public ASR test sets with reference transcripts.
Usage: python -I fetch_sets.py <data_dir>
Audio goes to <data_dir>/<set>/<n>.wav (16 kHz mono s16), refs to manifest.jsonl.
"""
import io, json, os, random, subprocess, sys, tarfile, time, urllib.parse, urllib.request, csv

DATA = sys.argv[1]
UA = {"User-Agent": "asr-bench/0.1"}
DS = "https://datasets-server.huggingface.co/rows"


def get(url, tries=10):
    for i in range(tries):
        try:
            time.sleep(0.4)
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            wait = 15 * (i + 1) if "429" in str(e) else 2 + 3 * i
            print("retry", i, str(e)[:60], "wait", wait, flush=True)
            time.sleep(wait)


def rows(dataset, config, split, offset, length=100):
    q = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=offset, length=length))
    return json.loads(get(f"{DS}?{q}"))


def to_wav(raw, out):
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", out], input=raw)
    assert p.returncode == 0, out


def write_set(name, items):
    d = os.path.join(DATA, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "manifest.jsonl"), "w") as f:
        for i, it in enumerate(items):
            path = os.path.join(d, f"{i:03d}.wav")
            if not os.path.exists(path):
                to_wav(it["raw"](), path)
            f.write(json.dumps(dict(id=it["id"], path=path, ref=it["ref"], dur=it["dur"], lang=it.get("lang", "en"))) + "\n")
    print(name, len(items), "utts", round(sum(i["dur"] for i in items) / 60, 1), "min", flush=True)


def esb(config, split, n, total, seed, min_dur=0.0, name=None):
    rnd = random.Random(seed)
    picked = []
    seen = set()
    while len(picked) < n:
        off = rnd.randrange(0, total - 100)
        page = rows("hf-audio/open-asr-leaderboard", config, split, off)
        cands = [r["row"] for r in page["rows"] if r["row"]["audio_length_s"] >= min_dur and r["row"]["id"] not in seen]
        rnd.shuffle(cands)
        for r in cands[:3]:  # spread picks across the split
            seen.add(r["id"])
            src = r["audio"][0]["src"]
            picked.append(dict(id=r["id"], ref=r["text"], dur=r["audio_length_s"],
                               raw=(lambda s=src: get(s))))
            if len(picked) >= n:
                break
    write_set(name or config, picked)


def ami_sdm(n, seed, min_dur):
    total = rows("edinburghcstr/ami", "sdm", "test", 0, 1)["num_rows_total"]
    rnd = random.Random(seed)
    picked, seen = [], set()
    while len(picked) < n:
        off = rnd.randrange(0, total - 100)
        page = rows("edinburghcstr/ami", "sdm", "test", off)
        cands = [r["row"] for r in page["rows"]
                 if r["row"]["end_time"] - r["row"]["begin_time"] >= min_dur and r["row"]["audio_id"] not in seen]
        rnd.shuffle(cands)
        for r in cands[:3]:
            seen.add(r["audio_id"])
            src = r["audio"][0]["src"]
            picked.append(dict(id=r["audio_id"], ref=r["text"], dur=round(r["end_time"] - r["begin_time"], 2),
                               raw=(lambda s=src: get(s))))
            if len(picked) >= n:
                break
    write_set("ami_sdm", picked)
    return total


def ami_sdm_chunks(n_chunks, seed, total, target=(20.0, 25.0)):
    """Consecutive SDM utterances of one meeting concatenated (0.3 s gaps) into 20-25 s chunks:
    a stand-in for the app's meeting chunks. Reference = the utterance refs joined."""
    import numpy as np, soundfile as sf
    d = os.path.join(DATA, "ami_sdm_chunks")
    os.makedirs(d, exist_ok=True)
    rnd = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n_chunks and tries < 60:
        tries += 1
        off = rnd.randrange(0, total - 100)
        page = [r["row"] for r in rows("edinburghcstr/ami", "sdm", "test", off)["rows"]]
        page.sort(key=lambda r: (r["meeting_id"], r["begin_time"]))
        cur, dur = [], 0.0
        for r in page:
            if cur and (r["meeting_id"] != cur[0]["meeting_id"] or r["begin_time"] < cur[-1]["begin_time"]):
                cur, dur = [], 0.0
            cur.append(r)
            dur += (r["end_time"] - r["begin_time"]) + 0.3
            if dur > target[1]:
                cur, dur = [], 0.0
            elif dur >= target[0]:
                break
        if not (target[0] <= dur <= target[1]):
            continue
        i = len(out)
        path = os.path.join(d, f"{i:03d}.wav")
        pieces = []
        for r in cur:
            tmp = os.path.join(d, "_tmp.wav")
            to_wav(get(r["audio"][0]["src"]), tmp)
            a, sr = sf.read(tmp, dtype="float32")
            pieces += [a, np.zeros(int(0.3 * 16000), dtype="float32")]
        os.remove(tmp)
        sf.write(path, np.concatenate(pieces), 16000, subtype="PCM_16")
        out.append(dict(id=f"{cur[0]['meeting_id']}_{cur[0]['begin_time']:.1f}", path=path,
                        ref=" ".join(r["text"] for r in cur), dur=round(dur, 2), lang="en"))
    with open(os.path.join(d, "manifest.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print("ami_sdm_chunks", len(out), "chunks", round(sum(o["dur"] for o in out) / 60, 1), "min", flush=True)


def fleurs(lang, n):
    """Stream FLEURS dev audio tarball and stop after n clips (FLEURS dev ids are disjoint from train)."""
    base = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{lang}"
    tsv = get(f"{base}/dev.tsv").decode()
    refs = {}
    for row in csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        refs[row[1]] = row[2]  # file name -> raw transcription
    d = os.path.join(DATA, f"fleurs_{lang}")
    os.makedirs(d, exist_ok=True)
    req = urllib.request.Request(f"{base}/audio/dev.tar.gz", headers=UA)
    items = []
    with urllib.request.urlopen(req, timeout=300) as r:
        with tarfile.open(fileobj=r, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                fn = os.path.basename(m.name)
                if fn not in refs:
                    continue
                raw = tf.extractfile(m).read()
                path = os.path.join(d, f"{len(items):03d}.wav")
                to_wav(raw, path)
                import soundfile as sf
                info = sf.info(path)
                items.append(dict(id=fn, path=path, ref=refs[fn], dur=round(info.duration, 2), lang=lang.split("_")[0]))
                if len(items) >= n:
                    break
    with open(os.path.join(d, "manifest.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"fleurs_{lang}", len(items), "utts", round(sum(i["dur"] for i in items) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    which = sys.argv[2:] or ["ls", "ami", "sdm", "chunks", "e22", "fleurs"]
    if "ls" in which:
        esb("librispeech", "test.clean", 50, 2620, 1, name="librispeech_clean")
    if "ami" in which:
        esb("ami", "test", 80, 12643, 2, min_dur=2.0, name="ami_ihm")
    if "e22" in which:
        esb("earnings22", "test", 60, 2741, 3, min_dur=2.0, name="earnings22")
    total = None
    if "sdm" in which:
        total = ami_sdm(80, 4, 2.0)
    if "chunks" in which:
        ami_sdm_chunks(12, 5, total or rows("edinburghcstr/ami", "sdm", "test", 0, 1)["num_rows_total"])
    if "fleurs" in which:
        for lang in ["de_de", "fr_fr", "hi_in"]:
            fleurs(lang, 30)
