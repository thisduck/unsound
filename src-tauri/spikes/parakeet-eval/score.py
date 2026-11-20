"""Score asr_eval / parakeet-eval output against manifest references.

  python -I score.py <data_dir> <results_dir>

Results files are named <engine>.<set>.jsonl; <data_dir>/<set>/manifest.jsonl
holds the references. English sets use the Whisper EnglishTextNormalizer
(`whisper-normalizer` package, a port of openai/whisper's normalizers, the
same normaliser the Open ASR Leaderboard uses) on both reference and
hypothesis. Other languages use NFKC + lowercase + strip Unicode punctuation
and symbols (P*, S*) while keeping combining marks (Whisper's
BasicTextNormalizer strips M* too, which mangles Devanagari).
Corpus WER = total word edits / total reference words, with a 95% bootstrap
interval over utterances (1000 resamples).
"""
import glob, json, os, random, re, sys, unicodedata

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

EN = EnglishTextNormalizer()


def norm_other(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


def edits(ref, hyp):
    if not ref:
        return len(hyp.split()), 0
    o = jiwer.process_words(ref, hyp if hyp else "")
    return o.substitutions + o.deletions + o.insertions, len(ref.split())


def char_edits(ref, hyp):
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    if not r:
        return len(h), 0
    o = jiwer.process_characters(r, h if h else "")
    return o.substitutions + o.deletions + o.insertions, len(r)


def main():
    data, res = sys.argv[1], sys.argv[2]
    rows = []
    nonspeech = []
    for f in sorted(glob.glob(os.path.join(res, "*.jsonl"))):
        base = os.path.basename(f)[:-6]
        if "." not in base:
            continue
        engine, sset = base.split(".", 1)
        man = os.path.join(data, sset, "manifest.jsonl")
        if not os.path.exists(man):
            continue
        refs = {}
        for l in open(man):
            m = json.loads(l)
            refs[m["id"]] = m
        per = []
        meta = None
        lat = []
        for l in open(f):
            r = json.loads(l)
            if r.get("meta"):
                meta = r
                continue
            m = refs[r["id"]]
            lang = m.get("lang", "en")
            n = EN if lang == "en" else norm_other
            ref, hyp = n(m["ref"]), n(r["hyp"])
            e, w = edits(ref, hyp)
            ce, cw = char_edits(ref, hyp)
            per.append((e, w, ce, cw, r["hyp"] == ""))
            lat.append((r["dur"], r["ms"]))
        if not per:
            continue
        if sum(p[1] for p in per) == 0:  # non-speech set: every output word is a hallucination
            nonspeech.append((engine, sset, len(per), sum(p[0] for p in per), sum(not p[4] for p in per)))
            continue
        E = sum(p[0] for p in per); W = sum(p[1] for p in per)
        CE = sum(p[2] for p in per); CW = sum(p[3] for p in per)
        rnd = random.Random(0)
        boots = []
        for _ in range(1000):
            s = [per[rnd.randrange(len(per))] for _ in per]
            boots.append(100 * sum(p[0] for p in s) / max(1, sum(p[1] for p in s)))
        boots.sort()
        audio = sum(d for d, _ in lat); compute = sum(ms for _, ms in lat) / 1000
        rows.append(dict(engine=engine, set=sset, n=len(per), words=W, wer=100 * E / W, lo=boots[25], hi=boots[974],
                         cer=100 * CE / CW, empty=sum(p[4] for p in per), audio_min=audio / 60,
                         rtfx=audio / compute if compute else 0, load_ms=meta and meta["load_ms"],
                         first_ms=meta and meta["first_run_ms"]))
    print(f"{'set':<20}{'engine':<22}{'n':>4}{'words':>7}{'WER%':>7}{'95% CI':>15}{'CER%':>7}{'empty':>6}{'RTFx':>8}")
    for r in sorted(rows, key=lambda r: (r["set"], r["engine"])):
        print(f"{r['set']:<20}{r['engine']:<22}{r['n']:>4}{r['words']:>7}{r['wer']:>7.2f}"
              f"{'[%.1f, %.1f]' % (r['lo'], r['hi']):>15}{r['cer']:>7.2f}{r['empty']:>6}{r['rtfx']:>8.1f}")
    for e, sset, n, words, nonempty in sorted(nonspeech):
        print(f"{sset:<20}{e:<22}{n:>4} clips, {nonempty} with output, {words} hallucinated words")
    json.dump(rows, open(os.path.join(res, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
