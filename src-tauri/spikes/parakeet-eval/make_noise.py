"""Non-speech clips (no reference: any words output are hallucinations)."""
import json, os, subprocess, sys
import numpy as np, soundfile as sf

D = os.path.join(sys.argv[1], "nonspeech")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
SR = 16000
items = []


def save(name, a):
    p = os.path.join(D, name + ".wav")
    sf.write(p, np.clip(a, -1, 1).astype("float32"), SR, subtype="PCM_16")
    items.append(dict(id=name, path=p, ref="", dur=len(a) / SR, lang="en"))


def db(x):
    return 10 ** (x / 20)


def pink(n):
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.maximum(1, np.arange(len(f))))
    p = np.fft.irfft(f, n)
    return p / np.sqrt(np.mean(p ** 2))


save("silence_10s", np.zeros(10 * SR))
save("roomtone_m60_10s", pink(10 * SR) * db(-60))
save("roomtone_m45_10s", pink(10 * SR) * db(-45))
save("white_m30_10s", rng.standard_normal(10 * SR) * db(-30))
save("hum50_m35_10s", np.sin(2 * np.pi * 50 * np.arange(10 * SR) / SR) * db(-35) + pink(10 * SR) * db(-60))


def load(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="float32")


L = "/Library/Audio/Apple Loops/Apple"
mix_parts = [f"{L}/09 Disco Funk/Sky Lounge Electric Piano 01.caf", f"{L}/09 Disco Funk/Disco Delight Slap Bass.caf",
             f"{L}/09 Disco Funk/Boogie Right Funk Guitar 01.caf"]
parts = [load(p) for p in mix_parts if os.path.exists(p)]
if parts:
    n = 12 * SR
    mix = sum(np.tile(p, n // len(p) + 1)[:n] for p in parts)
    save("music_funk_12s", mix / np.max(np.abs(mix)) * 0.5)
for name, rel in [("music_choir", "01 Hip Hop/City of Angeles Choir.caf"),
                  ("music_voices_break", "10 Vintage Breaks/Distant Voices Break.caf")]:
    p = f"{L}/{rel}"
    if os.path.exists(p):
        a = load(p)
        a = np.tile(a, (10 * SR) // len(a) + 1)[: 10 * SR]
        save(name + "_10s", a / np.max(np.abs(a)) * 0.5)

with open(os.path.join(D, "manifest.jsonl"), "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
print(len(items), "clips")
