#!/bin/bash
# run_matrix.sh <engine-label> <set>...   (sequential: one process per set)
#
# Env: WORK (holds data/, results/, models/), APP_TARGET (cargo target dir of
# the unsound crate, built with --example asr_eval), SPIKE_TARGET (target dir of
# this crate), FA_BIN (fluidaudio-eval binary), MODELS (unsound's models dir).
# Writes $WORK/results/<label>.<set>.jsonl and a /usr/bin/time -l log beside it.
set -u
APP=$APP_TARGET/release/examples/asr_eval
TC=$SPIKE_TARGET/release/parakeet-eval
M=$WORK/models
label=$1; shift
mkdir -p "$WORK/results"
for set in "$@"; do
  man=$WORK/data/$set/manifest.jsonl
  out=$WORK/results/$label.$set.jsonl
  case $label in
    whisper-turbo) cmd=("$APP" whisper "$MODELS/ggml-large-v3-turbo.bin" eval "$man" "$out");;
    sherpa-v3)     cmd=("$APP" sherpa-parakeet "$MODELS/parakeet-tdt-v3" eval "$man" "$out");;
    wcpp-v3)       cmd=("$APP" wcpp-parakeet "$M/wcpp/ggml-parakeet-tdt-0.6b-v3-q8_0.bin" eval "$man" "$out");;
    tcshared-v3)   cmd=("$APP" tc "$M/tc/parakeet-tdt-0.6b-v3-Q8_0.gguf" eval "$man" "$out");;
    tcshared-v2)   cmd=("$APP" tc "$M/tc/parakeet-tdt-0.6b-v2-Q8_0.gguf" eval "$man" "$out");;
    tc-v3)         cmd=("$TC" "$M/tc/parakeet-tdt-0.6b-v3-Q8_0.gguf" eval "$man" "$out");;
    tc-v2)         cmd=("$TC" "$M/tc/parakeet-tdt-0.6b-v2-Q8_0.gguf" eval "$man" "$out");;
    fa-v3-ane)     cmd=("$FA_BIN" v3 "$M/fa" ane eval "$man" "$out");;
    fa-v3-gpu)     cmd=("$FA_BIN" v3 "$M/fa" gpu eval "$man" "$out");;
    fa-v2-ane)     cmd=("$FA_BIN" v2 "$M/fa" ane eval "$man" "$out");;
    *) echo "unknown $label"; exit 1;;
  esac
  /usr/bin/time -l "${cmd[@]}" > "$WORK/results/$label.$set.log" 2>&1
  grep -E "peak memory footprint" "$WORK/results/$label.$set.log" | tr '\n' ' '
  echo " <- $label $set"
done
