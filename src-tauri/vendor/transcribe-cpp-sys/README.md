# transcribe-cpp-sys (patched, spike)

transcribe-cpp-sys 0.4.2 (the crates.io tarball of
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), MIT, the
engine Handy uses), wired in by `[patch.crates-io]` in `src-tauri/Cargo.toml`
and used only by the spike examples (`transcribe-cpp` is a dev-dependency).
It builds transcribe.cpp against the ggml llama-cpp-sys-2 installs, as the
vendored whisper-rs-sys does, so the program keeps one ggml:

1. `CMakeLists.txt`: `TRANSCRIBE_USE_SYSTEM_GGML` does `find_package(ggml)`
   instead of `add_subdirectory(ggml)`. The vendored `ggml/` tree (25 MB) is
   dropped from this copy.
2. `bindings/rust/sys/build.rs`: sets that option with
   `ggml_DIR=$DEP_LLAMA_GGML_CMAKE_DIR/ggml`, `GGML_METAL=ON` (for the Metal
   capability probe), `-DTRANSCRIBE_NO_GGML_REG_FILTER`, and skips the `ggml*`
   archives in the link manifest. `src/lib.rs` names `llama_cpp_sys_2` so its
   archives are linked.
3. `src/transcribe.cpp`: the backend registration filter is compiled out.
   It relies on a transcribe.cpp patch to ggml (`ggml_backend_set_reg_filter`)
   that upstream ggml lacks, and in a shared ggml it would filter llama.cpp's
   backends too. The only loss is the allowed-backend mask (env/API).

transcribe.cpp 0.4.2 vendors ggml 0.26.0 plus four patches; llama-cpp-sys-2
0.1.157 ships ggml 0.24.0. Every ggml symbol libtranscribe references exists
in the 0.24 archives, and on this machine (M5 Max, Metal) Parakeet TDT v2 and
v3 gave byte-identical transcripts on 379 clips built either way. Of the four
patches, three are absent here: the reg filter (above), a threadpool barrier
yield under CPU oversubscription (MSVC/x86 and 2-vCPU runners), and a NULL
check when Metal cannot map a buffer. llama-cpp-sys-2 0.1.161 ships ggml
0.26.0, whose sources differ from transcribe.cpp's only by those patches.

## Updating

When llama-cpp-2 moves, re-run the spike checks (README in
`src-tauri/spikes/parakeet-eval/`): transcripts must stay non-empty and match
the standalone `parakeet-eval` build, which uses transcribe.cpp's own ggml.
When transcribe.cpp moves, re-apply the three patches (marked `unsound:`) and
check that `nm -u libtranscribe.a` needs no ggml symbol the shared ggml lacks.
