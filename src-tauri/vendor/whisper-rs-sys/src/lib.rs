#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// unsound: whisper.cpp links no ggml of its own; llama-cpp-sys-2's archives
// provide it. Name the crate so rustc links it even into binaries that use
// whisper-rs but never touch llama-cpp-2 (otherwise ggml_* is undefined).
extern crate llama_cpp_sys_2;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
