#include <include/whisper.h>
// unsound: whisper.cpp's own Parakeet TDT engine (same ggml), see README.md.
#include <include/parakeet.h>
#include <ggml.h>

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
