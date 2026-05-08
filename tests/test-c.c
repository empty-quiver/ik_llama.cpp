#include "llama.h"

#ifdef GGML_USE_KOMPUTE
#include "ggml-kompute.h"
#endif

// PR1 scaffolding smoke test: make sure the new ON_DEVICE flag value exists,
// has the documented bit pattern, and is distinct from the existing
// PARTIAL_ONLY flag.  These are compile-time checks; if the macros disappear
// or collide the build fails here before anyone notices at runtime.
_Static_assert(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == 1,
    "LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY must remain bit 0 for ABI stability");
_Static_assert(LLAMA_STATE_SEQ_FLAGS_ON_DEVICE == 2,
    "LLAMA_STATE_SEQ_FLAGS_ON_DEVICE must be bit 1 (mainline-compatible)");
_Static_assert((LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == 0,
    "PARTIAL_ONLY and ON_DEVICE flags must be disjoint bits");

// Type-check the public API takes and returns the expected types when called
// with the new flag.  Never invoked, just compiled.
static void llama_state_seq_ondevice_typecheck(struct llama_context * ctx,
                                                uint8_t * dst, const uint8_t * src,
                                                size_t size, llama_seq_id seq_id) {
    size_t got;
    got = llama_state_seq_get_size(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    got = llama_state_seq_get_data(ctx, dst, size, seq_id, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    got = llama_state_seq_set_data(ctx, src, size, seq_id, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    (void)got;
}

int main(void) {
    // Reference the typecheck function so it isn't stripped at -O0 / -Werror=unused
    (void)llama_state_seq_ondevice_typecheck;
    return 0;
}
