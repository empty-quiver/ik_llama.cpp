// PR2/PR3 demo for LLAMA_STATE_SEQ_FLAGS_ON_DEVICE.
//
// PR2 covered non-hybrid attention only. PR3 lifts the recurrent-layer guard
// in the device serializer, so this demo also passes bitwise on hybrid models
// (e.g. Qwen3.5 0.8B drafter, Qwen3.5-MoE/27B). The flow is identical for
// both arches — round-trip identity is what proves the qnext s_l rows are
// staged + restored correctly through the ON_DEVICE D2D mirror path.
//
// Flow:
//   1. Load a model.
//   2. Decode N=32 prompt tokens into seq 0.
//   3. Snapshot via llama_state_seq_get_data(..., ON_DEVICE).
//   4. Decode 8 more tokens (deterministic argmax on each step). Capture final
//      logits as L_orig.
//   5. Restore via llama_state_seq_set_data(..., ON_DEVICE).
//   6. Decode the same 8 tokens. Capture final logits as L_restored.
//   7. Byte-compare (epsilon=0). PASS if identical.

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

static llama_token argmax_token(const float * logits, int32_t n_vocab) {
    llama_token best = 0;
    float       best_v = logits[0];
    for (int32_t t = 1; t < n_vocab; ++t) {
        if (logits[t] > best_v) {
            best_v = logits[t];
            best   = t;
        }
    }
    return best;
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, int n_past, llama_seq_id seq) {
    // Single-batch greedy. Each call decodes the full slice; n_past is the
    // logical position prefix.
    (void)seq;
    return llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(toks.data()), toks.size(), n_past, 0)) == 0;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    // Force a small, deterministic config.
    params.n_predict = 8;
    if (params.prompt.empty()) {
        params.prompt = "The quick brown fox jumps over the lazy dog and";
    }

    print_build_info();

    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model   * model = init.model;
    llama_context * ctx   = init.context;
    if (!model || !ctx) {
        fprintf(stderr, "FAIL: failed to init model/context\n");
        return 2;
    }

    // Tokenize and trim/pad prompt to ~32 tokens.
    std::vector<llama_token> prompt = common_tokenize(ctx, params.prompt, true);
    if ((int)prompt.size() > 32) {
        prompt.resize(32);
    }
    fprintf(stderr, "prompt tokens: %zu\n", prompt.size());

    int  n_past = 0;
    if (!decode_tokens(ctx, prompt, n_past, 0)) {
        fprintf(stderr, "FAIL: decode of prompt failed\n");
        return 3;
    }
    n_past = prompt.size();

    const int    n_vocab = llama_n_vocab(model);
    const size_t kEightFloatBytes = sizeof(float) * (size_t)n_vocab;

    // Snapshot via ON_DEVICE.
    const llama_state_seq_flags ON_DEV = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    const size_t snap_size = llama_state_seq_get_size(ctx, /*seq*/0, ON_DEV);
    if (snap_size == 0) {
        fprintf(stderr, "FAIL: state_seq_get_size returned 0\n");
        return 4;
    }
    std::vector<uint8_t> snap(snap_size + 64, 0); // small headroom for magic+seq_id
    const size_t written = llama_state_seq_get_data(ctx, snap.data(), snap.size(), 0, ON_DEV);
    if (written == 0) {
        fprintf(stderr, "FAIL: state_seq_get_data ON_DEVICE returned 0\n");
        return 5;
    }
    fprintf(stderr, "ON_DEVICE snapshot: logical_size=%zu host_buffer=%zu (only meta+magic actually written)\n",
            written, snap.size());

    // Decode N_predict more tokens with greedy argmax. Capture last-position logits.
    std::vector<llama_token> follow_toks;
    follow_toks.reserve(params.n_predict);
    std::vector<float> L_orig(n_vocab, 0.0f);

    {
        // Sample-then-decode loop. After each decode, llama_get_logits_ith(ctx, -1)
        // gives the last-position logits (with batch_size=1 input, that's just
        // llama_get_logits).
        std::vector<llama_token> last(1);
        for (int step = 0; step < params.n_predict; ++step) {
            const float * logits = llama_get_logits(ctx);
            llama_token   tok    = argmax_token(logits, n_vocab);
            follow_toks.push_back(tok);

            last[0] = tok;
            if (!decode_tokens(ctx, last, n_past, 0)) {
                fprintf(stderr, "FAIL: decode step %d failed (orig run)\n", step);
                return 6;
            }
            n_past += 1;
        }
        // Capture final-position logits AFTER the last decode.
        memcpy(L_orig.data(), llama_get_logits(ctx), kEightFloatBytes);
    }

    fprintf(stderr, "orig follow tokens:");
    for (auto t : follow_toks) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    // Restore.
    const size_t restored = llama_state_seq_set_data(ctx, snap.data(), snap.size(), 0, ON_DEV);
    if (restored == 0) {
        fprintf(stderr, "FAIL: state_seq_set_data ON_DEVICE returned 0\n");
        return 7;
    }
    n_past = (int)prompt.size();
    fprintf(stderr, "restored snapshot: %zu bytes consumed\n", restored);

    // Replay same follow tokens.
    std::vector<float> L_restored(n_vocab, 0.0f);
    {
        std::vector<llama_token> last(1);
        for (int step = 0; step < (int)follow_toks.size(); ++step) {
            last[0] = follow_toks[step];
            if (!decode_tokens(ctx, last, n_past, 0)) {
                fprintf(stderr, "FAIL: decode step %d failed (restored run)\n", step);
                return 8;
            }
            n_past += 1;
        }
        memcpy(L_restored.data(), llama_get_logits(ctx), kEightFloatBytes);
    }

    // Compare with epsilon=0 (bitwise float identity).
    int32_t mismatch_count = 0;
    float   max_abs_diff   = 0.0f;
    int32_t first_idx      = -1;
    for (int32_t t = 0; t < n_vocab; ++t) {
        const float a = L_orig[t];
        const float b = L_restored[t];
        if (memcmp(&a, &b, sizeof(float)) != 0) {
            if (first_idx < 0) first_idx = t;
            mismatch_count++;
            const float d = (a > b) ? (a - b) : (b - a);
            if (d > max_abs_diff) max_abs_diff = d;
        }
    }

    if (mismatch_count == 0) {
        fprintf(stdout, "PASS: L_orig == L_restored (bitwise) over %d vocab logits\n", n_vocab);
    } else {
        fprintf(stdout, "FAIL: %d/%d vocab logits differ; max |diff|=%g; first mismatch at idx %d (orig=%g, restored=%g)\n",
                mismatch_count, n_vocab, (double)max_abs_diff, first_idx, (double)L_orig[first_idx], (double)L_restored[first_idx]);
    }

    llama_free(ctx);
    llama_free_model(model);
    return mismatch_count == 0 ? 0 : 9;
}
