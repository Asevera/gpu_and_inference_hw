import torch
from torch.profiler import ProfilerActivity, profile as torch_profile, record_function
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


@torch.inference_mode()
def optimized_loop(model, input_ids, n_steps):
    past_key_values = None
    # Prefill: full prompt; decode: one token at a time with KV cache.
    cur_input = input_ids
    next_token = None
    out_tokens = torch.empty(n_steps, dtype=torch.long, device=input_ids.device)

    for step in range(n_steps):
        outputs = model(
            input_ids=cur_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        out_tokens[step] = next_token.squeeze()
        cur_input = next_token.unsqueeze(0)

    return out_tokens.tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"

    with torch_profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("generation_loop"):
            loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by=sort_by, row_limit=20))
    trace_path = RESULTS_DIR / trace_name
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace exported to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    del model
    torch.cuda.empty_cache()
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
# 1. KV cache (use_cache=True, pass only the last token after prefill):
#    ~largest win — V0 recomputes attention over the full growing sequence each
#    step (O(T) per step → O(T²) total); with cache each decode step is O(1) in
#    sequence length.
# 2. Removed per-step .item() and torch.cat — no CPU↔GPU sync or realloc per
#    token; accumulate token ids in a preallocated GPU tensor, .tolist() once.
# 3. bfloat16 in generate_optimized — halves memory bandwidth vs fp32.
# 4. @torch.inference_mode() — disables autograd bookkeeping on every forward.
#
# Biggest impact and why:
#
# KV cache. The baseline’s dominant cost is re-running the full transformer on an
# ever-longer input_ids tensor every step; decode with cache only forwards the new
# token and reuses stored K/V. Perfetto shows dense short kernels per step on the
# optimized trace vs growing matmul/attention cost in V0.
