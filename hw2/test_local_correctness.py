"""CPU-only checks for HW2 — safe to run on a Mac before paying for GPU time.

Run from the hw2 directory (after installing requirements in a venv):

    cd hw2
    pytest test_local_correctness.py -q

What this validates:
  - optimized_loop matches the reference slow loop token-for-token
  - profile() exports a readable Chrome trace JSON
  - hw2_task imports cleanly

What this does NOT validate:
  - Speedup vs the official fp32 CUDA baseline (needs L40S + hw2/hw2_task.py main)
  - bf16 numerics on GPU (CPU Llama may not support bfloat16 the same way)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

# hw2/ on sys.path when pytest is launched from hw2/
from hw2_task import optimized_loop, profile

HW2_DIR = Path(__file__).parent
if str(HW2_DIR) not in sys.path:
    sys.path.insert(0, str(HW2_DIR))


SEED = 0
N_STEPS = 8
PROMPT_LEN = 16


def _slow_loop_reference(model, input_ids, n_steps: int) -> list[int]:
    """Same logic as utils.slow_loop — kept here so tests never touch CUDA."""
    generated_ids = input_ids.clone()
    generated_tokens: list[int] = []
    for _ in range(n_steps):
        outputs = model(input_ids=generated_ids)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id.item())
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    return generated_tokens


def _build_tiny_cpu_model() -> LlamaForCausalLM:
    torch.manual_seed(SEED)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=PROMPT_LEN + N_STEPS + 8,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _get_cpu_input_ids() -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return torch.randint(0, 256, (1, PROMPT_LEN), generator=gen, dtype=torch.long)


@pytest.fixture
def cpu_model_and_inputs():
    return _build_tiny_cpu_model(), _get_cpu_input_ids()


def test_optimized_loop_matches_slow_reference(cpu_model_and_inputs):
    model, input_ids = cpu_model_and_inputs
    expected = _slow_loop_reference(model, input_ids, N_STEPS)
    actual = optimized_loop(model, input_ids, N_STEPS)
    assert actual == expected, (
        "optimized_loop must match the slow baseline token-for-token "
        f"(expected {expected}, got {actual})"
    )


def test_profile_exports_chrome_trace(cpu_model_and_inputs, tmp_path, monkeypatch):
    model, input_ids = cpu_model_and_inputs
    monkeypatch.setattr("hw2_task.RESULTS_DIR", tmp_path)
    monkeypatch.setattr("hw2_task.PROFILE_STEPS", 3)

    trace_name = "local_cpu_trace.json"
    profile(optimized_loop, model, input_ids, trace_name)

    trace_path = tmp_path / trace_name
    assert trace_path.is_file(), "profile() should write a Chrome trace file"

    data = json.loads(trace_path.read_text())
    events = data.get("traceEvents", [])
    assert len(events) > 0, "trace should contain at least one event"

    names = {e.get("name") for e in events if isinstance(e, dict)}
    assert any("generation_loop" in (n or "") for n in names), (
        "trace should include the record_function label"
    )


def test_hw2_task_imports():
    import hw2_task  # noqa: F401

    assert callable(hw2_task.optimized_loop)
    assert callable(hw2_task.profile)
    assert callable(hw2_task.generate_optimized)
