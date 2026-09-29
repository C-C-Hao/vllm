# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance must not depend on how a token's prefix was computed.

The other tests in this directory disable prefix caching. With
VLLM_BATCH_INVARIANT=1 prefix caching stays on for every backend except
FLASHINFER and TRITON_MLA, so these tests check, with prefix caching on, that
the generated tokens and their logprobs are bitwise identical when the prefix
was:

* read from the prefix cache instead of computed (cold vs warm),
* read from the cache up to a block boundary inside the shared prefix,
* computed token by token in decode instead of in one prefill,
* recomputed after the request was preempted,
* shared by concurrent requests instead of computed by one request alone.

All prompts are token ids, so no comparison depends on a text round trip.
Each test fails, instead of passing vacuously, when the prefix cache was not
hit, when no preemption happened, or when too few compared positions could
differ at all. Prompt logprobs are never requested: they disable prefix-cache
reads for the request.
"""

import contextlib
import gc
import math
import os
import random

import pytest
from utils import BACKENDS, TEST_MODEL, skip_unsupported

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

BLOCK_SIZE = 16
MAX_TOKENS = 32
# A greedy position can only show a difference if its token is not certain.
MIN_INFORMATIVE = 16
INFORMATIVE_LOGPROB = math.log1p(-1e-3)


@contextlib.contextmanager
def _batch_invariant_llm(backend: str, **kwargs):
    # The autouse conftest fixture is function scoped; this engine outlives it.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_BATCH_INVARIANT", "1")
        mp.setattr(envs, "VLLM_BATCH_INVARIANT", True)
        llm = LLM(
            model=TEST_MODEL,
            max_model_len=4096,
            max_num_seqs=16,
            block_size=BLOCK_SIZE,
            # 320 blocks hold every test's cache but force preemption in
            # test_preemption_and_recompute.
            num_gpu_blocks_override=320,
            enable_prefix_caching=True,
            enforce_eager=False,
            disable_log_stats=False,
            gpu_memory_utilization=float(
                os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.4")
            ),
            seed=0,
            attention_config={"backend": backend},
            **kwargs,
        )
        try:
            yield llm
        finally:
            del llm
            gc.collect()


@pytest.fixture(scope="module", params=BACKENDS)
def llm(request):
    with _batch_invariant_llm(request.param) as engine:
        yield engine


def _prompt(llm: LLM, seed: int, n: int) -> list[int]:
    """Random ordinary-vocabulary token ids."""
    rng = random.Random(seed)
    high = min(llm.get_tokenizer().vocab_size, 32000)
    return [rng.randrange(1000, high) for _ in range(n)]


def _two_suffixes(llm: LLM, seed: int, n1: int, n2: int):
    """Two token suffixes that differ in their first token."""
    a, b = _prompt(llm, seed, n1), _prompt(llm, seed + 1, n2)
    while b[0] == a[0]:
        b = b[1:] + b[:1]
    return a, b


def _generate(llm: LLM, prompts: list[list[int]], max_tokens: int = MAX_TOKENS):
    """Greedy, with every sampling knob explicit so generation_config
    defaults cannot apply."""
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
        logprobs=1,
    )
    assert sp.prompt_logprobs is None
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False
    )
    assert len(outs) == len(prompts)
    results = []
    for out in outs:
        c = out.outputs[0]
        assert len(c.token_ids) == max_tokens and c.logprobs is not None
        lps = [d[t].logprob for t, d in zip(c.token_ids, c.logprobs)]
        results.append((list(c.token_ids), lps, out.num_cached_tokens))
    return results


def _assert_same(ref, out, label: str) -> int:
    """Asserts bitwise equal tokens and logprobs; returns informative count."""
    (ref_ids, ref_lps, _), (ids, lps, _) = ref, out
    assert ids == ref_ids, f"{label}: tokens differ\n{ref_ids}\n{ids}"
    diff = [abs(a - b) for a, b in zip(ref_lps, lps)]
    assert max(diff) == 0.0, (
        f"{label}: logprobs differ at {sum(d > 0 for d in diff)}/{len(diff)} "
        f"positions, max |delta| {max(diff):.3e}"
    )
    return sum(lp < INFORMATIVE_LOGPROB for lp in ref_lps)


def _assert_informative(n: int) -> None:
    assert n >= MIN_INFORMATIVE, (
        f"only {n} compared positions have p(top-1) < 1 - 1e-3: too weak"
    )


def _cold(llm: LLM, prompt: list[int], max_tokens: int = MAX_TOKENS):
    assert llm.reset_prefix_cache()
    (out,) = _generate(llm, [prompt], max_tokens)
    assert out[2] == 0, "cold run read the prefix cache"
    return out


def _warm_after(llm: LLM, seed_prompt: list[int], prompt: list[int], limit: int):
    """Runs `prompt` after `seed_prompt` put its blocks in the cache."""
    assert llm.reset_prefix_cache()
    _generate(llm, [seed_prompt])
    (out,) = _generate(llm, [prompt])
    hit = out[2]
    assert 0 < hit <= limit and hit % BLOCK_SIZE == 0, (
        f"expected a block-aligned prefix-cache hit in (0, {limit}], got {hit}"
    )
    return out


@skip_unsupported
@pytest.mark.parametrize("prefix_len", [1000, 2999])
def test_cold_vs_warm_prefix_cache(llm, prefix_len):
    prefix = _prompt(llm, prefix_len, prefix_len)
    s1, s2 = _two_suffixes(llm, 1, 40, 57)
    seed_prompt, prompt = prefix + s1, prefix + s2
    cold = _cold(llm, prompt)
    warm = _warm_after(llm, seed_prompt, prompt, limit=prefix_len)
    _assert_informative(_assert_same(cold, warm, "cold vs warm"))


@skip_unsupported
@pytest.mark.parametrize(
    "case,tail",
    [("shared", 1), ("shared", 15), ("repeat", 1), ("repeat", 16)],
)
def test_prefix_cache_hit_ending_mid_block(llm, case, tail):
    """shared: the cached prefix ends `tail` tokens into a block that the
    seeding request filled with different tokens.
    repeat: the same prompt again; `tail` tokens are left to recompute."""
    if case == "shared":
        shared = _prompt(llm, 3, 64 * BLOCK_SIZE + tail)
        s1, s2 = _two_suffixes(llm, 4, 20, 33)
        seed_prompt, prompt = shared + s1, shared + s2
        limit = len(shared)
    else:
        prompt = seed_prompt = _prompt(llm, 6, 64 * BLOCK_SIZE + tail)
        limit = len(prompt) - 1
    cold = _cold(llm, prompt)
    warm = _warm_after(llm, seed_prompt, prompt, limit=limit)
    _assert_informative(_assert_same(cold, warm, f"{case} tail={tail}"))


@skip_unsupported
def test_decode_vs_prefill(llm):
    """The logprob of each decoded token equals the logprob of the same token
    sampled at the end of a prefill of prompt + earlier generated tokens."""
    n = 48
    prompt = _prompt(llm, 7, 777)
    assert llm.reset_prefix_cache()
    ids, lps, hit = _generate(llm, [prompt], max_tokens=n)[0]
    assert hit == 0
    informative = 0
    for k in range(n):
        prefill = _cold(llm, prompt + ids[:k], max_tokens=1)
        ref = ([ids[k]], [lps[k]], 0)
        informative += _assert_same(ref, prefill, f"decode vs prefill, token {k}")
    _assert_informative(informative)


@skip_unsupported
def test_preemption_and_recompute(llm):
    prompts = [_prompt(llm, 100 + i, 512) for i in range(8)]
    refs = [_cold(llm, p, max_tokens=256) for p in prompts]

    def preemptions() -> int:
        return sum(
            m.value for m in llm.get_metrics() if m.name == "vllm:num_preemptions"
        )

    assert llm.reset_prefix_cache()
    before = preemptions()
    # 8 x (512 + 256) tokens do not fit in 320 blocks of 16 tokens.
    outs = _generate(llm, prompts, max_tokens=256)
    assert preemptions() > before, "no request was preempted"
    informative = 0
    for i, (ref, out) in enumerate(zip(refs, outs)):
        informative += _assert_same(ref, out, f"preempted batch, request {i}")
    _assert_informative(informative)


@skip_unsupported
def test_concurrent_shared_prefix(llm):
    shared = _prompt(llm, 8, 1000)
    prompts = [shared + _prompt(llm, 200 + i, 20 + 5 * i) for i in range(8)]
    alone = [_cold(llm, p) for p in prompts]
    assert llm.reset_prefix_cache()
    together = _generate(llm, prompts)
    assert sum(out[2] for out in together) > 0, "no in-batch prefix-cache hit"
    informative = 0
    for i, (ref, out) in enumerate(zip(alone, together)):
        informative += _assert_same(ref, out, f"concurrent request {i}")
    _assert_informative(informative)
