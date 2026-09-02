#!/usr/bin/env python3
"""`lm-evaluation-harness` model adapter around `CoreMLDecoder`.

Lets `lm_eval` (https://github.com/EleutherAI/lm-evaluation-harness) run its
existing IFEval, MMLU-Pro, and MATH-500 (`hendrycks_math500`) task
definitions -- prompt formatting, few-shot sampling, and (critically) each
benchmark's answer scoring/parsing -- against a `.mlpackage` exported by
`export_llm_to_coreml.py`, instead of reimplementing any of that.

This only needs to implement `generate_until`: all three target benchmarks
(IFEval, MMLU-Pro, `hendrycks_math500`) are `generate_until` tasks in
`lm-evaluation-harness` (free-form generation, scored by a verifier or
answer extraction) -- none of them are `loglikelihood`-based multiple choice,
so `CoreMLDecoder`'s existing single-token-at-a-time greedy `generate()` is
already the right shape of primitive; no teacher-forced-logprob code path is
needed. `loglikelihood`/`loglikelihood_rolling` are left unimplemented and
raise if a task actually needs them.

`CoreMLDecoder.generate()` only knows how to stop on `eos_token_id` or the
model's max context length -- it has no notion of the arbitrary stop
*strings* (`until`) `generate_until` requests carry (e.g. MMLU-Pro's
`"Question:"`, marking the start of the next few-shot example). This adapter
decodes the accumulated token ids to text after every new token and checks
for `until` there, stopping generation as soon as one appears -- an O(n^2)
sequence of `tokenizer.decode` calls, fine at the token budgets a Core ML
decode-benchmark suite runs at (tens to low hundreds of tokens), not
something to reuse for a truly long-generation workload.

Only exists so `--model coreml` resolves for `lm_eval`; import it (or pass
`--include_path` pointing at this directory) before invoking the harness --
see `run_quality_eval.py`, which does this for you.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_llm_decode_benchmark import CoreMLDecoder  # noqa: E402


@register_model("coreml")
class CoreMLLM(LM):
    """`lm_eval` model backend for a Core ML `.mlpackage` from `export_llm_to_coreml.py`.

    Usage: `lm_eval run --model coreml --model_args pretrained=model.mlpackage ...`
    (with this module imported first so the `coreml` name is registered).
    """

    def __init__(
        self,
        pretrained: str,
        compute_units: str = "ALL",
        max_gen_toks: int = 256,
        **kwargs,
    ) -> None:
        super().__init__()
        from transformers import AutoTokenizer

        model_dir = Path(pretrained).parent
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.decoder = CoreMLDecoder(pretrained, compute_units=compute_units)
        self.default_max_gen_toks = max_gen_toks

    @property
    def tokenizer_name(self) -> str:
        """Fingerprint for lm_eval's request cache, used only with --apply-chat-template."""
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Format few-shot chat history via the HF tokenizer's own chat template.

        Same job export_llm_to_coreml.py's tokenizer performs at export time --
        reused here since --apply-chat-template needs it before generation, not
        baked into the exported graph.
        """
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _generate_one(self, context: str, gen_kwargs: dict) -> str:
        until = gen_kwargs.get("until") or []
        if isinstance(until, str):
            until = [until]
        max_gen_toks = gen_kwargs.get("max_gen_toks") or self.default_max_gen_toks

        prompt_ids = self.tokenizer(context, return_tensors="np")["input_ids"][
            0
        ].tolist()
        # Leave room for at least one generated token even for a prompt near
        # this model's max context length, rather than raising.
        max_gen_toks = min(
            max_gen_toks,
            max(1, self.decoder.max_context_length - len(prompt_ids)),
        )

        generated_ids: list[int] = []
        for token_id, _dt in self.decoder.generate(
            prompt_ids, max_gen_toks, self.tokenizer.eos_token_id
        ):
            generated_ids.append(token_id)
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            cut = min((text.find(u) for u in until if u in text), default=None)
            if cut is not None:
                return text[:cut]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_until(self, requests: list) -> list[str]:
        results = []
        for request in requests:
            context, gen_kwargs = request.args
            results.append(self._generate_one(context, gen_kwargs))
        return results

    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "CoreMLLM only implements generate_until -- IFEval, MMLU-Pro, and "
            "hendrycks_math500 (this adapter's target benchmarks) are all "
            "generate_until tasks and never call loglikelihood. If a task that "
            "needs multiple-choice loglikelihood scoring is added, this adapter "
            "needs teacher-forced logprobs from CoreMLDecoder, which it doesn't "
            "currently expose (see the module docstring)."
        )

    def loglikelihood_rolling(self, requests: list) -> list[float]:
        raise NotImplementedError(
            "CoreMLLM only implements generate_until -- see loglikelihood()."
        )
