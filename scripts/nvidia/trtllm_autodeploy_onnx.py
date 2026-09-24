"""Run an ONNX decoder LLM on TensorRT-LLM via AutoDeploy, using ``onnxsim.to_torch``.

TensorRT-LLM has no ONNX importer (its legacy TensorRT backend and ONNX tooling were
removed in 1.3). AutoDeploy, its path for arbitrary models, takes a PyTorch module
from a *model factory*, exports it with ``torch.export``, matches attention / RoPE /
RMSNorm onto its canonical ops and swaps attention for TensorRT-LLM's paged-KV-cache
kernels. This registers an ``OnnxModelForCausalLM`` factory whose module is
``onnxsim.to_torch.onnx_to_torch(model)``: the ONNX graph interpreted op by op, with
decoder attention emitted as causal SDPA and the ONNX KV cache stripped (AutoDeploy
inserts its own).

    python trtllm_autodeploy_onnx.py EXPORT_DIR/onnx/model_fp16.onnx [--tokenizer EXPORT_DIR]

``EXPORT_DIR`` is an ``optimum``/``onnx-community``-style directory (``config.json``,
``tokenizer.json`` next to ``onnx/``); the tokenizer defaults to the ONNX file's
grandparent directory. Runs the prompts through AutoDeploy's ``LLM`` API, greedy, and
prints the generated text and throughput.

Needs a TensorRT-LLM (1.2.1 tested) venv with onnxsim installed (``--no-deps``, so its
pinned onnx/torch stay put). ``TLLM_WORKER_USE_SINGLE_PROCESS=1`` is set here: it keeps
the worker in this process, which both avoids ``MPI_Comm_spawn`` (broken under the pip
``openmpi`` wheel) and means the factory registered below is visible to the worker.
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

import onnx  # noqa: E402
import torch  # noqa: E402
from tensorrt_llm._torch.auto_deploy.models.factory import (  # noqa: E402
    FullModelExportInfo,
    ModelFactory,
    ModelFactoryRegistry,
)

from onnxsim.to_torch import onnx_state_dict, onnx_to_torch  # noqa: E402


@ModelFactoryRegistry.register("OnnxModelForCausalLM")
class OnnxModelForCausalLMFactory(ModelFactory):
    """AutoDeploy model factory for an ONNX decoder LLM (``model`` = the .onnx path).

    ``model_kwargs`` are passed through to :func:`onnxsim.to_torch.onnx_to_torch`
    (``inputs``, ``outputs``, ``strip_kv_cache``, ``attention``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._onnx = None
        self._param_names = None

    def _load_onnx(self):
        if self._onnx is None:
            self._onnx = onnx.load(self.model)
        return self._onnx

    def _build_model(self, device):
        mod = onnx_to_torch(self._load_onnx(), device=str(device), **self.model_kwargs)
        self._param_names = mod.param_names
        return mod.eval()

    def _load_checkpoint(self, model, device):
        # `model` is the exported GraphModule by now: same state-dict keys, but the
        # ONNX-name mapping only lives on the module _build_model returned.
        sd = onnx_state_dict(self._param_names, self._load_onnx())
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(
                f"ONNX initializers missing for parameters: {missing[:5]}"
            )
        model.to(device)

    def get_export_infos(self, model):
        return [FullModelExportInfo()]

    def init_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer)


PROMPTS = [
    "Explain what an RMSNorm layer does in a transformer, in three sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is the capital of Japan, and what is it famous for?",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("onnx_model")
    ap.add_argument(
        "--tokenizer", help="default: the ONNX file's grandparent directory"
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=2,
        help="AutoDeploy traces with this batch size; torch.export specializes a size-1 "
        "example dim, so keep it >= 2 even for batch-1 serving",
    )
    ap.add_argument("--kv-fraction", type=float, default=0.5)
    ap.add_argument(
        "--matmul-nbits",
        choices=("dequant", "packed"),
        default="dequant",
        help="int4 MatMulNBits: dequantize once to fp16 (default), or keep packed int4 "
        "and run onnxsim's Triton W4A16 kernels",
    )
    ap.add_argument(
        "--compile-backend",
        default="torch-simple",
        help="AutoDeploy compile backend (torch-simple, torch-cudagraph, torch-opt, ...)",
    )
    args = ap.parse_args()

    from tensorrt_llm import SamplingParams
    from tensorrt_llm._torch.auto_deploy import LLM
    from transformers import AutoTokenizer

    tok_dir = args.tokenizer or str(Path(args.onnx_model).resolve().parent.parent)
    tok = AutoTokenizer.from_pretrained(tok_dir)
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in PROMPTS
    ]
    extra = {}
    if args.compile_backend == "torch-cudagraph":
        # AutoDeploy's default capture list includes batch sizes above max_batch_size,
        # which overflow its per-batch buffers ("Data too large for buffer
        # 'cu_seqlen'"); capture only what can actually occur.
        extra["cuda_graph_batch_sizes"] = [
            b for b in (1, 2, 4, 8, 16, 32, 64) if b <= args.max_batch_size
        ]
    t0 = time.perf_counter()
    llm = LLM(
        model=args.onnx_model,
        tokenizer=tok_dir,
        model_factory="OnnxModelForCausalLM",
        model_kwargs={"matmul_nbits": args.matmul_nbits},
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        compile_backend=args.compile_backend,
        kv_cache_config={"free_gpu_memory_fraction": args.kv_fraction},
        **extra,
    )
    print(f"AutoDeploy build: {time.perf_counter() - t0:.1f} s")
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_k=1)
    llm.generate(texts[:1], sp)  # warmup
    for t in texts:
        t0 = time.perf_counter()
        r = llm.generate([t], sp)[0]
        dt = time.perf_counter() - t0
        n = len(r.outputs[0].token_ids)
        print(f"--- {n} tokens in {dt * 1000:.1f} ms = {n / dt:.1f} tok/s (end to end)")
        print(r.outputs[0].text)
    llm.shutdown()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
