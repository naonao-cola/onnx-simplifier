"""Pinned model sources and the fixed evaluation text for the LLM measurements.

Every model is fetched from the Hugging Face Hub at a pinned revision and its weights file is
checked against a sha256 before use.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MINILM = dict(
    repo="sentence-transformers/all-MiniLM-L6-v2",
    rev="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    sha256="53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
)
SMOLLM = dict(
    repo="HuggingFaceTB/SmolLM2-135M",
    rev="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    sha256="80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
)

ENC_SEQ = 128  # encoder sequence length (padded)
PREFILL = 128  # decoder prefill length (padded)
MAXLEN = 256  # decoder KV-cache length

# 20 real sentences for the encoder (embedding cosine vs fp32 torch)
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A man is playing a guitar on the street corner.",
    "The stock market fell sharply after the interest rate announcement.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "She booked a flight to Tokyo for the cherry blossom season.",
    "The new smartphone has a faster processor and a better camera.",
    "Heavy rain caused flooding in several low-lying neighborhoods.",
    "He spent the afternoon reading a novel in the park.",
    "Neural networks learn representations from large amounts of data.",
    "The restaurant is famous for its handmade pasta and wood-fired pizza.",
    "Children were building sandcastles on the beach at sunset.",
    "The committee postponed the vote until next Thursday.",
    "Regular exercise improves both physical and mental health.",
    "The museum's new exhibition features ancient Egyptian artifacts.",
    "Traffic on the highway was backed up for miles after the accident.",
    "Quantum computers use qubits that can exist in superposition.",
    "The chef added a pinch of salt to bring out the flavor.",
    "Our team won the championship after a dramatic overtime goal.",
    "The library extended its opening hours during exam week.",
    "Electric vehicles are becoming cheaper as battery costs decline.",
]

# 10 prompts for the decoder (greedy continuation vs fp32 torch)
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "The main advantages of renewable energy are",
    "To make a cup of tea, first",
    "In mathematics, a prime number is",
    "The history of the Roman Empire",
    "def fibonacci(n):",
    "The best way to learn a new language is",
    "Water boils at a temperature of",
    "Artificial intelligence will change the world by",
]


def fetch(
    spec: dict,
    files=(
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ),
) -> Path:
    """Download ``files`` of a pinned repo; returns the snapshot dir. Verifies the weights' sha256."""
    from huggingface_hub import hf_hub_download

    path = None
    for f in files:
        path = Path(hf_hub_download(spec["repo"], f, revision=spec["rev"]))
    snap = path.parent
    h = hashlib.sha256()
    with open(snap / "model.safetensors", "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != spec["sha256"]:
        raise SystemExit(
            f"{spec['repo']}: model.safetensors sha256 {h.hexdigest()} != pinned {spec['sha256']}"
        )
    return snap
