"""Name -> constructor / loader for every model family."""
from __future__ import annotations

from pathlib import Path

from ..config import ARTIFACTS_DIR, MODELS

MODEL_NAMES = ["tfidf", "bilstm", "sbert_zeroshot", "sbert", "roberta", "bert"]


def build(name: str, labels: list[str], **kw):
    if name == "tfidf":
        from .tfidf import TfidfMapper
        return TfidfMapper(labels, **kw)
    if name == "bilstm":
        from .bilstm import BiLSTMMapper
        return BiLSTMMapper(labels, **kw)
    if name in ("bert", "roberta"):
        from .transformer_clf import TransformerMapper
        return TransformerMapper(labels, MODELS[name], display_name=name, **kw)
    if name in ("sbert", "sbert_zeroshot"):
        from .sbert import SBertMapper
        return SBertMapper(labels, MODELS["sbert"], finetune=(name == "sbert"), display_name=name, **kw)
    raise KeyError(name)


def load(name: str, root: Path = ARTIFACTS_DIR):
    path = Path(root) / name
    if name == "tfidf":
        from .tfidf import TfidfMapper as C
    elif name == "bilstm":
        from .bilstm import BiLSTMMapper as C
    elif name in ("bert", "roberta"):
        from .transformer_clf import TransformerMapper as C
    elif name.startswith("sbert"):
        from .sbert import SBertMapper as C
    else:
        raise KeyError(name)
    return C.load(path)
