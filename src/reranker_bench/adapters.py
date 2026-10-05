"""Local CPU pair scorers; no model identifier is ever resolved over the network."""
from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Sequence
from .common import ROOT, model_catalog, offline, read_json, sha256

class ModelValidationError(RuntimeError):
    pass

def verify_model_files(model_id: str, model_root: str | Path, manifest_path: str | Path | None = None):
    model_dir = Path(model_root).resolve() / model_id
    manifest = read_json(manifest_path or ROOT / "configs/assets.json")
    entries = [entry for entry in manifest["assets"] if entry["id"] == model_id and entry["kind"] == "model"]
    if len(entries) != 1:
        raise ModelValidationError(f"No unique asset manifest for {model_id}")
    if not model_dir.is_dir():
        raise ModelValidationError(f"Model directory missing: {model_dir}")
    for entry in entries[0]["files"]:
        target = (model_dir / entry["path"]).resolve()
        if not target.is_relative_to(model_dir) or not target.is_file():
            raise ModelValidationError(f"Missing or unsafe model file: {entry['path']}")
        if target.stat().st_size != entry["size_bytes"] or sha256(target) != entry["sha256"]:
            raise ModelValidationError(f"Model hash mismatch: {entry['path']}")
    return model_dir

class PairScorer:
    def __init__(self, model_id: str, model_root: str | Path, *, max_length: int = 512,
                 threads: int = 4, verify: bool = True):
        if threads < 1 or max_length < 32:
            raise ValueError("threads must be >=1 and max_length >=32")
        offline()
        # Transformers copies bundled custom code into this local module cache.
        # Keep it beside the model assets so a restricted VDI never touches a
        # global cache directory or attempts a Hub fetch.
        os.environ["HF_MODULES_CACHE"] = str(Path(model_root).resolve().parent / "cache" / "hf_modules")
        catalog = model_catalog()
        if model_id not in catalog:
            raise KeyError(f"Unknown model_id: {model_id}")
        info = catalog[model_id]
        if max_length > info["pair_limit"]:
            raise ValueError(f"{model_id} pair limit is {info['pair_limit']} tokens")
        self.model_id = model_id
        self.info = info
        self.max_length = max_length
        self.threads = threads
        self.model_dir = verify_model_files(model_id, model_root) if verify else Path(model_root) / model_id
        import torch
        torch.set_num_threads(threads)
        self.torch = torch
        self.kind = info["loader"]
        if self.kind == "sentence_transformers":
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(str(self.model_dir), device="cpu", max_length=max_length,
                                      trust_remote_code=False, local_files_only=True)
            self.tokenizer = self.model.tokenizer
            self.score_semantics = "native CrossEncoder score; rank within a query only"
        elif self.kind == "transformers":
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
            trust = bool(info.get("trust_remote_code"))
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True,
                                                            trust_remote_code=trust)
            extra = {}
            if model_id == "jina-turbo-en":
                # Its old ALiBi implementation eagerly allocates a square
                # 8192-token bias even for short pairs, and constructs CPU
                # tensors during Transformers' meta-device lazy loading.
                # Limit the buffer to this run's declared cap and initialize
                # the small model directly on CPU. ALiBi has no learned
                # position weights, so this does not alter scores <= cap.
                config = AutoConfig.from_pretrained(str(self.model_dir), local_files_only=True,
                                                    trust_remote_code=True)
                config.max_position_embeddings = max_length
                config.num_labels = 1  # pinned checkpoint has one ranking logit
                extra = {"config": config, "low_cpu_mem_usage": False}
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=trust, use_safetensors=True,
                attn_implementation=info.get("default_attention", "eager"), **extra)
            if model_id == "jina-turbo-en":
                # Nonpersistent custom buffers are not in the checkpoint;
                # Transformers 5 can leave their meta-init storage undefined.
                embeddings = self.model.bert.embeddings
                embeddings.register_buffer("position_ids", torch.arange(max_length).expand((1, -1)), persistent=False)
                embeddings.register_buffer("token_type_ids", torch.zeros((1, max_length), dtype=torch.long), persistent=False)
                encoder = self.model.bert.encoder
                encoder.register_buffer("alibi", encoder.rebuild_alibi_tensor(size=max_length, device="cpu"),
                                        persistent=False)
            self.model.to("cpu").eval()
            self.score_semantics = "raw sequence-classification logit; rank within a query only"
        else:
            raise ModelValidationError(f"Unsupported loader: {self.kind}")
        backbone = self.model.model if self.kind == "sentence_transformers" else self.model
        backbone.to(device="cpu", dtype=torch.float32).eval()
        self.precision = str(next(backbone.parameters()).dtype).removeprefix("torch.")
        if self.precision != "float32":
            raise ModelValidationError(f"Expected float32 CPU weights, found {self.precision}")

    def token_lengths(self, pairs: Sequence[tuple[str, str]]) -> list[int]:
        lengths = []
        for query, document in pairs:
            tokens = self.tokenizer(query, document, add_special_tokens=True, truncation=False)
            lengths.append(len(tokens["input_ids"]))
        return lengths

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int = 8) -> list[float]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not pairs:
            return []
        scores: list[float] = []
        if self.kind == "sentence_transformers":
            with self.torch.inference_mode():
                raw = self.model.predict(list(pairs), batch_size=batch_size,
                                         show_progress_bar=False, convert_to_numpy=True)
            scores = [float(v) for v in raw.reshape(-1)]
        else:
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                inputs = self.tokenizer([q for q, _ in batch], [d for _, d in batch],
                                        padding=True, truncation=True, max_length=self.max_length,
                                        return_tensors="pt")
                with self.torch.inference_mode():
                    raw = self.model(**inputs).logits.detach().cpu()
                if raw.numel() != len(batch):
                    raise ModelValidationError("Expected exactly one score per query-document pair")
                scores.extend(float(v) for v in raw.reshape(-1).tolist())
        if len(scores) != len(pairs) or any(not math.isfinite(v) for v in scores):
            raise ModelValidationError("Non-finite score or score count mismatch")
        return scores

def rerank_prefix(candidate_ids: Sequence[str], scores: Sequence[float], depth: int) -> list[str]:
    if depth < 1 or depth > len(candidate_ids) or len(scores) != depth:
        raise ValueError("Scores must match the original prefix depth")
    if len(candidate_ids) != len(set(candidate_ids)) or any(not math.isfinite(float(v)) for v in scores):
        raise ValueError("Duplicate candidate or invalid score")
    prefix = sorted(range(depth), key=lambda index: (-float(scores[index]), index))
    return [candidate_ids[i] for i in prefix] + list(candidate_ids[depth:])

def validate_model(scorer: PairScorer) -> dict:
    """A small semantic and batch consistency smoke gate, separate from accuracy."""
    pairs = [("Which planet is the Red Planet?", "Mars is known as the Red Planet."),
             ("Which planet is the Red Planet?", "Venus has a thick atmosphere."),
             ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
             ("Who wrote Hamlet?", "An unrelated discussion about diesel engines.")]
    one = scorer.score(pairs, batch_size=1)
    many = scorer.score(pairs, batch_size=4)
    max_delta = max(abs(a - b) for a, b in zip(one, many))
    passed = one[0] > one[1] and one[2] > one[3] and max_delta <= 1e-3
    return {"model_id": scorer.model_id, "status": "passed" if passed else "failed",
            "checks": {"obvious_pair_order": one[0] > one[1] and one[2] > one[3],
                       "batch_consistency_max_abs_delta": max_delta,
                       "finite_and_complete": True},
            "scores": one, "max_length": scorer.max_length, "threads": scorer.threads}
