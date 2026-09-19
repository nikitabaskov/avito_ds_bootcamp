import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def pick_triples(
    positives: list[np.ndarray],
    known: list[np.ndarray],
    hits: np.ndarray,
    band: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors, pos, neg = [], [], []
    for a, (rows, seen, found) in enumerate(zip(positives, known, hits, strict=True)):
        candidates = found[band[0] : band[1]]
        candidates = candidates[(candidates >= 0) & ~np.isin(candidates, seen)]
        if rows.size and candidates.size:
            anchors.append(a)
            pos.append(rng.choice(rows))
            neg.append(rng.choice(candidates))
    return np.array(anchors), np.array(pos), np.array(neg)


def unique_batches(
    keys: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    queue = deque(rng.permutation(len(keys)).tolist())
    batches = []
    while queue:
        batch, used_keys, used_items, skipped = [], set(), set(), []
        while queue and len(batch) < batch_size:
            i = queue.popleft()
            items = {int(positives[i]), int(negatives[i])}
            if keys[i] in used_keys or used_items & items or len(items) < 2:
                skipped.append(i)
                continue
            batch.append(i)
            used_keys.add(keys[i])
            used_items |= items
        queue.extendleft(reversed(skipped))
        if len(batch) < batch_size:
            break
        batches.append(np.array(batch))
    return batches


def contrastive_loss(queries: torch.Tensor, docs: torch.Tensor, scale: float) -> torch.Tensor:
    logits = queries @ docs.T * scale
    return F.cross_entropy(logits, torch.arange(len(queries), device=queries.device))


def embed(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    features = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in model.preprocess(texts).items()
    }
    return F.normalize(model(features)["sentence_embedding"].float(), dim=-1)


def train_encoder(
    model: SentenceTransformer,
    anchors: list[str],
    passages: list[str],
    triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    batches: list[np.ndarray],
    lr: float,
    warmup_share: float,
    scale: float,
    log_every: int = 100,
) -> list[dict]:
    a_idx, p_idx, n_idx = triples
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup = max(1, int(len(batches) * warmup_share))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            (step + 1) / warmup, max(0.0, (len(batches) - step) / (len(batches) - warmup))
        ),
    )
    model.train()
    log, running, started = [], [], time.perf_counter()
    for step, batch in enumerate(batches, 1):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
            q = embed(model, [anchors[i] for i in a_idx[batch]])
            d = embed(model, [passages[i] for i in np.concatenate([p_idx[batch], n_idx[batch]])])
        loss = contrastive_loss(q, d, scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        running.append(loss.item())
        if step % log_every == 0 or step == len(batches):
            log.append(
                {
                    "step": step,
                    "loss": float(np.mean(running)),
                    "lr": schedule.get_last_lr()[0],
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            print(log[-1], flush=True)
            running = []
    model.eval()
    return log
