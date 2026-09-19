import numpy as np
import torch

from candgen.core.finetune import contrastive_loss, pick_triples, unique_batches


def test_triples_take_negatives_from_band_and_skip_known_items():
    hits = np.array([[5, 6, 7, 8, -1], [1, 2, 3, -1, -1], [9, 9, 9, 9, 9]])
    positives = [np.array([5]), np.array([1]), np.array([], dtype=np.int64)]
    known = [np.array([5, 7]), np.array([1, 2, 3]), np.array([])]
    anchors, pos, neg = pick_triples(positives, known, hits, (1, 5), np.random.default_rng(0))
    assert anchors.tolist() == [0]
    assert pos.tolist() == [5]
    assert neg[0] in {6, 8}


def test_batches_never_repeat_text_or_item():
    keys = np.array(["a", "a", "b", "c", "d", "e"])
    positives = np.array([1, 2, 1, 3, 4, 5])
    negatives = np.array([10, 11, 12, 13, 4, 14])
    batches = unique_batches(keys, positives, negatives, 2, np.random.default_rng(0))
    for batch in batches:
        assert len(set(keys[batch])) == len(batch)
        items = np.concatenate([positives[batch], negatives[batch]])
        assert len(set(items.tolist())) == len(items)
    used = np.concatenate(batches)
    assert 4 not in used
    assert len(used) == len(set(used.tolist()))


def test_loss_prefers_matching_documents():
    q = torch.eye(3)
    good = contrastive_loss(q, torch.cat([torch.eye(3), -torch.eye(3)]), 20.0)
    bad = contrastive_loss(q, torch.cat([torch.eye(3).flip(0), torch.eye(3)]), 20.0)
    assert good < bad
