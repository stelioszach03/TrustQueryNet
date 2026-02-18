import numpy as np

from trustquerynet.active.acquisition import compute_scores, select_query_indices
from trustquerynet.active.loop import _select_initial_trusted_indices
from trustquerynet.active.selectors import kcenter_greedy_select


def test_bald_requires_multiple_passes_and_returns_one_score_per_sample():
    samples = np.array(
        [
            [[0.9, 0.1], [0.6, 0.4]],
            [[0.8, 0.2], [0.4, 0.6]],
            [[0.85, 0.15], [0.5, 0.5]],
        ]
    )
    scores = compute_scores("bald", samples)
    assert scores.shape == (2,)
    assert np.all(scores >= 0.0)


def test_kcenter_returns_unique_indices():
    embeddings = np.array([[0.0, 0.0], [0.2, 0.0], [10.0, 0.0], [10.2, 0.0]])
    selected = kcenter_greedy_select(embeddings, selected_mask=np.array([True, False, False, False]), budget=2)
    assert len(np.unique(selected)) == 2


def test_hybrid_selection_returns_budgeted_candidates():
    probs = np.array(
        [
            [0.95, 0.05],
            [0.51, 0.49],
            [0.55, 0.45],
            [0.52, 0.48],
            [0.60, 0.40],
        ]
    )
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 0.0],
            [10.1, 0.1],
            [5.0, 5.0],
        ]
    )
    selected = select_query_indices(
        "hybrid",
        budget=2,
        probs=probs,
        embeddings=embeddings,
        selected_mask=np.array([False, False, False, False, False]),
        shortlist_factor=2,
    )
    assert selected.shape == (2,)
    assert len(np.unique(selected)) == 2


def test_random_selection_is_reproducible_with_seed():
    probs = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.55, 0.45],
        ]
    )
    selected_a = select_query_indices("random", budget=2, probs=probs, seed=123)
    selected_b = select_query_indices("random", budget=2, probs=probs, seed=123)
    assert np.array_equal(selected_a, selected_b)
    assert len(np.unique(selected_a)) == 2


def test_initial_trusted_indices_respect_fraction():
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    trusted = _select_initial_trusted_indices(labels, fraction=0.25, seed=42)
    assert len(trusted) == 2
