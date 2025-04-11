import numpy as np
import pytest

from protein_rna_3d.metrics import calculate_d0, calculate_tm_score


def test_calculate_d0():
    """Tests the d0 calculation based on competition rules."""
    assert calculate_d0(10) == 0.3
    assert calculate_d0(11) == 0.3
    assert calculate_d0(12) == 0.4
    assert calculate_d0(15) == 0.4
    assert calculate_d0(16) == 0.5
    assert calculate_d0(19) == 0.5
    assert calculate_d0(20) == 0.6
    assert calculate_d0(23) == 0.6
    assert calculate_d0(24) == 0.7
    assert calculate_d0(29) == 0.7
    # Test the formula for Lref >= 30
    assert calculate_d0(30) == pytest.approx(0.6 * (29.5**0.5) - 2.5)
    assert calculate_d0(100) == pytest.approx(0.6 * (99.5**0.5) - 2.5)
    # Test edge case where Lref is exactly 30
    assert calculate_d0(30) > 0  # Ensure d0 is positive


def test_calculate_tm_score_perfect_match():
    """Tests TM-score with identical structures."""
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    l_ref = 3
    aligned_indices = np.arange(l_ref)
    # Using a length where d0 is simple
    l_ref_simple = 10  # d0 = 0.3
    aligned_indices_simple = np.arange(l_ref_simple)
    coords_simple = np.random.rand(l_ref_simple, 3)

    # Test with L=3
    score = calculate_tm_score(coords, coords, aligned_indices, aligned_indices, l_ref)
    assert score == pytest.approx(1.0)  # Perfect match should yield TM-score of 1.0

    # Test with L=10
    score_simple = calculate_tm_score(
        coords_simple,
        coords_simple,
        aligned_indices_simple,
        aligned_indices_simple,
        l_ref_simple,
    )
    assert score_simple == pytest.approx(1.0)


def test_calculate_tm_score_no_match():
    """Tests TM-score with completely different structures (large distance)."""
    pred_coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    ref_coords = np.array([[100.0, 100.0, 100.0], [101.0, 101.0, 101.0]])
    l_ref = 20  # d0 = 0.6
    aligned_indices = np.arange(2)
    score = calculate_tm_score(
        pred_coords, ref_coords, aligned_indices, aligned_indices, l_ref
    )
    # Score should be close to 0
    assert score < 0.01


def test_calculate_tm_score_partial_match():
    """Tests TM-score with a known, simple case."""
    pred_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ref_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    l_ref = 15  # d0 = 0.4
    aligned_indices = np.arange(2)
    score = calculate_tm_score(
        pred_coords, ref_coords, aligned_indices, aligned_indices, l_ref
    )
    # With perfect alignment d=0, score = L_align / L_ref = 2 / 15
    assert score == pytest.approx(2.0 / 15.0)

    # Case with some distance
    pred_coords_dist = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.4]])  # distance = 0.4
    ref_coords_dist = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    l_ref_dist = 15  # d0 = 0.4
    d0_sq = 0.4**2
    d1_sq = 0.0
    d2_sq = 0.4**2
    expected_sum = (1.0 / (1.0 + d1_sq / d0_sq)) + (1.0 / (1.0 + d2_sq / d0_sq))
    expected_sum = 1.0 + (1.0 / (1.0 + 1.0))  # 1 + 0.5 = 1.5
    expected_score = (1.0 / l_ref_dist) * expected_sum  # 1.5 / 15 = 0.1

    score_dist = calculate_tm_score(
        pred_coords_dist, ref_coords_dist, aligned_indices, aligned_indices, l_ref_dist
    )
    assert score_dist == pytest.approx(expected_score)


def test_tm_score_no_aligned_residues():
    """Tests TM-score when there are no aligned residues."""
    pred_coords = np.array([[1.0, 2.0, 3.0]])
    ref_coords = np.array([[4.0, 5.0, 6.0]])
    l_ref = 10
    aligned_indices_pred = np.array([], dtype=int)
    aligned_indices_ref = np.array([], dtype=int)
    score = calculate_tm_score(
        pred_coords, ref_coords, aligned_indices_pred, aligned_indices_ref, l_ref
    )
    assert score == 0.0


def test_tm_score_invalid_input():
    """Tests TM-score with invalid input values."""
    coords = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(
        ValueError, match="Aligned index arrays must have the same length."
    ):
        calculate_tm_score(coords, coords, np.array([0]), np.array([]), 1)
    with pytest.raises(
        ValueError, match="Reference length \(l_ref\) must be positive."
    ):
        calculate_tm_score(coords, coords, np.array([0]), np.array([0]), 0)
    with pytest.raises(
        ValueError, match="Reference length \(l_ref\) must be positive."
    ):
        calculate_tm_score(coords, coords, np.array([0]), np.array([0]), -5)

    # Test case where d0 might become non-positive (requires specific Lref >= 30)
    # The formula 0.6 * (Lref - 0.5)**0.5 - 2.5 = 0 happens when
    # 0.6 * sqrt(Lref - 0.5) = 2.5
    # sqrt(Lref - 0.5) = 2.5 / 0.6 = 4.166...
    # Lref - 0.5 = 17.36...
    # Lref = 17.86...
    # So for Lref >= 18, d0 should be positive according to this formula.
    # Let's double-check the Lref ranges given.
    # Lref=29 -> d0=0.7
    # Lref=30 -> d0=0.6*(29.5)**0.5 - 2.5 = 0.6*5.43... - 2.5 = 3.25 - 2.5 = 0.75 > 0
    # It seems d0 will be positive for all valid Lref based on the provided piecewise definition
    # and formula. The explicit ValueError for d0<=0 might be overly cautious unless
    # there's a misunderstanding of the formula or Lref ranges.
    # We'll keep the check but note it's unlikely to trigger with the current rules.
