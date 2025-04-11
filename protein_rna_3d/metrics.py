import numpy as np


def calculate_d0(l_ref: int) -> float:
    """Calculates the d0 scaling factor based on reference length."""
    if l_ref < 12:
        return 0.3
    elif l_ref < 16:
        return 0.4
    elif l_ref < 20:
        return 0.5
    elif l_ref < 24:
        return 0.6
    elif l_ref < 30:
        return 0.7
    else:
        # Formula for Lref >= 30: 0.6 * (Lref - 0.5)^(1/2) - 2.5
        # Clamped at a minimum value, although the exact minimum isn't specified
        # in the description, other sources often use a minimum d0 around 0.5 or higher.
        # We'll use the direct formula for now.
        # Note: CASP uses slightly different formulas sometimes.
        # d0 = 1.24 * (l_ref - 15) ** (1.0 / 3.0) - 1.8 # Common alternative
        # Let's stick to the competition description formula precisely:
        return 0.6 * (l_ref - 0.5) ** 0.5 - 2.5


def calculate_tm_score(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    aligned_indices_pred: np.ndarray,
    aligned_indices_ref: np.ndarray,
    l_ref: int,
) -> float:
    """Calculates the TM-score between predicted and reference coordinates.

    Args:
        pred_coords: Predicted coordinates (N, 3).
        ref_coords: Reference coordinates (M, 3).
        aligned_indices_pred: Indices of aligned residues in pred_coords (L_align,).
        aligned_indices_ref: Indices of aligned residues in ref_coords (L_align,).
        l_ref: Length of the reference structure.

    Returns:
        The TM-score.
    """
    if len(aligned_indices_pred) != len(aligned_indices_ref):
        raise ValueError("Aligned index arrays must have the same length.")
    if l_ref <= 0:
        raise ValueError("Reference length (l_ref) must be positive.")

    l_align = len(aligned_indices_pred)
    if l_align == 0:
        return 0.0  # No aligned residues, TM-score is 0

    # Select the coordinates of the aligned residues
    pred_aligned = pred_coords[aligned_indices_pred]
    ref_aligned = ref_coords[aligned_indices_ref]

    # Calculate squared distances between aligned pairs
    d_sq = np.sum((pred_aligned - ref_aligned) ** 2, axis=1)

    # Calculate d0
    d0 = calculate_d0(l_ref)
    d0_sq = d0**2

    # Handle potential non-positive d0 from formula for small Lref >= 30
    # (though unlikely for typical RNA sizes, safety check)
    if d0 <= 0:
        # In practice, for Lref >= 30, d0 should be positive.
        # If formula gives non-positive, fallback or error might be needed.
        # TM-score paper/implementations might clamp d0_sq or use alternatives.
        # For now, let's raise an error as it indicates an edge case or formula issue
        # for the given Lref.
        raise ValueError(
            f"Calculated d0 is non-positive ({d0}) for Lref={l_ref}. Check formula/Lref."
        )

    # Calculate the sum term
    score_sum = np.sum(1.0 / (1.0 + d_sq / d0_sq))

    # Calculate TM-score
    tm_score = (1.0 / l_ref) * score_sum

    # TM-score is defined between 0 and 1
    return max(0.0, min(1.0, tm_score))


# Note: This implementation assumes the alignment (rotation/translation)
# and residue pairing has already been done by a tool like US-align.
# It calculates the score based on the provided aligned coordinates.
