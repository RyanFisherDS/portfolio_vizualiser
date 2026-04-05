from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

@dataclass
class StyleBox:
    """
    3x3 Morningstar style box represented as a weight matrix.
    
    Rows (size):   0=Large, 1=Mid, 2=Small
    Cols (style):  0=Value, 1=Blend, 2=Growth
    
    Weights should sum to 1.0 (or close to it — we normalise).
    """
    name: str
    weights: np.ndarray  # shape (3, 3)

    SIZE_LABELS  = ["Large", "Mid", "Small"]
    STYLE_LABELS = ["Value", "Blend", "Growth"]

    def __post_init__(self):
        self.weights = np.array(self.weights, dtype=float)
        assert self.weights.shape == (3, 3), "weights must be 3x3"
        total = self.weights.sum()
        if not np.isclose(total, 1.0, atol=1e-3):
            print(f"[{self.name}] Weights sum to {total:.4f}, normalising.")
            self.weights /= total

    def display(self):
        print(f"\n{self.name}")
        print(f"{'':8}", end="")
        for s in self.STYLE_LABELS:
            print(f"{s:>10}", end="")
        print(f"{'Total':>10}")
        for i, size in enumerate(self.SIZE_LABELS):
            print(f"{size:<8}", end="")
            for j in range(3):
                print(f"{self.weights[i, j]:>9.1%}", end=" ")
            print(f"{self.weights[i, :].sum():>9.1%}")
        print(f"{'Total':<8}", end="")
        for j in range(3):
            print(f"{self.weights[:, j].sum():>9.1%}", end=" ")
        print(f"{self.weights.sum():>9.1%}")


def overlap(source: StyleBox, universe: StyleBox, verbose: bool = True) -> float:
    """
    What percentage of `source` is represented within `universe`?

    Uses the overlap coefficient on each cell:
        overlap_ij = min(source_ij, universe_ij)

    Summed across all cells this gives the fraction of `source`'s
    exposure that is 'covered' by `universe`.  Think of it as:
    'For each unit of source, how much of the same cell exists in universe?'

    Returns
    -------
    float
        Overlap as a fraction [0, 1].
    """
    overlap_matrix = np.minimum(source.weights, universe.weights)
    total_overlap = overlap_matrix.sum()

    if verbose:
        source.display()
        universe.display()

        print(f"\nOverlap matrix ({source.name} covered by {universe.name})")
        print(f"{'':8}", end="")
        for s in StyleBox.STYLE_LABELS:
            print(f"{s:>10}", end="")
        print()
        for i, size in enumerate(StyleBox.SIZE_LABELS):
            print(f"{size:<8}", end="")
            for j in range(3):
                print(f"{overlap_matrix[i, j]:>9.1%}", end=" ")
            print()

        print(f"\n→ {source.name} overlap in {universe.name}: {total_overlap:.1%}")
        print(
            f"  i.e. {total_overlap:.1%} of {source.name}'s style exposure "
            f"exists within {universe.name}"
        )

def blend(boxes: Sequence[Tuple[StyleBox, float]], name: str = "Blended Portfolio") -> StyleBox:
    if not boxes:
        raise ValueError("At least one StyleBox must be provided to blend().")

    total_weight = sum(w for _, w in boxes)
    if total_weight <= 0:
        raise ValueError(f"Total blend weight must be positive, got {total_weight:.4f}")

    if not np.isclose(total_weight, 1.0, atol=1e-6):
        boxes = [(box, w / total_weight) for box, w in boxes]

    blended = np.zeros((3, 3), dtype=float)
    for box, w in boxes:
        blended += box.weights * w

    return StyleBox(name=name, weights=blended)


# ---------------------------------------------------------------------------
# Example — Avantis SCV vs Avantis Global Equity
# These are approximate weights; replace with current fund data.
# ---------------------------------------------------------------------------

# AVSG (Small Cap Value) — heavily concentrated in small/value
avsg = StyleBox(
    name="Avantis Global Small Cap Value",
    weights=[
        [0.00, 0.00, 0.00],   # Large
        [0.08, 0.07, 0.03],   # Mid
        [0.45, 0.26, 0.12],   # Small
    ]
)

# AVCG (Global Equity) — broad market, large-cap dominated, slight value tilt
avcg = StyleBox(
    name="Avantis Global Equity",
    weights=[
        [0.19, 0.29, 0.13],   # Large
        [0.10, 0.11, 0.06],   # Mid
        [0.05, 0.05, 0.03],   # Small
    ]
)

# AVEM (Emerging Markets) — more mid/small, more value than growth
avem = StyleBox(
    name="Avantis Emerging Markets",
    weights=[
        [0.24, 0.28, 0.24],   # Large
        [0.09, 0.06, 0.04],   # Mid
        [0.03, 0.02, 0.01],   # Small
    ]
)

if __name__ == "__main__":
    # e.g. 75% AVCG, 15% AVSG, 10% AVEM
    portfolio = blend([
    (avcg, 0.75),
    (avsg, 0.15),
    (avem, 0.10),
    ], name="Example Portfolio")
    portfolio.display()