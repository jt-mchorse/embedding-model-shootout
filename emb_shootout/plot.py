"""Pareto frontier plot for the embedding shootout (issue #3).

Matplotlib is lazy-imported so the base install stays dep-free. Install the
plot extra to render:

    pip install -e '.[plot]'
    emb-shootout sweep plot --results-dir results --out-png docs/pareto.png \\
                            --out-svg docs/pareto.svg

The figure plots every `SweepResult` as a point on (cost_per_million_tokens,
recall@5), highlights the Pareto frontier in a contrasting color, and draws
a connecting polyline through the frontier only when ≥2 distinct frontier
points exist. With a single result the title says so explicitly — the chart
is real, just narrow.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .pareto import _recall_at_5, pareto_frontier
from .sweep import SweepResult


def _import_matplotlib():
    """Import matplotlib only when actually rendering. Returns (plt, FigureClass)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless — CI has no display
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to render Pareto plots. "
            "Install the plot extra: pip install 'embedding-model-shootout[plot]'"
        ) from exc
    return plt


#: Recall@5 is a proportion: `SweepResult.__post_init__` enforces
#: ``0.0 <= v <= 1.0`` for every ``recall_at_k`` value. The plot's y-axis is
#: that same domain, so padding may not run past either end of it.
_RECALL_DOMAIN = (0.0, 1.0)


def _should_draw_polyline(frontier: Sequence[SweepResult]) -> bool:
    """Whether the frontier has ≥2 distinct coordinates to connect.

    Deliberately **not** `not _is_colocated(results)`. This is the other half of
    #137 and the trap in fixing it: once the title stops asking a frontier
    question, the tempting next move is to make the polyline "consistent" and
    ask the results question too. It is not the same question, and it fails in
    both directions —

    - two models at the same cost with different recalls give a genuine
      two-point frontier that `_is_colocated(results)` would suppress the line
      for;
    - a one-point frontier over separated results would gain a line through a
      single point.

    The polyline was always right. It asks "can I draw a line through these
    points", which is a property of the points being drawn.
    """
    coords = {(r.cost_per_million_tokens, _recall_at_5(r)) for r in frontier}
    return len(coords) >= 2


def _is_colocated(results: Sequence[SweepResult]) -> bool:
    """True when every result shares one cost, or every result shares one recall.

    A property of **all results**, which is what the phrase "all points
    co-located on at least one axis" claims. The title used to derive it from
    ``len(distinct_frontier_coords) < 2`` instead -- a property of the
    *frontier* -- and the two only coincide when the frontier is the whole
    population (#137).

    They come apart in the case this repo exists to produce. When one model
    dominates every other, the frontier collapses to a single point while the
    results are spread across both axes, and the figure was captioned "all
    points co-located on at least one axis" over a plot showing them clearly
    separated. Measured on two models at ($1.00, 0.90) and ($2.00, 0.50):
    frontier of 1, two distinct costs, two distinct recalls, and a title
    asserting the opposite.

    `distinct_frontier_coords` is still exactly right for the *polyline* -- you
    cannot draw a line through one point -- which is why the value survives and
    only this second consumer changed. One computed quantity answering two
    different questions is the whole defect.
    """
    costs = {r.cost_per_million_tokens for r in results}
    recalls = {_recall_at_5(r) for r in results}
    return len(costs) == 1 or len(recalls) == 1


def _default_title(results: Sequence[SweepResult], frontier: Sequence[SweepResult]) -> str:
    """The figure title when the caller supplies none.

    Split out of `render_pareto` so it can be tested without matplotlib. The
    render path has no CI coverage (#111, gated on D-008), which is how a
    caption stating the opposite of the data survived; a decision that is a
    pure function of two lists does not need the `plot` extra to be checked.
    """
    if len(results) == 1:
        return "Pareto frontier — single point (real-provider runs pending)"
    if _is_colocated(results):
        return "Pareto frontier — all points co-located on at least one axis"
    if len(frontier) == 1:
        return f"Cost vs recall@5 — {frontier[0].embedder_name} dominates every other model"
    return "Cost vs recall@5 — Pareto frontier highlighted in red"


def _axis_limits(results: Sequence[SweepResult]) -> tuple[tuple[float, float], tuple[float, float]]:
    """``((xmin, xmax), (ymin, ymax))`` with padding, clamped to each domain.

    The upper y bound was already clamped to 1.0; the lower was not, so a run
    of low recalls produced ``ylim (-0.02, 0.04)`` -- an axis extending into
    negative recall, a region `SweepResult.__post_init__` refuses to store
    (#137). The guard covered one operand of the expression it protects, the
    same shape as the title bug above it.

    Cost has a floor at 0 for the same reason (`cost_per_million_tokens` is
    validated ``>= 0.0``) and no documented ceiling, so only its lower bound is
    clamped.
    """
    x_vals = [r.cost_per_million_tokens for r in results]
    y_vals = [_recall_at_5(r) for r in results]
    x_pad = max(0.05, (max(x_vals) - min(x_vals)) * 0.08)
    y_pad = max(0.02, (max(y_vals) - min(y_vals)) * 0.08)
    lo, hi = _RECALL_DOMAIN
    return (
        (max(0.0, min(x_vals) - x_pad), max(x_vals) + x_pad),
        (max(lo, min(y_vals) - y_pad), min(hi, max(y_vals) + y_pad)),
    )


def render_pareto(
    results: Sequence[SweepResult],
    *,
    out_png: Path | str | None = None,
    out_svg: Path | str | None = None,
    title: str | None = None,
) -> tuple[list[SweepResult], Path | None, Path | None]:
    """Render the (cost, recall@5) plot to PNG and/or SVG.

    Returns `(frontier, png_path_or_None, svg_path_or_None)`. At least one of
    `out_png` or `out_svg` must be provided. Empty `results` raises — the plot
    has nothing to show, and silently writing a blank figure would mask a
    missing-results bug.
    """
    if not results:
        raise ValueError("results must be non-empty; nothing to plot")
    if out_png is None and out_svg is None:
        raise ValueError("must provide at least one of out_png or out_svg")

    plt = _import_matplotlib()

    frontier = pareto_frontier(results)
    # Match frontier membership by object identity, not `embedder_name`:
    # `pareto_frontier` returns the actual input objects, and two distinct
    # results can share a name (D-007 writes one file per run, so the same
    # provider run twice yields two same-named results). Name-matching would
    # paint a dominated run with the frontier highlight (#69).
    frontier_ids = {id(r) for r in frontier}

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    # All points first (non-frontier in muted color, frontier in highlight).
    for r in results:
        x = r.cost_per_million_tokens
        y = _recall_at_5(r)
        if id(r) in frontier_ids:
            ax.scatter(x, y, s=90, color="#d62728", zorder=3, edgecolor="black", linewidth=0.5)
        else:
            ax.scatter(x, y, s=70, color="#7f7f7f", zorder=2, edgecolor="black", linewidth=0.3)
        # Label every point with the embedder name.
        ax.annotate(
            r.embedder_name,
            (x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
        )

    # Polyline through the frontier only when ≥2 distinct frontier points.
    if _should_draw_polyline(frontier):
        xs = [r.cost_per_million_tokens for r in frontier]
        ys = [_recall_at_5(r) for r in frontier]
        ax.plot(xs, ys, color="#d62728", linewidth=1.6, zorder=2, linestyle="--", alpha=0.8)

    ax.set_xlabel("Cost per million tokens ($)")
    ax.set_ylabel("Recall@5")
    if title is None:
        title = _default_title(results, frontier)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    # Pad axes so labels don't clip.
    if len(results) > 1:
        (x_lo, x_hi), (y_lo, y_hi) = _axis_limits(results)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()

    png_path = Path(out_png) if out_png else None
    svg_path = Path(out_svg) if out_svg else None
    for p in (png_path, svg_path):
        if p is None:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
    plt.close(fig)
    return frontier, png_path, svg_path
