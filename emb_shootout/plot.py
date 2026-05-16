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

from .pareto import pareto_frontier
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
    frontier_names = {r.embedder_name for r in frontier}

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    # All points first (non-frontier in muted color, frontier in highlight).
    for r in results:
        x = r.cost_per_million_tokens
        y = float(r.recall_at_k.get(5, 0.0))
        if r.embedder_name in frontier_names:
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
    distinct_frontier_coords = {
        (r.cost_per_million_tokens, float(r.recall_at_k.get(5, 0.0))) for r in frontier
    }
    if len(distinct_frontier_coords) >= 2:
        xs = [r.cost_per_million_tokens for r in frontier]
        ys = [float(r.recall_at_k.get(5, 0.0)) for r in frontier]
        ax.plot(xs, ys, color="#d62728", linewidth=1.6, zorder=2, linestyle="--", alpha=0.8)

    ax.set_xlabel("Cost per million tokens ($)")
    ax.set_ylabel("Recall@5")
    if title is None:
        if len(results) == 1:
            title = "Pareto frontier — single point (real-provider runs pending)"
        elif len(distinct_frontier_coords) < 2:
            title = "Pareto frontier — all points co-located on at least one axis"
        else:
            title = "Cost vs recall@5 — Pareto frontier highlighted in red"
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    # Pad axes so labels don't clip.
    if len(results) > 1:
        x_vals = [r.cost_per_million_tokens for r in results]
        y_vals = [float(r.recall_at_k.get(5, 0.0)) for r in results]
        x_pad = max(0.05, (max(x_vals) - min(x_vals)) * 0.08)
        y_pad = max(0.02, (max(y_vals) - min(y_vals)) * 0.08)
        ax.set_xlim(min(x_vals) - x_pad, max(x_vals) + x_pad)
        ax.set_ylim(min(y_vals) - y_pad, min(1.0, max(y_vals) + y_pad))

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
