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

import io
import os
from collections.abc import Sequence
from pathlib import Path

from .io_utils import atomic_write_bytes_many, check_writable
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
            "Install the plot extra: pip install -e '.[plot]' from the checkout"
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


def disambiguated_labels(results: Sequence[SweepResult]) -> list[str]:
    """One label per result, such that no two labels in a chart are equal.

    A name that appears **once** is returned exactly as it is. Only names that
    collide gain a suffix, and the suffix is the result's 1-based position in
    the sequence — which is `sorted(results_dir.glob("*.json"))` order on the
    CLI path, so an operator can map ``#3`` back to the third file.

    That means the ordinals **can have gaps**, and the gaps are the point:
    ``["a", "b", "a"]`` labels as ``["a #1", "b", "a #3"]``, not ``a #1`` /
    ``a #2``. A per-name occurrence counter would read more naturally and would
    point at nothing — "the second `a`" is not a file you can open, while "the
    third result" is.

    #69 established that two distinct `SweepResult`s can share an
    `embedder_name` — D-007 writes one file per run, so the same provider run
    twice yields two same-named results — and stopped keying the frontier
    *colour* on that name. The annotation eleven lines below it still did, so
    the chart said "openai-3-small is on the frontier" and "openai-3-small is
    dominated" with nothing to tell a reader which point was which run (#151).

    **Sparse, not uniform.** This is the string analogue of
    `llm-cost-optimizer` D-021's set-wide widening, with one deliberate
    difference: there every label shares the widened format, because a column
    of numbers at mixed precision reads as mixed quantities. Here the labels are
    names, an unsuffixed name is not ambiguous about anything, and decorating
    every point to disambiguate two of them would churn every chart this repo
    has ever produced. So only the collision is touched.

    **What this is not.** The honest fix for "which run is this point" is a run
    id on `SweepResult` — the loader drops the filename that actually
    distinguishes the two, and `render_pareto` takes only
    `Sequence[SweepResult]`. That is a schema change touching `from_dict`,
    `to_dict`, every committed result JSON and D-007, and it is deliberately out
    of scope here. If provenance ever becomes a first-class field, labelling
    from it is strictly better than an ordinal.

    **A suffix can collide with a name** (#239). A name is any string, so it
    can already end in `` #<n>``: ``["m #2", "m", "m"]`` used to label as
    ``m #2`` / ``m #2`` / ``m #3``, and through `sweep plot` two runs were
    called ``hash #2`` with only one on the frontier, which is #151 again. So a
    candidate is re-suffixed with the same ordinal while it equals any input
    name or any label already handed out: that example becomes ``m #2`` /
    ``m #2 #2`` / ``m #3``. The last suffix is still the position, a unique
    name is still untouched, and the loop ends because each pass makes the
    candidate longer and the set it must avoid is finite.
    """
    counts: dict[str, int] = {}
    for r in results:
        counts[r.embedder_name] = counts.get(r.embedder_name, 0) + 1
    # Every input name is reserved up front, so a suffixed label cannot take a
    # name a unique result keeps unchanged, wherever that result sits (#239).
    taken = set(counts)
    labels: list[str] = []
    for i, r in enumerate(results):
        if counts[r.embedder_name] == 1:
            labels.append(r.embedder_name)
            continue
        label = f"{r.embedder_name} #{i + 1}"
        while label in taken:
            label = f"{label} #{i + 1}"
        taken.add(label)
        labels.append(label)
    return labels


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
        # The *disambiguated* label, not the bare name (#151). When two runs of
        # one provider are plotted and one dominates the other, this sentence
        # named a string that is on the chart twice — once on the frontier and
        # once dominated — so "X dominates every other model" was unreadable
        # against a chart showing an X being dominated. Matched by identity for
        # the reason #69 gives one function down: the name is not a key.
        labels = disambiguated_labels(results)
        winner = next(
            (label for label, r in zip(labels, results, strict=True) if r is frontier[0]),
            frontier[0].embedder_name,
        )
        return f"Cost vs recall@5 — {winner} dominates every other model"
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


def _require_suffix_matches(path: Path, fmt: str) -> None:
    """Refuse an output path whose extension names another format (#220).

    `--out-png chart.svg` is a contradiction the flag cannot settle silently:
    writing PNG bytes under `.svg` breaks every viewer, and writing an SVG (what
    an extension-inferred `savefig` did) ignores the flag. No extension is fine:
    the format comes from the flag and the file is written at exactly `path`.
    Case-insensitive, so `pareto.PNG` is a PNG.
    """
    if path.suffix and path.suffix.lower() != f".{fmt}":
        raise ValueError(
            f"the {fmt.upper()} output path {str(path)!r} ends in {path.suffix!r}; "
            f"use a .{fmt} path or no extension"
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

    # Before the matplotlib import (#167): an incomparable result set is bad
    # input whether or not the plotting extra is installed, and it must not be
    # reported as "matplotlib missing" (exit 3) instead of refused (exit 2).
    frontier = pareto_frontier(results)
    # An empty path is refused rather than dropped (#228): the `is None` check
    # above let `--out-png ""` through, this list then filtered it out on
    # truthiness, and the command wrote nothing and exited 0.
    for p, fmt in ((out_png, "png"), (out_svg, "svg")):
        if p is not None and str(p).strip() == "":
            raise ValueError(f"the {fmt.upper()} output path must be non-empty; got {str(p)!r}")
    outputs = [(Path(p), fmt) for p, fmt in ((out_png, "png"), (out_svg, "svg")) if p is not None]
    for path, fmt in outputs:
        _require_suffix_matches(path, fmt)
    # Two flags naming one file (#228). #220 lets a path have no extension, so
    # `--out-png same --out-svg same` passed both suffix checks; the PNG was
    # written, then overwritten by the SVG, and the summary reported both.
    # Compared resolved (`./same`, a symlinked directory) and, for a file that
    # already exists, by identity -- which also catches `SAME` on a
    # case-insensitive filesystem. A case variant of a file that does not exist
    # yet is not detected there.
    if len(outputs) == 2:
        png_path, svg_path = outputs[0][0], outputs[1][0]
        same = png_path.resolve() == svg_path.resolve() or (
            png_path.exists() and svg_path.exists() and os.path.samefile(png_path, svg_path)
        )
        if same:
            raise ValueError(
                f"--out-png {str(png_path)!r} and --out-svg {str(svg_path)!r} are the same "
                "file; the SVG would overwrite the PNG"
            )
    # Every output is writable before any is written (#234). The loop at the
    # bottom saves them in turn, so an unwritable --out-svg (its parent a file,
    # an existing directory) raised only after --out-png had been replaced, and
    # the command exited 2 "failed to write plot" with the old chart gone. The
    # same preflight #200 gave `sweep run --out`; raises the OSError the CLI
    # already maps to exit 2.
    for path, _fmt in outputs:
        check_writable(path)

    plt = _import_matplotlib()
    # Match frontier membership by object identity, not `embedder_name`:
    # `pareto_frontier` returns the actual input objects, and two distinct
    # results can share a name (D-007 writes one file per run, so the same
    # provider run twice yields two same-named results). Name-matching would
    # paint a dominated run with the frontier highlight (#69).
    frontier_ids = {id(r) for r in frontier}

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    # Labels that survive two runs of the same provider (#151). Computed over
    # the whole sequence, because whether a name needs disambiguating is a
    # property of the set rather than of the point.
    labels = disambiguated_labels(results)

    # All points first (non-frontier in muted color, frontier in highlight).
    for label, r in zip(labels, results, strict=True):
        x = r.cost_per_million_tokens
        y = _recall_at_5(r)
        if id(r) in frontier_ids:
            ax.scatter(x, y, s=90, color="#d62728", zorder=3, edgecolor="black", linewidth=0.5)
        else:
            ax.scatter(x, y, s=70, color="#7f7f7f", zorder=2, edgecolor="black", linewidth=0.3)
        # Label every point with its disambiguated name (#151). `r.embedder_name`
        # is not unique — that is #69's own finding, three lines above.
        #
        # `parse_math=False` (#214): matplotlib reads any text holding a pair of
        # unescaped `$` as mathtext. A name is data, not markup -- `tier $0.02$
        # small` was drawn re-typeset with its `$` gone, and `cost $\frac$ model`
        # raised a ParseSyntaxException that `sweep plot` printed as a bare
        # `error:` at exit 2, for a result `sweep aggregate` accepts.
        ax.annotate(
            label,
            (x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            parse_math=False,
        )

    # Polyline through the frontier only when ≥2 distinct frontier points.
    if _should_draw_polyline(frontier):
        xs = [r.cost_per_million_tokens for r in frontier]
        ys = [_recall_at_5(r) for r in frontier]
        ax.plot(xs, ys, color="#d62728", linewidth=1.6, zorder=2, linestyle="--", alpha=0.8)

    ax.set_xlabel("Cost per million tokens ($)")
    ax.set_ylabel("Recall@5")
    if title is None:
        # The auto-chosen title quotes a label (the single winner, #151), so it
        # is drawn literally for the same reason as the labels (#214). An
        # operator's own `--title` keeps matplotlib's default, mathtext included.
        ax.set_title(_default_title(results, frontier), parse_math=False)
    else:
        ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    # Pad axes so labels don't clip -- for every non-empty result set, single
    # included (#139). The `len(results) > 1` this replaces read as "one point
    # needs no padding", and it guarded against a problem `_axis_limits` had
    # already solved: both pads have a floor (`max(0.05, ...)`, `max(0.02, ...)`),
    # so a zero spread still yields a real range. What it actually did was skip
    # the domain clamps on the branch this repo ships from -- the committed
    # result has `cost_per_million_tokens: 0.0`, and `docs/pareto.svg` carried
    # x ticks at -0.04 and -0.02. #137 called this "the guard covered one
    # operand of the expression it protects"; this is the same shape one level
    # up, at the call site rather than inside it.
    (x_lo, x_hi), (y_lo, y_hi) = _axis_limits(results)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()

    # `format=` from the flag, never inferred (#220). Without it matplotlib
    # picks the format from the extension: `--out-svg pareto` wrote a PNG to
    # `pareto.png` while the summary reported `pareto`, and `--out-png x.svg`
    # wrote an SVG. With it the bytes match the flag and land at exactly `path`.
    #
    # Rendered into memory, then written atomically (#245).
    # `fig.savefig(path)` wrote straight onto the destination, so a write that
    # failed partway (ENOSPC) left a truncated chart in place of the committed
    # `docs/pareto.png` -- the "never partial" promise `io_utils` makes for
    # every other artifact. Both charts are rendered before either is written,
    # so a render failure replaces nothing either. The writer creates the
    # parent and writes through a symlink, as `savefig` did.
    try:
        rendered = []
        for path, fmt in outputs:
            buf = io.BytesIO()
            fig.savefig(buf, dpi=150, format=fmt)
            rendered.append((path, buf.getvalue()))
    finally:
        plt.close(fig)
    # Both staged before either is renamed (#251): written one after the other,
    # a disk-full on the SVG came after the PNG had been replaced, and the
    # command exited 2 with two charts of different results. The error now
    # names the chart that failed.
    atomic_write_bytes_many(rendered)
    by_fmt = {fmt: path for path, fmt in outputs}
    return frontier, by_fmt.get("png"), by_fmt.get("svg")
