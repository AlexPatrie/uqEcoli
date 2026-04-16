"""Jupyter notebook integration for UQ results.

Provides Rich table display, ipywidgets-based interactive parameter
exploration, and quick plotting from ``QuantifyResult`` or exported
artifacts.

Usage:
    from uq.jupyter import display_sobol, UQWidget

    # From QuantifyResult
    display_sobol(result)

    # From exported artifacts
    widget = UQWidget.from_export("./uq_results")
    widget.show()

Requires:
    uv pip install ipywidgets plotly
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ── Legendre PCE evaluation ──


def _legendre_eval(x_norm, coeffs, mi):
    max_ord = int(mi.max()) if mi.size else 0
    n_p = mi.shape[1]
    P = np.zeros((max_ord + 1, n_p))
    P[0, :] = 1.0
    if max_ord >= 1:
        P[1, :] = x_norm
    for n in range(2, max_ord + 1):
        P[n, :] = ((2 * n - 1) * x_norm * P[n - 1, :] - (n - 1) * P[n - 2, :]) / n
    result = 0.0
    for t in range(len(coeffs)):
        term = float(coeffs[t])
        for p in range(n_p):
            term *= P[mi[t, p], p]
        result += term
    return result


def _normalize(x, bounds):
    return 2.0 * (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0


# ── Display functions ──


def display_sobol(result: Any) -> None:
    """Display Sobol indices as a Rich table in a Jupyter notebook.

    Args:
        result: A ``QuantifyResult`` from ``uq.workflow.quantify()``.
    """
    try:
        from IPython.display import display
        from rich.console import Console
        from rich.table import Table
    except ImportError as e:
        raise ImportError("display_sobol requires IPython and rich") from e

    console = Console()

    table = Table(title="Population Sobol Indices (S_Ti)", show_header=True)
    table.add_column("Parameter", style="bold")
    table.add_column("S_i (first)", justify="right")
    table.add_column("S_Ti (total)", justify="right")
    table.add_column("Bar", justify="left")

    s1 = result.strategy1.sobol
    for i, name in enumerate(result.parameter_names):
        fi = s1.first_order[i]
        ti = s1.total_order[i]
        bar = "█" * int(ti * 30)
        table.add_row(name, f"{fi:.4f}", f"{ti:.4f}", bar)

    console.print(table)

    # Strategy 4 summary
    if result.strategy4_per_stage:
        n = len(result.strategy4_per_stage)
        console.print(f"\n[bold]Strategy 4:[/bold] {n} growth stages")
        for j, r in enumerate(result.strategy4_per_stage):
            top_idx = int(np.argmax(r.sobol.total_order))
            top_name = result.parameter_names[top_idx]
            top_val = r.sobol.total_order[top_idx]
            lo, hi = int(100 * j / n), int(100 * (j + 1) / n)
            console.print(f"  θ {lo}-{hi}%: top = {top_name} ({top_val:.3f})")


def plot_spectrogram(result: Any) -> Any:
    """Return a Plotly heatmap of Strategy 4 Sobol indices.

    Args:
        result: A ``QuantifyResult``.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    names = result.parameter_names
    short = [n.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass") for n in names]
    stages = result.strategy4_per_stage
    n = len(stages)
    z = [[stages[j].sobol.total_order[i] for j in range(n)] for i in range(len(names))]
    x_labels = [f"θ{j}" for j in range(n)]

    fig = go.Figure(data=go.Heatmap(z=z, x=x_labels, y=short, colorscale="Viridis"))
    fig.update_layout(
        title="Sensitivity Spectrogram — S_Ti by Growth Stage",
        xaxis_title="Cell Cycle Stage",
        height=350, width=800,
    )
    return fig


# ── Interactive widget ──


class UQWidget:
    """Interactive ipywidgets-based parameter explorer.

    Provides sliders for each parameter and a reactive Plotly response
    curve that updates when sliders change.
    """

    def __init__(
        self,
        params: list[str],
        bounds: np.ndarray,
        coeffs: np.ndarray,
        mi: np.ndarray,
        sobol: dict[str, float] | None = None,
    ):
        self.params = params
        self.bounds = bounds
        self.coeffs = coeffs
        self.mi = mi
        self.sobol = sobol or {}

    @classmethod
    def from_export(cls, results_dir: str | Path) -> "UQWidget":
        """Load from a ``uq quantify`` export directory."""
        results_dir = Path(results_dir)
        data = json.loads((results_dir / "uq_results.json").read_text())
        params = list(data["parameters"].keys())
        sobol = data["phase1_population"]["sobol_total_order"]

        pop_dir = results_dir / "population_surrogate"
        coeffs = np.load(pop_dir / "coefficients.npy")
        mi = np.load(pop_dir / "multi_indices.npy")
        bounds = np.load(pop_dir / "input_bounds.npy")

        return cls(params=params, bounds=bounds, coeffs=coeffs, mi=mi, sobol=sobol)

    @classmethod
    def from_result(cls, result: Any) -> "UQWidget":
        """Create from a ``QuantifyResult`` object."""
        s = result.strategy1.surrogate
        sobol_dict = {
            n: float(result.strategy1.sobol.total_order[i])
            for i, n in enumerate(result.parameter_names)
        }
        return cls(
            params=result.parameter_names,
            bounds=s.input_bounds,
            coeffs=s.coefficients,
            mi=s.multi_indices,
            sobol=sobol_dict,
        )

    def show(self) -> None:
        """Display interactive sliders + response curve in a notebook."""
        try:
            import ipywidgets as widgets
            import plotly.graph_objects as go
            from IPython.display import display
        except ImportError as e:
            raise ImportError("UQWidget.show() requires ipywidgets and plotly") from e

        n_params = len(self.params)
        pcolors = ["#00f0ff", "#ff3366", "#00ff88", "#ffaa00", "#aa66ff", "#ff6600"]

        sliders = {}
        for i, p in enumerate(self.params):
            lo, hi = float(self.bounds[i, 0]), float(self.bounds[i, 1])
            short = p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass_frac")
            sliders[p] = widgets.FloatSlider(
                value=(lo + hi) / 2,
                min=lo, max=hi,
                step=(hi - lo) / 200,
                description=short,
                style={"description_width": "150px"},
                layout=widgets.Layout(width="100%"),
            )

        fig_widget = go.FigureWidget()
        readout = widgets.HTML(value="<h3>Ŷ = ---</h3>")

        def update(*args):
            x = np.array([sliders[p].value for p in self.params])
            x_norm = _normalize(x, self.bounds)
            y_hat = _legendre_eval(x_norm, self.coeffs, self.mi)
            readout.value = f"<h3 style='color:#33ff99'>Ŷ = {y_hat:.4f}</h3>"

            with fig_widget.batch_update():
                fig_widget.data = []
                n_sweep = 60
                for pi, p in enumerate(self.params):
                    lo, hi = float(self.bounds[pi, 0]), float(self.bounds[pi, 1])
                    sv = np.linspace(lo, hi, n_sweep)
                    sv_norm = (sv - lo) / (hi - lo + 1e-12)
                    sy = []
                    for v in sv:
                        x_sw = x.copy()
                        x_sw[pi] = v
                        sy.append(_legendre_eval(_normalize(x_sw, self.bounds), self.coeffs, self.mi))
                    short = p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass_frac")
                    color = pcolors[pi % len(pcolors)]
                    fig_widget.add_scatter(x=sv_norm, y=sy, mode="lines", name=short, line=dict(color=color))

                fig_widget.update_layout(
                    title=f"PCE Response Curves — Ŷ = {y_hat:.4f}",
                    xaxis_title="Normalized parameter [0=min, 1=max]",
                    yaxis_title="Ŷ",
                    height=350,
                )

        for s in sliders.values():
            s.observe(update, names="value")

        update()  # Initial render

        slider_box = widgets.VBox(list(sliders.values()))
        display(widgets.VBox([readout, widgets.HBox([slider_box, fig_widget])]))
