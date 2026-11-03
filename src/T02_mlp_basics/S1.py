import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="darkgrid")

    x_data = np.array([1, 2, 3, 4])
    y_data = np.array([1, 3, 3, 5])
    w_slider = mo.ui.slider(start=-2, stop=3, step=0.1, value=1, label="w")
    b_slider = mo.ui.slider(start=-2, stop=5, step=0.1, value=0, label="b")
    return b_slider, mo, np, plt, sns, w_slider, x_data, y_data


@app.cell(hide_code=True)
def _(b_slider, mo, np, plt, sns, w_slider, x_data, y_data):
    w = w_slider.value
    b = b_slider.value
    mse = np.mean((y_data - (w * x_data + b)) ** 2)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = w * x_line + b
    fig, ax = plt.subplots()
    ax.set_title(f"MSE = {mse:.3f}")
    sns.scatterplot(x=x_data, y=y_data, color="tab:orange", label="Data", s=80, ax=ax)
    sns.lineplot(
        x=x_line,
        y=y_line,
        color="tab:blue",
        label=f"y = {w:.1f}x + {b:.1f}",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 5)
    ax.set_ylim(-11, 18)
    ax.legend()
    ax.grid(True)

    output = mo.vstack([mo.hstack([w_slider, b_slider]), fig])
    output
    return


@app.cell(hide_code=True)
def _(b_slider, mo, np, w_slider, x_data, y_data):
    import plotly.graph_objects as go

    w_grid = np.linspace(-2, 3, 101)
    b_grid = np.linspace(-2, 5, 101)
    W, B = np.meshgrid(w_grid, b_grid)
    residuals = y_data[None, None, :] - (
        W[:, :, None] * x_data[None, None, :] + B[:, :, None]
    )
    mse_surface = np.mean(residuals**2, axis=2)

    cur_w = w_slider.value
    cur_b = b_slider.value
    cur_mse = np.mean((y_data - (cur_w * x_data + cur_b)) ** 2)

    surface_fig = go.Figure(
        data=[
            go.Surface(x=w_grid, y=b_grid, z=mse_surface, colorscale="Viridis"),
            go.Scatter3d(
                x=[cur_w],
                y=[cur_b],
                z=[cur_mse],
                mode="markers",
                marker=dict(size=6, color="red"),
                name="Current (w, b)",
            ),
        ]
    )
    surface_fig.update_layout(
        title="MSE surface over w and b",
        scene=dict(xaxis_title="w", yaxis_title="b", zaxis_title="MSE"),
        height=600,
    )
    mo.ui.plotly(surface_fig)
    return (go,)


@app.cell
def _(go, mo, np):
    _input_pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    _targets = np.array([0, 1, 1, 2])
    _sheet_values = np.linspace(0, 1, 25)
    _sheet_x1, _sheet_x2 = np.meshgrid(_sheet_values, _sheet_values)
    _sheet_y = _sheet_x1 + _sheet_x2
    _scatter_fig = go.Figure(
        data=[
            go.Surface(
                x=_sheet_x1,
                y=_sheet_x2,
                z=_sheet_y,
                opacity=0.4,
                colorscale="Blues",
                showscale=False,
                name="y = x1 + x2",
                hovertemplate="x1=%{x}<br>x2=%{y}<br>y=%{z}<extra></extra>",
            ),
            go.Scatter3d(
                x=_input_pairs[:, 0],
                y=_input_pairs[:, 1],
                z=_targets,
                mode="markers+text",
                text=[f"y={_target}" for _target in _targets],
                textposition="top center",
                marker=dict(
                    size=8,
                    color=_targets,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=2,
                    colorbar=dict(title="y"),
                ),
                name="Data points",
                hovertemplate="x1=%{x}<br>x2=%{y}<br>y=%{z}<extra></extra>",
            ),
        ]
    )
    _scatter_fig.update_layout(
        title="Input pairs and linear sheet",
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y"),
        height=550,
    )
    mo.ui.plotly(_scatter_fig)
    return


@app.cell
def _(go, mo, np):
    _xor_pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    _xor_labels = np.array([0, 1, 1, 0])
    _xor_fig = go.Figure(
        data=[
            go.Scatter3d(
                x=_xor_pairs[:, 0],
                y=_xor_pairs[:, 1],
                z=_xor_labels,
                mode="markers+text",
                text=[str(_label) for _label in _xor_labels],
                textposition="top center",
                marker=dict(
                    size=9,
                    color=_xor_labels,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="XOR label"),
                ),
                name="XOR outputs",
                hovertemplate="x1=%{x}<br>x2=%{y}<br>XOR=%{z}<extra></extra>",
            )
        ]
    )
    _xor_fig.update_layout(
        title="XOR problem",
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="XOR label"),
        height=550,
    )
    mo.ui.plotly(_xor_fig)
    return


if __name__ == "__main__":
    app.run()
