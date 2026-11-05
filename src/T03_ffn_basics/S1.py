import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    return go, mo, np


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
                name="XOR",
                hovertemplate="x1=%{x}<br>x2=%{y}<br>XOR=%{z}<extra></extra>",
            ),
            go.Surface(
                x=np.linspace(-0.2, 1.2, 2),
                y=np.linspace(-0.2, 1.2, 2),
                z=np.full((2, 2), 0.5),
                colorscale=[[0, "tomato"], [1, "tomato"]],
                opacity=0.4,
                showscale=False,
                name="ŷ = 0.5",
                showlegend=True,
                hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<br>ŷ=0.5<extra></extra>",
            ),
        ]
    )
    _xor_fig.update_layout(
        title="XOR problem",
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="XOR label"),
        height=550,
    )
    mo.ui.plotly(_xor_fig)
    return


@app.cell
def _(mo):
    xor_w1_slider = mo.ui.slider(
        start=-10, stop=10, step=0.5, value=5, label="w1", show_value=True
    )
    xor_w2_slider = mo.ui.slider(
        start=-10, stop=10, step=0.5, value=5, label="w2", show_value=True
    )
    xor_b_slider = mo.ui.slider(
        start=-15, stop=15, step=0.25, value=-2.5, label="b", show_value=True
    )
    return xor_b_slider, xor_w1_slider, xor_w2_slider


@app.cell
def _(go, mo, np, xor_b_slider, xor_w1_slider, xor_w2_slider):
    _xor_pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    _xor_labels = np.array([0, 1, 1, 0])
    _X = _xor_pairs.astype(float)

    # For reference: logistic regression fit by gradient descent on (x1, x2).
    # By XOR's symmetry it collapses to w = 0, b = 0 (flat p = 0.5, no line).
    _wf = np.zeros(2)
    _bf = 0.0
    _lam, _lr = 0.001, 0.5
    for _ in range(20000):
        _p = 1 / (1 + np.exp(-(_X @ _wf + _bf)))
        _wf -= _lr * (_X.T @ (_p - _xor_labels) / len(_xor_labels) + _lam * _wf)
        _bf -= _lr * np.mean(_p - _xor_labels)

    # Hand-picked line from the sliders
    _w = np.array([xor_w1_slider.value, xor_w2_slider.value], dtype=float)
    _b = float(xor_b_slider.value)

    _p_pts = 1 / (1 + np.exp(-(_X @ _w + _b)))
    _correct = (_p_pts >= 0.5).astype(int) == _xor_labels
    _acc = np.mean(_correct)

    # Probability surface over the (x1, x2) plane
    _g = np.linspace(-0.2, 1.2, 50)
    _G1, _G2 = np.meshgrid(_g, _g)
    _P = 1 / (1 + np.exp(-(_w[0] * _G1 + _w[1] * _G2 + _b)))

    _traces = [
        go.Scatter3d(
            x=_xor_pairs[:, 0],
            y=_xor_pairs[:, 1],
            z=_xor_labels,
            mode="markers+text",
            text=[
                f"{_label} {'✓' if _ok else '✗'}"
                for _label, _ok in zip(_xor_labels, _correct)
            ],
            textposition="top center",
            marker=dict(
                size=9,
                color=_xor_labels,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                line=dict(color=["black" if _ok else "red" for _ok in _correct], width=4),
            ),
            name="XOR",
            hovertemplate="x1=%{x}<br>x2=%{y}<br>XOR=%{z}<extra></extra>",
        ),
        go.Surface(
            x=_g,
            y=_g,
            z=_P,
            colorscale="Viridis",
            cmin=0,
            cmax=1,
            opacity=0.6,
            showscale=False,
            name="Logistic",
            showlegend=True,
            hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<br>p=%{z:.3f}<extra></extra>",
        ),
    ]

    # Decision line: w1*x1 + w2*x2 + b = 0  (only exists if w is not ~0)
    _norm = np.linalg.norm(_w)
    if _norm > 1e-6:
        _t = np.linspace(-2, 2, 400)
        _n = _w / _norm
        _line = -_b / _norm * _n + np.outer(_t, [-_n[1], _n[0]])
        _line = _line[np.all((_line >= -0.2) & (_line <= 1.2), axis=1)]
        _traces.append(
            go.Scatter3d(
                x=_line[:, 0],
                y=_line[:, 1],
                z=np.full(len(_line), 0.5),
                mode="lines",
                line=dict(color="red", width=6),
                name="p = 0.5",
                hoverinfo="skip",
            )
        )

    _x_fig = go.Figure(data=_traces)
    _x_fig.update_layout(
        title="Logistic regression on XOR in (x1, x2)",
        scene=dict(
            xaxis=dict(title="x1", range=[-0.2, 1.2]),
            yaxis=dict(title="x2", range=[-0.2, 1.2]),
            zaxis=dict(title="XOR label", range=[-0.2, 1.2]),
        ),
        height=550,
    )
    mo.vstack(
        [
            mo.hstack([xor_w1_slider, xor_w2_slider, xor_b_slider], justify="start"),
            mo.md(
                f"""
    $p = \\sigma({_w[0]:.2f}\\,x_1 + ({_w[1]:.2f})\\,x_2 + ({_b:.2f}))$

    Predicted $p$ at (0,0), (0,1), (1,0), (1,1): {", ".join(f"{_v:.3f}" for _v in _p_pts)}
    — **{int(_correct.sum())} / 4 correct ({_acc:.0%})**

    No line gets all 4 right: XOR is not linearly separable.
    (Gradient descent gives $w_1 = {_wf[0]:.2f}$, $w_2 = {_wf[1]:.2f}$, $b = {_bf:.2f}$:
    a flat $p = 0.5$ with no line at all.)
    """
            ),
            mo.ui.plotly(_x_fig),
        ]
    )
    return


@app.cell
def _(go, mo, np):
    _xor_pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    _xor_labels = np.array([0, 1, 1, 0])

    def _step(_z):
        return (_z >= 0).astype(int)

    _s = _xor_pairs[:, 0] + _xor_pairs[:, 1]
    _h1 = _step(_s - 0.5)  # at least one input is 1
    _h2 = _step(_s - 1.5)  # both inputs are 1

    # Linear regression (least squares) on (h1, h2) -> XOR
    _A = np.column_stack([_h1, _h2, np.ones(len(_h1))])
    (_w1, _w2, _b), *_ = np.linalg.lstsq(_A, _xor_labels, rcond=None)
    _yhat = _A @ np.array([_w1, _w2, _b])
    _mse = np.mean((_yhat - _xor_labels) ** 2)

    _g = np.linspace(-0.2, 1.2, 2)
    _G1, _G2 = np.meshgrid(_g, _g)
    _Z = _w1 * _G1 + _w2 * _G2 + _b

    _lin_fig = go.Figure(
        data=[
            go.Scatter3d(
                x=_h1,
                y=_h2,
                z=_xor_labels,
                mode="markers+text",
                text=[f"({_a},{_c})" for _a, _c in _xor_pairs],
                textposition="top center",
                marker=dict(
                    size=9,
                    color=_xor_labels,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1,
                ),
                name="XOR",
                customdata=_xor_pairs,
                hovertemplate=(
                    "x=(%{customdata[0]},%{customdata[1]})"
                    "<br>h1=%{x}<br>h2=%{y}<br>XOR=%{z}<extra></extra>"
                ),
            ),
            go.Surface(
                x=_g,
                y=_g,
                z=_Z,
                colorscale=[[0, "tomato"], [1, "tomato"]],
                opacity=0.4,
                showscale=False,
                name="ŷ",
                showlegend=True,
                hovertemplate="h1=%{x:.2f}<br>h2=%{y:.2f}<br>ŷ=%{z:.2f}<extra></extra>",
            ),
        ]
    )
    _lin_fig.update_layout(
        title="Linear regression in hidden space (h1, h2)",
        scene=dict(
            xaxis=dict(title="h1", range=[-0.2, 1.2]),
            yaxis=dict(title="h2", range=[-0.2, 1.2]),
            zaxis=dict(title="XOR label"),
        ),
        height=550,
    )
    mo.vstack(
        [
            mo.md(
                f"""
    Least-squares plane: $\\hat{{y}} = {_w1:.2f}\\,h_1 + ({_w2:.2f})\\,h_2 + ({_b:.2f})$
    — MSE = {_mse:.3f}
    """
            ),
            mo.ui.plotly(_lin_fig),
        ]
    )
    return


@app.cell
def _(go, mo, np):
    _xor_pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    _xor_labels = np.array([0, 1, 1, 0])

    def _step(_z):
        return (_z >= 0).astype(int)

    _s = _xor_pairs[:, 0] + _xor_pairs[:, 1]
    _h1 = _step(_s - 0.5)  # at least one input is 1
    _h2 = _step(_s - 1.5)  # both inputs are 1

    # Logistic regression on (h1, h2) -> XOR, fit by gradient descent.
    # The data is linearly separable, so a small L2 penalty keeps w finite.
    _H = np.column_stack([_h1, _h2])
    _w = np.zeros(2)
    _b = 0.0
    _lam, _lr = 0.001, 0.5
    for _ in range(20000):
        _p = 1 / (1 + np.exp(-(_H @ _w + _b)))
        _w -= _lr * (_H.T @ (_p - _xor_labels) / len(_xor_labels) + _lam * _w)
        _b -= _lr * np.mean(_p - _xor_labels)

    # Probability surface over the (h1, h2) plane
    _g = np.linspace(-0.2, 1.2, 50)
    _G1, _G2 = np.meshgrid(_g, _g)
    _P = 1 / (1 + np.exp(-(_w[0] * _G1 + _w[1] * _G2 + _b)))

    # Decision line: w1*h1 + w2*h2 + b = 0  (where p = 0.5)
    _t = np.linspace(-2, 2, 400)
    _n = _w / np.linalg.norm(_w)
    _line = -_b / np.linalg.norm(_w) * _n + np.outer(_t, [-_n[1], _n[0]])
    _inside = np.all((_line >= -0.2) & (_line <= 1.2), axis=1)
    _line = _line[_inside]

    _h_fig = go.Figure(
        data=[
            go.Scatter3d(
                x=_h1,
                y=_h2,
                z=_xor_labels,
                mode="markers+text",
                text=[f"({_a},{_b})" for _a, _b in _xor_pairs],
                textposition="top center",
                marker=dict(
                    size=9,
                    color=_xor_labels,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1,
                    # colorbar=dict(title="XOR label"),
                ),
                name="XOR",
                customdata=_xor_pairs,
                hovertemplate=(
                    "x=(%{customdata[0]},%{customdata[1]})"
                    "<br>h1=%{x}<br>h2=%{y}<br>XOR=%{z}<extra></extra>"
                ),
            ),
            go.Surface(
                x=_g,
                y=_g,
                z=_P,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                opacity=0.6,
                showscale=False,
                name="Logistic",
                showlegend=True,
                hovertemplate="h1=%{x:.2f}<br>h2=%{y:.2f}<br>p=%{z:.3f}<extra></extra>",
            ),
            go.Scatter3d(
                x=_line[:, 0],
                y=_line[:, 1],
                z=np.full(len(_line), 0.5),
                mode="lines",
                line=dict(color="red", width=6),
                name="p = 0.5",
                hoverinfo="skip",
            ),
        ]
    )
    _h_fig.update_layout(
        title="XOR in hidden space (h1, h2)",
        scene=dict(
            xaxis=dict(title="h1", range=[-0.2, 1.2]),
            yaxis=dict(title="h2", range=[-0.2, 1.2]),
            zaxis=dict(title="XOR label", range=[-0.2, 1.2]),
        ),
        height=550,
    )
    mo.ui.plotly(_h_fig)
    return


if __name__ == "__main__":
    app.run()
