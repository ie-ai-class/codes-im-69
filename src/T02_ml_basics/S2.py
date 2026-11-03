import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="darkgrid")
    return mo, np, plt, sns


@app.cell
def _(mo):
    line_choice = mo.ui.radio(
        options=["Linear", "Logistic", "Both"],
        value="Linear",
        label="Show line",
        inline=True,
    )
    return (line_choice,)


@app.cell
def _(line_choice, mo, np, plt, sns):
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([0, 0, 1, 0, 1, 1])

    x_line = np.linspace(-2, 11, 200)

    # Linear fit: y = w*x + b
    w_lin, b_lin = 0.2, -0.2
    y_lin = w_lin * x_line + b_lin

    # Logistic regression (illustrative): y = sigmoid(w*x + b)
    w_log, b_log = 2.0, -7.0
    y_log = 1 / (1 + np.exp(-(w_log * x_line + b_log)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(x=x, y=y, s=100, color="black", zorder=3, ax=ax)
    if line_choice.value in ("Linear", "Both"):
        sns.lineplot(
            x=x_line, y=y_lin, ax=ax, label=f"linear: y = {w_lin}x + ({b_lin})"
        )
    if line_choice.value in ("Logistic", "Both"):
        sns.lineplot(
            x=x_line, y=y_log, ax=ax, label=f"logistic: y = σ({w_log}x + ({b_log}))"
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-0.3, 1.4)
    ax.set_xlim(-2, 11)
    ax.legend(loc="lower right")
    mo.vstack([line_choice, fig])
    return b_log, w_log, x, y


@app.cell
def _(mo):
    p_slider = mo.ui.slider(
        start=0.01,
        stop=0.99,
        step=0.01,
        value=0.5,
        label="Probability p",
        show_value=True,
    )
    return (p_slider,)


@app.cell
def _(mo, np, p_slider, plt):
    p = p_slider.value
    odds = p / (1 - p)
    logit = np.log(odds)

    p_grid = np.linspace(0.01, 0.99, 300)
    odds_grid = p_grid / (1 - p_grid)
    logit_grid = np.log(odds_grid)

    ODDS_MAX = 20  # fixed y-limit so the axis doesn't jump while sliding

    fig_odds, (ax_odds, ax_logit) = plt.subplots(1, 2, figsize=(10, 4))

    ax_odds.plot(p_grid, odds_grid)
    ax_odds.axhline(1, color="gray", ls="--", lw=1)
    if odds <= ODDS_MAX:
        ax_odds.plot(p, odds, "o", color="red", ms=10)
    else:  # off the chart: pin a marker to the top edge
        ax_odds.plot(p, ODDS_MAX, "^", color="red", ms=10)
    ax_odds.set_ylim(-1, ODDS_MAX)
    ax_odds.set_xlabel("p")
    ax_odds.set_ylabel("odds = p / (1 − p)")
    ax_odds.set_title("Odds: range [0, ∞)")

    ax_logit.plot(p_grid, logit_grid)
    ax_logit.axhline(0, color="gray", ls="--", lw=1)
    ax_logit.plot(p, logit, "o", color="red", ms=10)
    ax_logit.set_ylim(-5, 5)
    ax_logit.set_xlabel("p")
    ax_logit.set_ylabel("log-odds = ln(p / (1 − p))")
    ax_logit.set_title("Log-odds (logit): range (−∞, ∞)")

    fig_odds.tight_layout()

    mo.vstack(
        [
            p_slider,
            mo.md(
                f"""
    | probability $p$ | odds $\\frac{{p}}{{1-p}}$ | log-odds $\\ln\\frac{{p}}{{1-p}}$ |
    |:---:|:---:|:---:|
    | {p:.2f} | {odds:.3f} | {logit:.3f} |
    """
            ),
            fig_odds,
        ]
    )
    return


@app.cell
def _(mo):
    x_slider = mo.ui.slider(
        start=0, stop=7, step=0.1, value=3.5, label="Input x", show_value=True
    )
    return (x_slider,)


@app.cell
def _(b_log, mo, np, plt, w_log, x, x_slider, y):
    # Chain: x  ->  z = w*x + b (log-odds)  ->  odds = e^z  ->  p = odds / (1 + odds) = σ(z)
    _x0 = x_slider.value
    _z0 = w_log * _x0 + b_log
    _odds0 = np.exp(_z0)
    _p0 = 1 / (1 + np.exp(-_z0))

    _xg = np.linspace(0, 7, 300)
    _zg = w_log * _xg + b_log
    _oddsg = np.exp(_zg)
    _pg = 1 / (1 + np.exp(-_zg))

    _ODDS_MAX = 20

    _fig, (_ax_z, _ax_o, _ax_p) = plt.subplots(1, 3, figsize=(14, 4))

    # 1) Log-odds: a straight line in x
    _ax_z.plot(_xg, _zg)
    _ax_z.axhline(0, color="gray", ls="--", lw=1)
    _ax_z.plot(_x0, _z0, "o", color="red", ms=10)
    _ax_z.set_ylim(-8, 8)
    _ax_z.set_ylabel("log-odds z = wx + b")
    _ax_z.set_title("Log-odds: linear in x")

    # 2) Odds: exponential in x
    _ax_o.plot(_xg, _oddsg)
    _ax_o.axhline(1, color="gray", ls="--", lw=1)
    if _odds0 <= _ODDS_MAX:
        _ax_o.plot(_x0, _odds0, "o", color="red", ms=10)
    else:
        _ax_o.plot(_x0, _ODDS_MAX, "^", color="red", ms=10)
    _ax_o.set_ylim(-1, _ODDS_MAX)
    _ax_o.set_ylabel("odds = e^z")
    _ax_o.set_title("Odds: exponential in x")

    # 3) Probability: sigmoid in x, with the data
    _ax_p.scatter(x, y, s=60, color="black", zorder=3)
    _ax_p.plot(_xg, _pg)
    _ax_p.axhline(0.5, color="gray", ls="--", lw=1)
    _ax_p.plot(_x0, _p0, "o", color="red", ms=10)
    _ax_p.set_ylim(-0.1, 1.1)
    _ax_p.set_ylabel("p = σ(z) = odds / (1 + odds)")
    _ax_p.set_title("Probability: sigmoid in x")

    for _ax in (_ax_z, _ax_o, _ax_p):
        _ax.axvline(_x0, color="red", ls=":", lw=1)
        _ax.set_xlabel("x")
        _ax.set_xlim(0, 7)

    _fig.tight_layout()

    mo.vstack(
        [
            x_slider,
            mo.md(
                f"""
    | $x$ | log-odds $z = {w_log}x + ({b_log})$ | odds $e^z$ | probability $p = \\sigma(z)$ |
    |:---:|:---:|:---:|:---:|
    | {_x0:.1f} | {_z0:.3f} | {_odds0:.3f} | {_p0:.3f} |
    """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    w_slider = mo.ui.slider(
        start=-1, stop=4, step=0.05, value=0.5, label="w (slope)", show_value=True
    )
    b_slider = mo.ui.slider(
        start=-15, stop=5, step=0.1, value=0.0, label="b (intercept)", show_value=True
    )
    return b_slider, w_slider


@app.cell
def _(b_slider, mo, np, plt, w_slider, x, y):
    import plotly.graph_objects as go

    def _bce(w, b):
        """Mean binary cross-entropy of σ(w*x + b) on the data (broadcasts over w, b)."""
        _z = np.asarray(w)[..., None] * x + np.asarray(b)[..., None]
        _p = np.clip(1 / (1 + np.exp(-_z)), 1e-12, 1 - 1e-12)
        return -(y * np.log(_p) + (1 - y) * np.log(1 - _p)).mean(axis=-1)

    _w, _b = w_slider.value, b_slider.value
    _loss = float(_bce(_w, _b))

    # --- Logistic curve on the data ---
    _xg = np.linspace(-2, 11, 300)
    _pg = 1 / (1 + np.exp(-(_w * _xg + _b)))
    _p_data = 1 / (1 + np.exp(-(_w * x + _b)))

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.scatter(x, y, s=100, color="black", zorder=3)
    _ax.plot(_xg, _pg, label=f"y = σ({_w:.2f}x + ({_b:.2f}))")
    _ax.vlines(
        x, y, _p_data, color="red", ls=":", lw=1
    )  # gap between label and prediction
    _ax.axhline(0.5, color="gray", ls="--", lw=1)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_xlim(-2, 11)
    _ax.set_ylim(-0.3, 1.3)
    _ax.legend(loc="lower right")
    _ax.set_title(f"BCE loss = {_loss:.3f}")

    # --- BCE loss surface over (w, b) ---
    _W, _B = np.meshgrid(np.linspace(-1, 4, 101), np.linspace(-15, 5, 101))
    _L = _bce(_W, _B)

    _surf = go.Figure(
        data=[
            go.Surface(x=_W, y=_B, z=_L, colorscale="Viridis", opacity=0.9),
            go.Scatter3d(
                x=[_w],
                y=[_b],
                z=[_loss],
                mode="markers",
                marker=dict(size=6, color="red"),
                name="current (w, b)",
            ),
        ]
    )
    _surf.update_layout(
        title="Binary cross-entropy loss surface",
        scene=dict(xaxis_title="w", yaxis_title="b", zaxis_title="BCE loss"),
        width=750,
        height=600,
        uirevision="keep",  # keep the camera angle when sliders change
    )

    mo.vstack([w_slider, b_slider, _fig, _surf])
    return (go,)


@app.cell
def _(np, plt):
    # --- 2D data: two overlapping Gaussian blobs ---
    _rng = np.random.default_rng(0)
    n_2d = 30
    X0_2d = _rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n_2d, 2))  # class 0
    X1_2d = _rng.normal(loc=[4.5, 4.5], scale=1.0, size=(n_2d, 2))  # class 1
    _X = np.vstack([X0_2d, X1_2d])
    _y = np.r_[np.zeros(n_2d), np.ones(n_2d)]

    # --- Fit logistic regression with plain gradient descent on BCE ---
    _w = np.zeros(2)
    b_2d = 0.0
    for _ in range(5000):
        _p = 1 / (1 + np.exp(-(_X @ _w + b_2d)))
        _w -= 0.1 * (_X.T @ (_p - _y)) / len(_y)
        b_2d -= 0.1 * (_p - _y).mean()
    w1_2d, w2_2d = _w

    # --- Textbook view: scatter + straight decision boundary ---
    _x1 = np.linspace(-1, 7.5, 100)
    _x2 = -(w1_2d * _x1 + b_2d) / w2_2d  # w1·x1 + w2·x2 + b = 0

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.scatter(
        X0_2d[:, 0],
        X0_2d[:, 1],
        color="tab:blue",
        edgecolor="k",
        label="class 0 (y = 0)",
    )
    _ax.scatter(
        X1_2d[:, 0],
        X1_2d[:, 1],
        color="tab:red",
        marker="^",
        edgecolor="k",
        label="class 1 (y = 1)",
    )
    _ax.plot(_x1, _x2, "k-", lw=2, label="decision boundary (p = 0.5)")
    _ax.set_xlim(-1, 7.5)
    _ax.set_ylim(-1, 7.5)
    _ax.set_aspect("equal")
    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_title(f"${w1_2d:.2f}x_1 + {w2_2d:.2f}x_2 + ({b_2d:.2f}) = 0$")
    _ax.legend(loc="upper left")
    _fig
    return X0_2d, X1_2d, b_2d, n_2d, w1_2d, w2_2d


@app.cell
def _(X0_2d, X1_2d, b_2d, go, mo, n_2d, np, w1_2d, w2_2d):
    _X0, _X1, _n = X0_2d, X1_2d, n_2d
    _w1, _w2, _b = w1_2d, w2_2d, b_2d

    # --- Probability surface p = σ(w1·x1 + w2·x2 + b) ---
    _g = np.linspace(-1, 7.5, 80)
    _G1, _G2 = np.meshgrid(_g, _g)
    _P = 1 / (1 + np.exp(-(_w1 * _G1 + _w2 * _G2 + _b)))

    # Decision boundary: w1·x1 + w2·x2 + b = 0  →  x2 = −(w1·x1 + b) / w2
    _bx1 = _g
    _bx2 = -(_w1 * _bx1 + _b) / _w2
    _keep = (_bx2 >= _g[0]) & (_bx2 <= _g[-1])
    _bx1, _bx2 = _bx1[_keep], _bx2[_keep]

    _fig2d = go.Figure(
        data=[
            go.Surface(
                x=_G1,
                y=_G2,
                z=_P,
                colorscale="RdBu_r",
                cmin=0,
                cmax=1,
                opacity=0.8,
                # colorbar=dict(title="p(y=1)"),
                showscale=False
            ),
            # Data points sit at their label height: z = 0 or z = 1
            go.Scatter3d(
                x=_X0[:, 0],
                y=_X0[:, 1],
                z=np.zeros(_n),
                mode="markers",
                marker=dict(size=4, color="blue"),
                name="class 0",
            ),
            go.Scatter3d(
                x=_X1[:, 0],
                y=_X1[:, 1],
                z=np.ones(_n),
                mode="markers",
                marker=dict(size=4, color="red"),
                name="class 1",
            ),
            # Boundary on the surface (p = 0.5) and projected onto the floor (p = 0)
            go.Scatter3d(
                x=_bx1,
                y=_bx2,
                z=np.full_like(_bx1, 0.5),
                mode="lines",
                line=dict(color="black", width=6),
                name="p = 0.5 on surface",
            ),
            go.Scatter3d(
                x=_bx1,
                y=_bx2,
                z=np.zeros_like(_bx1),
                mode="lines",
                line=dict(color="black", width=6, dash="dash"),
                name="decision boundary (floor)",
            ),
        ]
    )
    _fig2d.update_layout(
        title=(f"p = σ({_w1:.2f}·x₁ + {_w2:.2f}·x₂ + ({_b:.2f}))"),
        scene=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="p(y = 1)"),
        width=800,
        height=650,
    )

    mo.vstack(
        [
            mo.md(
                f"Fitted: $w_1 = {_w1:.2f}$, $w_2 = {_w2:.2f}$, $b = {_b:.2f}$. "
                "The decision boundary is the straight line where "
                "$w_1 x_1 + w_2 x_2 + b = 0$, i.e. $p = 0.5$."
            ),
            _fig2d,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
