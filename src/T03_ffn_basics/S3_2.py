import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    return go, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Training the XOR Network on All Four Points

    The same two-layer network and starting weights as Section 3.3, but now every update uses
    **all four XOR points**. Each **epoch** runs the forward pass and backpropagation on the
    four inputs, averages the gradients, and takes one gradient-descent step:

    $$
    J = \frac{1}{4}\sum_{n=1}^{4}\bigl(y^{(n)} - \hat{y}^{(n)}\bigr)^2,
    \qquad
    w \leftarrow w - \eta\,\frac{\partial J}{\partial w}.
    $$

    Click the epoch buttons and watch the surface $\hat{y}(x_1, x_2)$ bend to fit the four points.
    Look out for the long **plateau** near $J = 0.25$, where every prediction sits close to $0.5$
    before the network suddenly finds the XOR shape.
    """)
    return


@app.cell
def _(np):
    # ── XOR data and the network from Section 3.3 ────────────────────────────
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([0, 1, 1, 0], dtype=float)

    def initial_params():
        return {
            "W1": np.array([[0.5, -0.5], [0.5, 0.5]]),  # row i feeds hidden unit i
            "b1": np.array([0.5, 0.0]),
            "W2": np.array([0.5, 0.5]),
            "b2": 0.0,
        }

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def hidden(P, Xin):
        return sigmoid(Xin @ P["W1"].T + P["b1"])

    def predict(P, Xin):
        return sigmoid(hidden(P, Xin) @ P["W2"] + P["b2"])

    def mse(P):
        return float(np.mean((Y - predict(P, X)) ** 2))

    def train_epoch(P):
        """Return the gradient of the MSE over all four points (one full batch)."""
        H = sigmoid(X @ P["W1"].T + P["b1"])  # (4, 2)
        yh = sigmoid(H @ P["W2"] + P["b2"])  # (4,)
        d2 = 2 * (yh - Y) * yh * (1 - yh) / len(Y)  # dJ/dz2 for each point, averaged
        d1 = np.outer(d2, P["W2"]) * H * (1 - H)  # dJ/dz1, (4, 2)
        return {"W2": d2 @ H, "b2": d2.sum(), "W1": d1.T @ X, "b1": d1.sum(axis=0)}

    def run_epochs(S, n, lr):
        P = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in S["params"].items()}
        hist = list(S["history"])
        trail = list(S["trail"])  # hidden-space positions (h1, h2) of the 4 points, per epoch
        for _ in range(n):
            g = train_epoch(P)
            P = {k: P[k] - lr * g[k] for k in P}
            hist.append(mse(P))
            trail.append(hidden(P, X))
        return {"params": P, "epoch": S["epoch"] + n, "history": hist, "trail": trail}

    def initial_state():
        P = initial_params()
        return {"params": P, "epoch": 0, "history": [mse(P)], "trail": [hidden(P, X)]}

    return X, Y, hidden, initial_state, predict, run_epochs


@app.cell
def _(initial_state, mo):
    get_S, set_S = mo.state(initial_state())
    return get_S, set_S


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(
        steps=[0.1, 0.2, 0.5, 1, 2, 5, 10],
        value=2,
        show_value=True,
        label="learning rate η",
    )
    return (lr_slider,)


@app.cell
def _(initial_state, lr_slider, mo, run_epochs, set_S):
    _lr = lr_slider.value

    def _btn(n):
        return mo.ui.button(label=f"+{n} epoch" + ("s" if n > 1 else ""),
                            on_click=lambda _: set_S(lambda S: run_epochs(S, n, _lr)))

    epoch_btns = [_btn(n) for n in (1, 10, 100, 1000)]
    reset_btn = mo.ui.button(label="⟲ Reset weights", kind="danger",
                             on_click=lambda _: set_S(initial_state()))
    return epoch_btns, reset_btn


@app.cell
def _(X, Y, go, np, predict):
    _g = np.linspace(-0.2, 1.2, 41)
    G1, G2 = np.meshgrid(_g, _g)
    GRID = np.column_stack([G1.ravel(), G2.ravel()])

    def surface_fig(P, epoch):
        Z = predict(P, GRID).reshape(G1.shape)
        yh = predict(P, X)
        ok = (yh >= 0.5).astype(int) == Y
        colors = ["#c0392b" if y == 0 else "#1f77b4" for y in Y]

        traces = [
            go.Surface(x=G1, y=G2, z=Z, colorscale="RdBu", cmin=0, cmax=1, opacity=0.85,
                       colorbar=dict(title="ŷ", len=0.6),
                       hovertemplate="x₁=%{x:.2f}<br>x₂=%{y:.2f}<br>ŷ=%{z:.3f}<extra></extra>",
                       name="ŷ(x₁, x₂)"),
            go.Surface(x=[-0.2, 1.2], y=[-0.2, 1.2], z=np.full((2, 2), 0.5),
                       colorscale=[[0, "gray"], [1, "gray"]], opacity=0.15, showscale=False,
                       hoverinfo="skip", name="ŷ = 0.5", showlegend=True),
            go.Scatter3d(x=X[:, 0], y=X[:, 1], z=Y, mode="markers+text",
                         marker=dict(size=8, color=colors, line=dict(color="black", width=2)),
                         text=[f"y={int(y)} {'✓' if k else '✗'}" for y, k in zip(Y, ok)],
                         textposition="top center", name="target y",
                         hovertemplate="x=(%{x:.0f}, %{y:.0f})<br>y=%{z:.0f}<extra></extra>"),
        ]

        fig = go.Figure(traces)
        fig.update_layout(
            title=f"Network output ŷ(x₁, x₂) after {epoch} epochs",
            height=560,
            margin=dict(l=0, r=0, t=50, b=0),
            uirevision="keep-camera",  # keep the user's camera angle between updates
            legend=dict(x=0, y=1),
            scene=dict(
                xaxis_title="x₁", yaxis_title="x₂", zaxis_title="ŷ",
                zaxis=dict(range=[0, 1.05]),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
            ),
        )
        return fig

    def loss_fig(history):
        ep = np.arange(len(history))
        fig = go.Figure(go.Scatter(x=ep, y=history, mode="lines", line=dict(color="#3a6ea5", width=2),
                                   name="J", hovertemplate="epoch %{x}<br>J=%{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[ep[-1]], y=[history[-1]], mode="markers",
                                 marker=dict(size=10, color="#c0392b"), name="now",
                                 hovertemplate="epoch %{x}<br>J=%{y:.4f}<extra></extra>"))
        fig.update_layout(
            title=f"Loss (MSE)  ·  J = {history[-1]:.4f}",
            xaxis_title="epoch", yaxis_title="J", yaxis_rangemode="tozero",
            height=300, margin=dict(l=10, r=10, t=45, b=10), showlegend=False,
        )
        return fig

    def hidden_fig(P, trail, epoch):
        """XOR in hidden space: the output neuron is a logistic regression on (h1, h2)."""
        H = trail[-1]
        yh = predict(P, X)
        ok = (yh >= 0.5).astype(int) == Y
        colors = ["#c0392b" if y == 0 else "#1f77b4" for y in Y]
        labels = [f"({int(a)},{int(b)})" for a, b in X]

        # Output-neuron surface over the hidden square (sigmoid outputs lie in (0, 1))
        g = np.linspace(-0.05, 1.05, 41)
        H1, H2 = np.meshgrid(g, g)
        w, b = P["W2"], P["b2"]
        Z = 1 / (1 + np.exp(-(w[0] * H1 + w[1] * H2 + b)))

        traces = [
            go.Surface(x=H1, y=H2, z=Z, colorscale="RdBu", cmin=0, cmax=1, opacity=0.6, showscale=False,
                       name="output neuron σ(w⁽²⁾·h + b⁽²⁾)", showlegend=True,
                       hovertemplate="h₁=%{x:.2f}<br>h₂=%{y:.2f}<br>ŷ=%{z:.3f}<extra></extra>"),
        ]
        # Trail of each point through hidden space (on the floor, z = 0), thinned to ≤ 300 points
        T = np.array(trail)  # (epochs+1, 4, 2)
        idx = np.unique(np.linspace(0, len(T) - 1, min(len(T), 300)).astype(int))
        for n in range(4):
            traces.append(go.Scatter3d(x=T[idx, n, 0], y=T[idx, n, 1], z=np.zeros(len(idx)), mode="lines",
                                       line=dict(color=colors[n], width=3), opacity=0.5,
                                       name="path since epoch 0" if n == 0 else None, showlegend=n == 0,
                                       legendgroup="trail", hoverinfo="skip"))
        traces.append(go.Scatter3d(
            x=H[:, 0], y=H[:, 1], z=Y, mode="markers+text",
            marker=dict(size=8, color=colors, line=dict(color="black", width=2)),
            text=[f"{lab} {'✓' if k else '✗'}" for lab, k in zip(labels, ok)], textposition="top center",
            name="XOR points at (h₁, h₂, y)", customdata=X,
            hovertemplate="x=(%{customdata[0]:.0f},%{customdata[1]:.0f})<br>h₁=%{x:.3f}<br>h₂=%{y:.3f}"
                          "<br>y=%{z:.0f}<extra></extra>"))
        traces.append(go.Scatter3d(x=H[:, 0], y=H[:, 1], z=np.zeros(4), mode="markers",
                                   marker=dict(size=4, color=colors), showlegend=False, hoverinfo="skip"))

        # Decision line w1*h1 + w2*h2 + b = 0 (where ŷ = 0.5)
        norm = np.linalg.norm(w)
        if norm > 1e-6:
            t = np.linspace(-3, 3, 600)
            nv = w / norm
            line = -b / norm * nv + np.outer(t, [-nv[1], nv[0]])
            line = line[np.all((line >= -0.05) & (line <= 1.05), axis=1)]
            if len(line):
                traces.append(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=np.full(len(line), 0.5), mode="lines",
                                           line=dict(color="black", width=6), name="ŷ = 0.5 boundary",
                                           hoverinfo="skip"))

        fig = go.Figure(traces)
        fig.update_layout(
            title=f"XOR in hidden space (h₁, h₂) after {epoch} epochs",
            height=560,
            margin=dict(l=0, r=0, t=50, b=0),
            uirevision="keep-camera-hidden",
            legend=dict(x=0, y=1),
            scene=dict(
                xaxis=dict(title="h₁", range=[-0.05, 1.05]),
                yaxis=dict(title="h₂", range=[-0.05, 1.05]),
                zaxis=dict(title="ŷ", range=[0, 1.05]),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
            ),
        )
        return fig

    return hidden_fig, loss_fig, surface_fig


@app.cell
def _(X, Y, predict):
    def prediction_table(P):
        yh = predict(P, X)
        rows = "\n".join(
            f"| {int(a)} | {int(b)} | {int(t)} | {p:.3f} | {(t - p) ** 2:.4f} | {'✓' if int(p >= 0.5) == t else '✗'} |"
            for (a, b), t, p in zip(X, Y, yh)
        )
        return (
            "| $x_1$ | $x_2$ | $y$ | $\\hat{y}$ | $(y-\\hat{y})^2$ | correct? |\n"
            "|---|---|---|---:|---:|:---:|\n" + rows
        )

    def param_table(P):
        rows = [
            ("$w^{(1)}_{11}$", P["W1"][0, 0]), ("$w^{(1)}_{12}$", P["W1"][0, 1]), ("$b^{(1)}_1$", P["b1"][0]),
            ("$w^{(1)}_{21}$", P["W1"][1, 0]), ("$w^{(1)}_{22}$", P["W1"][1, 1]), ("$b^{(1)}_2$", P["b1"][1]),
            ("$w^{(2)}_1$", P["W2"][0]), ("$w^{(2)}_2$", P["W2"][1]), ("$b^{(2)}$", P["b2"]),
        ]
        # Three columns: hidden unit 1 | hidden unit 2 | output
        md = "| Hidden unit 1 | | Hidden unit 2 | | Output | |\n|---|---:|---|---:|---|---:|\n"
        for i in range(3):
            cells = [rows[i], rows[3 + i], rows[6 + i]]
            md += "| " + " | ".join(f"{nm} | {v:.3f}" for nm, v in cells) + " |\n"
        return md

    return param_table, prediction_table


@app.cell
def _(
    epoch_btns,
    get_S,
    hidden_fig,
    loss_fig,
    lr_slider,
    mo,
    param_table,
    prediction_table,
    reset_btn,
    surface_fig,
):
    _S = get_S()
    _P = _S["params"]

    mo.vstack(
        [
            mo.hstack(
                [lr_slider, *epoch_btns, reset_btn, mo.md(f"**Epoch {_S['epoch']}**")],
                justify="start",
                align="center",
                gap=1,
            ),
            mo.hstack(
                [
                    mo.ui.plotly(surface_fig(_P, _S["epoch"])),
                    mo.ui.plotly(hidden_fig(_P, _S["trail"], _S["epoch"])),
                ],
                widths=[1, 1],
                align="start",
                gap=1,
            ),
            mo.md(
                "**Right plot:** the hidden layer moves each input $(x_1, x_2)$ to a new point "
                "$(h_1, h_2)$ (paths on the floor). The output neuron is just logistic regression on "
                "$(h_1, h_2)$, so its boundary $\\hat{y} = 0.5$ is a **straight line**. At the start, "
                "no straight line can separate the blue points ($y=1$) from the red ($y=0$). "
                "As training proceeds, the hidden layer moves them until one can."
            ),
            mo.hstack(
                [
                    mo.ui.plotly(loss_fig(_S["history"])),
                    mo.md("**Predictions on the four XOR points**\n\n" + prediction_table(_P)),
                    mo.md("**Current weights and biases**\n\n" + param_table(_P)),
                ],
                widths=[2, 2, 2],
                align="start",
                gap=2,
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
