import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Forward Pass and Backpropagation, Step by Step

    The two-layer network from Section 3.3, trained on the single XOR input
    $\mathbf{x} = (x_1, x_2) = (0, 1)$ with target $y = 1$ and loss $J = (y - \hat{y})^2$.

    Each **pass** has 12 steps: the **forward pass** (steps 1–5), **backpropagation**
    (steps 6–10), the **update** $w \leftarrow w - \eta\,\frac{\partial J}{\partial w}$ (step 11),
    and a **check** that the loss went down (step 12).
    Use **Next ▶** to go one step at a time, or **⏭ Run full pass** to jump to the end of the pass
    (press it repeatedly to train pass after pass).
    In the diagram, the **black** number inside a node is its forward value and the
    **red** number is the gradient $\partial J / \partial(\text{node})$ flowing backwards.
    """)
    return


@app.cell
def _(np):
    # ── Data and initial parameters (Section 3.3) ────────────────────────────
    X_IN = np.array([0.0, 1.0])  # (x1, x2)
    Y_T = 1.0  # target y
    LAST = 12  # number of steps in one pass

    def initial_params():
        return {
            "W1": np.array([[0.5, -0.5], [0.5, 0.5]]),  # row i feeds hidden unit i
            "b1": np.array([0.5, 0.0]),
            "W2": np.array([0.5, 0.5]),
            "b2": 0.0,
        }

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward(P):
        z1 = P["W1"] @ X_IN + P["b1"]
        h = sigmoid(z1)
        z2 = P["W2"] @ h + P["b2"]
        yh = sigmoid(z2)
        return {"z1": z1, "h": h, "z2": z2, "yh": yh, "J": (Y_T - yh) ** 2}

    def backward(P, f):
        dJ_dyh = 2 * (f["yh"] - Y_T)
        s2 = f["yh"] * (1 - f["yh"])  # sigma'(z2)
        dJ_dz2 = dJ_dyh * s2
        dJ_dh = dJ_dz2 * P["W2"]
        s1 = f["h"] * (1 - f["h"])  # sigma'(z1)
        dJ_dz1 = dJ_dh * s1
        return {
            "dJ_dyh": dJ_dyh,
            "s2": s2,
            "dJ_dz2": dJ_dz2,
            "W2": dJ_dz2 * f["h"],
            "b2": dJ_dz2,
            "dJ_dh": dJ_dh,
            "s1": s1,
            "dJ_dz1": dJ_dz1,
            "W1": np.outer(dJ_dz1, X_IN) + 0.0,  # + 0.0 turns -0.0 into 0.0
            "b1": dJ_dz1,
        }

    def update(P, g, lr):
        return {k: P[k] - lr * g[k] for k in ("W1", "b1", "W2", "b2")}

    def initial_state():
        P = initial_params()
        return {"params": P, "step": 0, "history": [float(forward(P)["J"])], "passes": 0}

    return (
        LAST,
        X_IN,
        Y_T,
        backward,
        forward,
        initial_state,
        update,
    )


@app.cell
def _(initial_state, mo):
    get_S, set_S = mo.state(initial_state())
    return get_S, set_S


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(
        steps=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        value=0.1,
        show_value=True,
        label="learning rate η",
    )
    return (lr_slider,)


@app.cell
def _(LAST, backward, forward, initial_state, lr_slider, mo, set_S, update):
    _lr = lr_slider.value

    def _commit(S):
        """Apply the gradient-descent update and start a new pass."""
        P = S["params"]
        newP = update(P, backward(P, forward(P)), _lr)
        return {
            "params": newP,
            "step": 1,
            "history": S["history"] + [float(forward(newP)["J"])],
            "passes": S["passes"] + 1,
        }

    def _advance(S):
        if S["step"] < LAST:
            return {**S, "step": S["step"] + 1}
        return _commit(S)

    def _full_pass(S):
        # Mid-pass: finish it. At the end of a pass: commit, then run the next one.
        if S["step"] == LAST:
            S = _commit(S)
        return {**S, "step": LAST}

    prev_btn = mo.ui.button(label="◀ Prev", on_click=lambda _: set_S(lambda S: {**S, "step": max(0, S["step"] - 1)}))
    next_btn = mo.ui.button(label="Next ▶", on_click=lambda _: set_S(_advance))
    pass_btn = mo.ui.button(label="⏭ Run full pass", kind="success", on_click=lambda _: set_S(_full_pass))
    reset_btn = mo.ui.button(label="⟲ Reset weights", kind="danger", on_click=lambda _: set_S(initial_state()))
    return next_btn, pass_btn, prev_btn, reset_btn


@app.cell
def _(LAST, X_IN, Y_T, np, plt):
    # ── Network diagram ──────────────────────────────────────────────────────
    C_POS, C_NEG = "#3a6ea5", "#7d3c98"  # weight sign colours
    C_FWD, C_BWD, C_NEW = "#e67e22", "#c0392b", "#1e8449"
    POS = {
        "x1": (0, 1.8), "x2": (0, -1.8),
        "z11": (3.6, 1.8), "h1": (5.6, 1.8),
        "z12": (3.6, -1.8), "h2": (5.6, -1.8),
        "z2": (9.2, 0), "yh": (11.2, 0), "J": (13.6, 0), "y": (13.6, -2.4),
    }
    R = 0.66
    W1_NAME = {(0, 0): "w^{(1)}_{11}", (0, 1): "w^{(1)}_{12}", (1, 0): "w^{(1)}_{21}", (1, 1): "w^{(1)}_{22}"}
    W2_NAME = {0: "w^{(2)}_1", 1: "w^{(2)}_2"}
    B1_NAME = {0: "b^{(1)}_1", 1: "b^{(1)}_2"}
    B2_NAME = "b^{(2)}"

    def _unit(a, b):
        d = np.subtract(b, a)
        return d / np.linalg.norm(d)

    def _edge(ax, a, b, color, lw, ls="-", alpha=1.0):
        u = _unit(POS[a], POS[b])
        s = np.add(POS[a], R * u)
        e = np.subtract(POS[b], R * u)
        ax.annotate("", xy=e, xytext=s, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                        alpha=alpha, mutation_scale=14))

    def _back_arrow(ax, a, b, label, rad, strong):
        """Curved dashed red arrow from node a back to node b, as in Figure 3."""
        A, B = np.array(POS[a], float), np.array(POS[b], float)
        d = B - A
        n = np.array([d[1], -d[0]]) / np.linalg.norm(d) * np.sign(rad)
        u = d / np.linalg.norm(d)
        c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
        start = A + R * (c * u + s * n)
        end = B + R * (-c * u + s * n)
        col = C_BWD if strong else "#e6a19a"
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(
            arrowstyle="-|>", color=col, lw=2.4 if strong else 1.5, ls="--",
            connectionstyle=f"arc3,rad={rad}", mutation_scale=14))
        mid = (start + end) / 2
        dd = end - start
        apex = mid + 0.5 * rad * np.array([dd[1], -dd[0]])
        lp = apex + 0.28 * n
        ax.text(*lp, label, color=col, fontsize=10, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    def _node(ax, key, sym, fill, val=None, grad=None, hl=None, val_color="black"):
        x, y = POS[key]
        ax.add_patch(plt.Circle((x, y), R, fc=fill, ec=hl or "#2c3e50", lw=3.2 if hl else 1.5, zorder=3))
        ax.text(x, y + (0.24 if val is not None else 0), sym, ha="center", va="center", fontsize=14, zorder=4)
        if val is not None:
            ax.text(x, y - 0.08, f"{val:.3f}", ha="center", va="center", fontsize=10,
                    color=val_color, weight="bold", zorder=4)
        if grad is not None:
            ax.text(x, y - 0.35, f"{grad:+.3f}", ha="center", va="center", fontsize=9.5,
                    color=C_BWD, weight="bold", zorder=4)

    def _label(ax, xy, text, color="black", ha="center", size=10.5):
        ax.text(*xy, text, ha=ha, va="center", fontsize=size, color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.92))

    def draw_network(P, f, g, newP, fn, k, title, title_color):
        fig, ax = plt.subplots(figsize=(16, 7.4))
        ax.set_xlim(-1.0, 14.8)
        ax.set_ylim(-3.5, 3.9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=16, color=title_color, loc="left", weight="bold")

        checking = k == LAST  # step 12: everything shown with the NEW parameters
        Pv, fv = (newP, fn) if checking else (P, f)
        vcol = C_NEW if checking else "black"

        # Layer labels and neuron boxes
        for xl, txt in [(0, "Input layer"), (4.6, "Hidden layer"), (10.2, "Output layer"), (13.6, "Loss")]:
            ax.text(xl, 3.65, txt, ha="center", fontsize=12, style="italic", color="gray")
        for (x0, y0, x1, y1) in [(2.8, 1.0, 6.4, 2.6), (2.8, -2.6, 6.4, -1.0), (8.4, -0.8, 12.0, 0.8)]:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", ec="#bbbbbb",
                                       lw=1.2, zorder=1))

        # ── Active highlights for this step ──
        fwd_edges = {1: ["x1z11", "x1z12", "x2z11", "x2z12"], 2: ["s1", "s2"], 3: ["h1z2", "h2z2"],
                     4: ["so"], 5: ["yhJ", "yJ"]}.get(k, [])
        grad_edges = {8: ["h1z2", "h2z2"], 10: ["x1z11", "x1z12", "x2z11", "x2z12"]}.get(k, [])
        hl_nodes = {1: ["z11", "z12"], 2: ["h1", "h2"], 3: ["z2"], 4: ["yh"], 5: ["J"]}.get(k, [])
        bhl_nodes = {6: ["yh"], 7: ["z2"], 8: ["z2"], 9: ["h1", "h2", "z11", "z12"], 10: ["z11", "z12"]}.get(k, [])
        if checking:
            hl_nodes = ["J"]

        def edge_style(key, w):
            if key in grad_edges:
                return C_BWD, 3.0
            if key in fwd_edges or (checking and key not in ("s1", "s2", "so", "yhJ", "yJ")):
                return (C_FWD if not checking else C_NEW), 3.0
            if w is None:
                return "#7f8c8d", 1.6
            return (C_POS if w >= 0 else C_NEG), 0.8 + 2.2 * min(abs(w), 1.5)

        # ── Input → hidden weights ──
        w1_edges = {"x1z11": ("x1", "z11", (0, 0), (1.8, 1.8), (0, -0.33)),
                    "x2z11": ("x2", "z11", (0, 1), (1.08, -0.72), (0, 0.33)),
                    "x1z12": ("x1", "z12", (1, 0), (1.08, 0.72), (0, -0.33)),
                    "x2z12": ("x2", "z12", (1, 1), (1.8, -1.8), (0, 0.33))}
        for key, (a, b, ij, lp, goff) in w1_edges.items():
            w = Pv["W1"][ij]
            col, lw = edge_style(key, w)
            _edge(ax, a, b, col, lw)
            name = W1_NAME[ij]
            if k == 11:
                _label(ax, lp, f"${name}$: {P['W1'][ij]:.3f} → {newP['W1'][ij]:.3f}", C_NEW)
            else:
                _label(ax, lp, f"${name}={w:.3f}$", vcol if checking else "black")
            if k == 10:
                _label(ax, np.add(lp, goff), f"$\\partial J/\\partial {name}={g['W1'][ij]:+.3f}$", C_BWD, size=9.5)

        # ── Hidden → output weights ──
        w2_edges = {"h1z2": ("h1", "z2", 0, (7.22, 0.99), (0, -0.33)),
                    "h2z2": ("h2", "z2", 1, (7.22, -0.99), (0, 0.33))}
        for key, (a, b, i, lp, goff) in w2_edges.items():
            w = Pv["W2"][i]
            col, lw = edge_style(key, w)
            _edge(ax, a, b, col, lw)
            name = W2_NAME[i]
            if k == 11:
                _label(ax, lp, f"${name}$: {P['W2'][i]:.3f} → {newP['W2'][i]:.3f}", C_NEW)
            else:
                _label(ax, lp, f"${name}={w:.3f}$", vcol if checking else "black")
            if 8 <= k <= 10:
                _label(ax, np.add(lp, goff), f"$\\partial J/\\partial {name}={g['W2'][i]:+.3f}$", C_BWD, size=9.5)

        # ── Sigmoid arrows and loss arrows ──
        for key, a, b in [("s1", "z11", "h1"), ("s2", "z12", "h2"), ("so", "z2", "yh")]:
            col, lw = edge_style(key, None)
            _edge(ax, a, b, col if key in fwd_edges else C_POS, lw if key in fwd_edges else 1.8)
            ax.text((POS[a][0] + POS[b][0]) / 2, POS[a][1] + 0.2, r"$\sigma$", ha="center", fontsize=13,
                    color=C_FWD if key in fwd_edges else C_POS)
        for key, a in [("yhJ", "yh"), ("yJ", "y")]:
            col = C_FWD if key in fwd_edges else "#a93226"
            _edge(ax, a, "J", col, 3.0 if key in fwd_edges else 1.8)
        ax.text(POS["J"][0], POS["J"][1] + 1.0, r"$J=(y-\hat{y})^2$", ha="center", fontsize=11, color="#a93226")

        # ── Biases ──
        def bias_text(name, old, new, gval, active_step, show_grad, xy, gxy, ha):
            if k == 11:
                _label(ax, xy, f"$+\\,{name}$: {old:.3f} → {new:.3f}", C_NEW, ha=ha)
            else:
                col = C_FWD if k == active_step else vcol
                _label(ax, xy, f"$+\\,{name}={new if checking else old:.3f}$", col, ha=ha)
            if show_grad:
                _label(ax, gxy, f"$\\partial J/\\partial {name}={gval:+.3f}$", C_BWD, ha=ha, size=9.5)

        bias_text(B1_NAME[0], P["b1"][0], newP["b1"][0], g["b1"][0], 1, k == 10, (3.0, 2.95), (3.0, 3.3), "right")
        bias_text(B1_NAME[1], P["b1"][1], newP["b1"][1], g["b1"][1], 1, k == 10, (3.0, -2.95), (3.0, -3.3), "right")
        bias_text(B2_NAME, P["b2"], newP["b2"], g["b2"], 3, 8 <= k <= 10, (9.2, -1.1), (9.2, -1.45), "center")

        # ── Backward arrows (local derivatives, as in Figure 3) ──
        if 6 <= k <= 10:
            _back_arrow(ax, "J", "yh", f"$\\partial J/\\partial\\hat{{y}}={g['dJ_dyh']:+.3f}$", 0.45, k == 6)
        if 7 <= k <= 10:
            _back_arrow(ax, "yh", "z2", f"$\\sigma'(z^{{(2)}})={g['s2']:.3f}$", 0.45, k == 7)
        if 9 <= k <= 10:
            _back_arrow(ax, "z2", "h1", f"$w^{{(2)}}_1={P['W2'][0]:.3f}$", 0.3, k == 9)
            _back_arrow(ax, "z2", "h2", f"$w^{{(2)}}_2={P['W2'][1]:.3f}$", -0.3, k == 9)
            _back_arrow(ax, "h1", "z11", f"$\\sigma'(z^{{(1)}}_1)={g['s1'][0]:.3f}$", 0.5, k == 9)
            _back_arrow(ax, "h2", "z12", f"$\\sigma'(z^{{(1)}}_2)={g['s1'][1]:.3f}$", -0.5, k == 9)

        # ── Nodes ──
        show = lambda s: checking or (k >= s)  # forward value known?
        bshow = lambda s: (not checking) and k >= s and k <= 11  # gradient known?
        fill_in, fill_pre, fill_act, fill_loss = "#e5e7e9", "white", "#dce9f7", "#fdecea"
        def hl(n):
            if n in hl_nodes:
                return C_NEW if checking else C_FWD
            return C_BWD if n in bhl_nodes else None
        _node(ax, "x1", "$x_1$", fill_in, X_IN[0])
        _node(ax, "x2", "$x_2$", fill_in, X_IN[1])
        _node(ax, "y", "$y$", fill_in, Y_T)
        ax.text(POS["y"][0] - 0.85, POS["y"][1], "target", ha="right", va="center", fontsize=10, color="gray")
        for i, (zk, hk) in enumerate([("z11", "h1"), ("z12", "h2")]):
            _node(ax, zk, f"$z^{{(1)}}_{i + 1}$", fill_pre, fv["z1"][i] if show(1) else None,
                  g["dJ_dz1"][i] if bshow(9) else None, hl(zk), vcol)
            _node(ax, hk, f"$h_{i + 1}$", fill_act, fv["h"][i] if show(2) else None,
                  g["dJ_dh"][i] if bshow(9) else None, hl(hk), vcol)
        _node(ax, "z2", "$z^{(2)}$", fill_pre, fv["z2"] if show(3) else None,
              g["dJ_dz2"] if bshow(7) else None, hl("z2"), vcol)
        _node(ax, "yh", r"$\hat{y}$", fill_act, fv["yh"] if show(4) else None,
              g["dJ_dyh"] if bshow(6) else None, hl("yh"), vcol)
        _node(ax, "J", "$J$", fill_loss, fv["J"] if show(5) else None, None, hl("J"), vcol)

        fig.tight_layout()
        return fig

    return (draw_network,)


@app.cell
def _(LAST, Y_T):
    # ── Step descriptions (LaTeX with live numbers) ──────────────────────────
    STEP_NAMES = [
        ("Ready", "start"),
        ("Hidden pre-activation", "fwd"), ("Hidden activation", "fwd"), ("Output pre-activation", "fwd"),
        ("Prediction ŷ", "fwd"), ("Loss J", "fwd"),
        ("∂J/∂ŷ", "bwd"), ("∂J/∂z⁽²⁾", "bwd"), ("Output-layer gradients", "bwd"),
        ("∂J/∂h, ∂J/∂z⁽¹⁾", "bwd"), ("Hidden-layer gradients", "bwd"),
        ("Update", "upd"), ("Did the loss go down?", "chk"),
    ]
    PHASE_COLOR = {"start": "#7f8c8d", "fwd": "#e67e22", "bwd": "#c0392b", "upd": "#1e8449", "chk": "#1e8449"}
    PHASE_NAME = {"start": "", "fwd": "Forward pass", "bwd": "Backpropagation", "upd": "Gradient descent",
                  "chk": "Forward pass with new parameters"}

    def n(v):
        return f"{v:.3f}"

    def p(v):
        return f"({v:.3f})"

    def pd(a, b):
        return "\\frac{\\partial " + a + "}{\\partial " + b + "}"

    def box(v):
        return "\\boxed{" + n(v) + "}"

    yname = "\\hat{y}"

    def step_text(k, P, f, g, newP, fn, lr):
        x1, x2 = 0, 1
        W1, b1, W2, b2 = P["W1"], P["b1"], P["W2"], P["b2"]
        if k == 0:
            return (
                "Input $\\mathbf{x} = (x_1, x_2) = (0, 1)$, target $y = 1$.\n\n"
                "Nothing has been computed yet. Press **Next ▶** to start the forward pass, or **⏭ Run full pass** to do the whole pass at once."
            )
        if k == 1:
            return (
                "**Step 1 — Hidden layer (pre-activation).** Weighted sums entering each hidden neuron:\n\n"
                "$$\\begin{aligned}"
                f"z^{{(1)}}_1 &= w^{{(1)}}_{{11}}x_1 + w^{{(1)}}_{{12}}x_2 + b^{{(1)}}_1 = {p(W1[0,0])}({x1}) + {p(W1[0,1])}({x2}) + {p(b1[0])} = {box(f['z1'][0])}\\\\"
                f"z^{{(1)}}_2 &= w^{{(1)}}_{{21}}x_1 + w^{{(1)}}_{{22}}x_2 + b^{{(1)}}_2 = {p(W1[1,0])}({x1}) + {p(W1[1,1])}({x2}) + {p(b1[1])} = {box(f['z1'][1])}"
                "\\end{aligned}$$"
            )
        if k == 2:
            return (
                "**Step 1 — Hidden layer (activation).** Apply the sigmoid $\\sigma(z) = \\dfrac{1}{1+e^{-z}}$:\n\n"
                "$$\\begin{aligned}"
                f"h_1 &= \\sigma(z^{{(1)}}_1) = \\sigma({n(f['z1'][0])}) = {box(f['h'][0])}\\\\"
                f"h_2 &= \\sigma(z^{{(1)}}_2) = \\sigma({n(f['z1'][1])}) = {box(f['h'][1])}"
                "\\end{aligned}$$"
            )
        if k == 3:
            return (
                "**Step 2 — Output layer (pre-activation).**\n\n"
                f"$$z^{{(2)}} = w^{{(2)}}_1 h_1 + w^{{(2)}}_2 h_2 + b^{{(2)}} = {p(W2[0])}{p(f['h'][0])} + {p(W2[1])}{p(f['h'][1])} + {p(b2)} = {box(f['z2'])}$$"
            )
        if k == 4:
            return (
                "**Step 2 — Output layer (activation).**\n\n"
                f"$${yname} = \\sigma(z^{{(2)}}) = \\sigma({n(f['z2'])}) = {box(f['yh'])}$$"
            )
        if k == 5:
            return (
                "**Step 3 — Loss.** Using the squared error:\n\n"
                f"$$J = (y - {yname})^2 = ({Y_T:g} - {n(f['yh'])})^2 = {box(f['J'])}$$\n\n"
                f"The network outputs ${n(f['yh'])}$ when it should output $1$. "
                "Now we work **backwards** to find how each weight affects $J$."
            )
        if k == 6:
            return (
                "**Backprop 1 — Start at the loss.** The error signal is *prediction minus target*:\n\n"
                f"$${pd('J', yname)} = 2({yname} - y) = 2({n(f['yh'])} - 1) = {box(g['dJ_dyh'])}$$\n\n"
                "It is negative: increasing $\\hat{y}$ would *decrease* $J$."
            )
        if k == 7:
            return (
                "**Backprop 2 — Through the output sigmoid.** Multiply by the local slope "
                "$\\sigma'(z) = \\sigma(z)\\bigl(1-\\sigma(z)\\bigr)$:\n\n"
                "$$\\begin{aligned}"
                f"{pd(yname, 'z^{(2)}')} &= \\sigma'(z^{{(2)}}) = {yname}(1-{yname}) = {n(f['yh'])}(1-{n(f['yh'])}) = {n(g['s2'])}\\\\"
                f"{pd('J', 'z^{(2)}')} &= {pd('J', yname)}\\cdot{pd(yname, 'z^{(2)}')} = {p(g['dJ_dyh'])}{p(g['s2'])} = {box(g['dJ_dz2'])}"
                "\\end{aligned}$$"
            )
        if k == 8:
            return (
                "**Backprop 3 — Output-layer gradients.** "
                "(error signal at the neuron) × (input to that weight):\n\n"
                "$$\\begin{aligned}"
                f"{pd('J', 'w^{(2)}_1')} &= {pd('J', 'z^{(2)}')}\\cdot h_1 = {p(g['dJ_dz2'])}{p(f['h'][0])} = {box(g['W2'][0])}\\\\"
                f"{pd('J', 'w^{(2)}_2')} &= {pd('J', 'z^{(2)}')}\\cdot h_2 = {p(g['dJ_dz2'])}{p(f['h'][1])} = {box(g['W2'][1])}\\\\"
                f"{pd('J', 'b^{(2)}')} &= {pd('J', 'z^{(2)}')}\\cdot 1 = {box(g['b2'])}"
                "\\end{aligned}$$"
            )
        if k == 9:
            return (
                "**Backprop 4 — Back into the hidden layer.** Moving back one layer multiplies in that "
                "layer's weight and one activation slope:\n\n"
                "$$\\begin{aligned}"
                f"{pd('J', 'h_1')} &= {pd('J', 'z^{(2)}')}\\cdot w^{{(2)}}_1 = {p(g['dJ_dz2'])}{p(W2[0])} = {n(g['dJ_dh'][0])}"
                f"&\\quad {pd('J', 'z^{(1)}_1')} &= {pd('J', 'h_1')}\\cdot\\sigma'(z^{{(1)}}_1) = {p(g['dJ_dh'][0])}{p(g['s1'][0])} = {box(g['dJ_dz1'][0])}\\\\"
                f"{pd('J', 'h_2')} &= {pd('J', 'z^{(2)}')}\\cdot w^{{(2)}}_2 = {p(g['dJ_dz2'])}{p(W2[1])} = {n(g['dJ_dh'][1])}"
                f"&\\quad {pd('J', 'z^{(1)}_2')} &= {pd('J', 'h_2')}\\cdot\\sigma'(z^{{(1)}}_2) = {p(g['dJ_dh'][1])}{p(g['s1'][1])} = {box(g['dJ_dz1'][1])}"
                "\\end{aligned}$$\n\n"
                "Since $\\sigma'(z) \\le 0.25$, the signal shrinks as it travels back."
            )
        if k == 10:
            rows = []
            for i in range(2):
                for j in range(2):
                    rows.append(
                        f"{pd('J', f'w^{{(1)}}_{{{i + 1}{j + 1}}}')} &= {pd('J', f'z^{{(1)}}_{i + 1}')}\\cdot x_{j + 1} "
                        f"= {p(g['dJ_dz1'][i])}({[x1, x2][j]}) = {box(g['W1'][i, j])}"
                    )
                rows.append(f"{pd('J', f'b^{{(1)}}_{i + 1}')} &= {pd('J', f'z^{{(1)}}_{i + 1}')}\\cdot 1 = {box(g['b1'][i])}")
            return (
                "**Backprop 5 — Hidden-layer gradients.** Again (error signal) × (input to that weight):\n\n"
                "$$\\begin{aligned}" + "\\\\".join(rows) + "\\end{aligned}$$\n\n"
                "$w^{(1)}_{11}$ and $w^{(1)}_{21}$ get **zero** gradient because their input is $x_1 = 0$."
            )
        if k == 11:
            items = [("w^{(2)}_1", W2[0], g["W2"][0], newP["W2"][0]),
                     ("w^{(2)}_2", W2[1], g["W2"][1], newP["W2"][1]),
                     ("b^{(2)}", b2, g["b2"], newP["b2"])]
            for i in range(2):
                for j in range(2):
                    items.append((f"w^{{(1)}}_{{{i + 1}{j + 1}}}", W1[i, j], g["W1"][i, j], newP["W1"][i, j]))
                items.append((f"b^{{(1)}}_{i + 1}", b1[i], g["b1"][i], newP["b1"][i]))
            rows = [f"{nm} &\\leftarrow {n(o)} - ({lr:g}){p(gr)} = {box(nw)}" for nm, o, gr, nw in items]
            return (
                f"**Update — gradient descent** with $\\eta = {lr:g}$: "
                "$\\;w \\leftarrow w - \\eta\\,\\dfrac{\\partial J}{\\partial w}$\n\n"
                "$$\\begin{aligned}" + "\\\\".join(rows) + "\\end{aligned}$$"
            )
        # k == LAST
        dJ = fn["J"] - f["J"]
        verdict = (
            "✅ The loss went **down**: the prediction moved toward the target."
            if dJ < 0 else
            "⚠️ The loss went **up**: the learning rate is too large and the step overshot."
        )
        return (
            "**Check — forward pass again with the new parameters.**\n\n"
            "$$\\begin{aligned}"
            f"h_1 &= \\sigma({n(fn['z1'][0])}) = {n(fn['h'][0])}, \\quad h_2 = \\sigma({n(fn['z1'][1])}) = {n(fn['h'][1])}\\\\"
            f"{yname} &= \\sigma({n(fn['z2'])}) = {n(fn['yh'])}\\\\"
            f"J &= (1 - {n(fn['yh'])})^2 = {box(fn['J'])}"
            "\\end{aligned}$$\n\n"
            f"| Quantity | Before update | After update |\n|---|---|---|\n"
            f"| Prediction $\\hat{{y}}$ | {n(f['yh'])} | **{n(fn['yh'])}** |\n"
            f"| Loss $J$ | {n(f['J'])} | **{n(fn['J'])}** |\n\n"
            f"{verdict} Press **Next ▶** to start the next pass with these parameters."
        )

    def progress_html(k):
        pills = []
        for i, (name, ph) in enumerate(STEP_NAMES[1:], start=1):
            col = PHASE_COLOR[ph]
            if i == k:
                style = f"background:{col};color:white;border:1px solid {col};font-weight:600"
            elif i < k:
                style = f"background:white;color:{col};border:1px solid {col}"
            else:
                style = "background:white;color:#aaa;border:1px solid #ddd"
            pills.append(f"<span style='{style};padding:2px 8px;border-radius:10px;margin:2px;"
                         f"font-size:0.85em;display:inline-block'>{i}. {name}</span>")
        return "<div style='line-height:1.9'>" + "".join(pills) + "</div>"

    assert len(STEP_NAMES) == LAST + 1
    return PHASE_COLOR, PHASE_NAME, STEP_NAMES, n, progress_html, step_text


@app.cell
def _(LAST, n, plt):
    def param_table(P, g, newP, k):
        def cell_g(v, known):
            return f"{v:+.3f}" if known else "—"

        def cell_n(v):
            return f"**{n(v)}**" if k >= 11 else "—"

        rows = [
            ("$w^{(2)}_1$", P["W2"][0], g["W2"][0], newP["W2"][0], k >= 8),
            ("$w^{(2)}_2$", P["W2"][1], g["W2"][1], newP["W2"][1], k >= 8),
            ("$b^{(2)}$", P["b2"], g["b2"], newP["b2"], k >= 8),
            ("$w^{(1)}_{11}$", P["W1"][0, 0], g["W1"][0, 0], newP["W1"][0, 0], k >= 10),
            ("$w^{(1)}_{12}$", P["W1"][0, 1], g["W1"][0, 1], newP["W1"][0, 1], k >= 10),
            ("$b^{(1)}_1$", P["b1"][0], g["b1"][0], newP["b1"][0], k >= 10),
            ("$w^{(1)}_{21}$", P["W1"][1, 0], g["W1"][1, 0], newP["W1"][1, 0], k >= 10),
            ("$w^{(1)}_{22}$", P["W1"][1, 1], g["W1"][1, 1], newP["W1"][1, 1], k >= 10),
            ("$b^{(1)}_2$", P["b1"][1], g["b1"][1], newP["b1"][1], k >= 10),
        ]
        md = "| Parameter | Old | Gradient $\\partial J/\\partial(\\cdot)$ | New |\n|---|---:|---:|---:|\n"
        md += "\n".join(f"| {nm} | {n(o)} | {cell_g(gr, kn)} | {cell_n(nw)} |" for nm, o, gr, nw, kn in rows)
        return md

    def loss_plot(history, k, J_new):
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        xs = list(range(len(history)))
        ax.plot(xs, history, "o-", color="#3a6ea5", ms=4, label="loss at start of pass")
        if k == LAST:
            ax.plot([xs[-1], xs[-1] + 1], [history[-1], J_new], "o--", color="#1e8449", ms=5,
                    label="after this update")
        ax.set_xlabel("pass")
        ax.set_ylabel("loss $J$")
        ax.set_ylim(bottom=0)
        ax.set_title(f"Loss history (current $J$ = {history[-1]:.4f})", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    return loss_plot, param_table


@app.cell
def _(
    LAST,
    PHASE_COLOR,
    PHASE_NAME,
    STEP_NAMES,
    backward,
    draw_network,
    forward,
    get_S,
    loss_plot,
    lr_slider,
    mo,
    next_btn,
    param_table,
    pass_btn,
    plt,
    prev_btn,
    progress_html,
    reset_btn,
    step_text,
    update,
):
    _S = get_S()
    _k = _S["step"]
    _lr = lr_slider.value
    _P = _S["params"]
    _f = forward(_P)
    _g = backward(_P, _f)
    _newP = update(_P, _g, _lr)
    _fn = forward(_newP)

    _name, _ph = STEP_NAMES[_k]
    _title = f"Pass {_S['passes'] + 1}  ·  Step {_k}/{LAST}: {_name}"
    if PHASE_NAME[_ph]:
        _title += f"   ({PHASE_NAME[_ph]})"

    _net = draw_network(_P, _f, _g, _newP, _fn, _k, _title, PHASE_COLOR[_ph])
    _net_html = mo.as_html(_net)
    plt.close(_net)
    _lp = loss_plot(_S["history"], _k, _fn["J"])
    _lp_html = mo.as_html(_lp)
    plt.close(_lp)

    mo.vstack(
        [
            mo.hstack(
                [lr_slider, prev_btn, next_btn, pass_btn, reset_btn,
                 mo.md(f"**Pass {_S['passes'] + 1}, step {_k}/{LAST}** · passes completed: {_S['passes']}")],
                justify="start",
                align="center",
            ),
            mo.Html(progress_html(_k)),
            _net_html,
            mo.hstack(
                [
                    mo.callout(mo.md(step_text(_k, _P, _f, _g, _newP, _fn, _lr)), kind="neutral"),
                    mo.vstack([mo.md(param_table(_P, _g, _newP, _k)), _lp_html]),
                ],
                widths=[3, 2],
                align="start",
                gap=2,
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
