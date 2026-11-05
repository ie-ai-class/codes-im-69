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
    # Gradient Descent, Step by Step

    Fit a line $\hat{y} = \theta_0 + \theta_1 x_1$ to noisy data by minimizing the MSE loss

    $$
    J(\theta) = \frac{1}{2m} \lVert X\theta - Y \rVert^2
    $$

    with the update rule $\theta \leftarrow \theta - \eta \, \frac{1}{m} X^\top (X\theta - Y)$.
    """)
    return


@app.cell
def _(np):
    # Dataset: 100 examples, column of ones for the intercept + one input variable
    np.random.seed(seed=1)
    m = 100
    X = np.ones([m, 2])
    X[:, 1] = np.linspace(-5, 5, m)
    theta_true = np.array([2, 2])  # intercept and slope used to generate data
    noise = np.random.randn(m) * 0.2
    Y = X @ theta_true + noise * 10

    def loss(theta):
        r = X @ theta - Y
        return (1 / (2 * m)) * r @ r

    # Loss surface over a grid of (theta0, theta1)
    no_dots = 40
    t0 = np.linspace(-5, 10, no_dots)
    t1 = np.linspace(-1, 5, no_dots)
    tx, ty = np.meshgrid(t0, t1, indexing="ij")
    cost = np.array([[loss(np.array([a, b])) for b in t1] for a in t0])
    return X, Y, cost, loss, m, tx, ty


@app.cell
def _(mo):
    controls = (
        mo.md(
            """
            **Gradient descent settings**

            {theta0} {theta1}

            {lr} {epochs}
            """
        )
        .batch(
            theta0=mo.ui.slider(-5, 10, step=0.5, value=0, label="start θ₀ (intercept)"),
            theta1=mo.ui.slider(-1, 5, step=0.5, value=5, label="start θ₁ (slope)"),
            lr=mo.ui.slider(0.01, 0.5, step=0.01, value=0.2, label="learning rate η"),
            epochs=mo.ui.slider(5, 60, step=1, value=30, label="epochs"),
        )
        .form(submit_button_label="Run gradient descent")
    )
    return (controls,)


@app.cell
def _(mo):
    get_step, set_step = mo.state(0)
    return get_step, set_step


@app.cell
def _(X, Y, controls, loss, m, np, set_step):
    _params = controls.value or {"theta0": 0, "theta1": 5, "lr": 0.2, "epochs": 30}
    lr = _params["lr"]
    epochs = int(_params["epochs"])

    # theta_hist[k] and cost_hist[k] are the parameters/loss after k updates
    _theta = np.array([_params["theta0"], _params["theta1"]], dtype=float)
    theta_hist = [_theta]
    for _ in range(epochs):
        _theta = _theta - lr * (1 / m) * X.T @ (X @ _theta - Y)
        theta_hist.append(_theta)
    theta_hist = np.array(theta_hist)
    cost_hist = np.array([loss(_t) for _t in theta_hist])

    set_step(0)  # restart from step 0 whenever settings change
    return cost_hist, epochs, theta_hist


@app.cell
def _(epochs, mo, set_step):
    prev_btn = mo.ui.button(label="◀ Prev", on_click=lambda _: set_step(lambda s: max(0, s - 1)))
    next_btn = mo.ui.button(label="Next ▶", on_click=lambda _: set_step(lambda s: min(epochs, s + 1)))
    jump_btn = mo.ui.button(label="+5 ▶▶", on_click=lambda _: set_step(lambda s: min(epochs, s + 5)))
    reset_btn = mo.ui.button(label="⟲ Reset", on_click=lambda _: set_step(0))
    return jump_btn, next_btn, prev_btn, reset_btn


@app.cell
def _(X, Y, cost, np, plt, tx, ty):
    def plot_step(i, theta_hist, cost_hist):
        """Draw data+fit, loss contour and loss surface after i updates."""
        fig = plt.figure(figsize=plt.figaspect(0.3))
        th0, th1 = theta_hist[: i + 1, 0], theta_hist[: i + 1, 1]

        # Data with fitted line
        ax2 = fig.add_subplot(1, 3, 1)
        ax2.plot(X[:, 1], Y, "bo", label="data")
        _xs = np.array([-5, 5])
        ax2.plot(_xs, theta_hist[i, 0] + theta_hist[i, 1] * _xs, "k-", label=r"fitted line: $y=x\theta$")
        ax2.set_ylim(Y.min() - 5, Y.max() + 5)
        ax2.set_xlabel("input: $x_1$", fontsize=15)
        ax2.set_ylabel("Y", fontsize=15)
        ax2.set_title("Labelled data & model output", fontsize=18)
        ax2.legend(loc=2)

        # Loss contour
        ax12 = fig.add_subplot(1, 3, 2)
        ax12.contour(tx, ty, cost, levels=25)
        ax12.plot(th0, th1, "r-")
        ax12.plot(th0[-1], th1[-1], "ro")
        ax12.set_xlim(tx.min(), tx.max())
        ax12.set_ylim(ty.min(), ty.max())
        ax12.set_xlabel(r"$\theta_0$")
        ax12.set_ylabel(r"$\theta_1$")
        ax12.set_title(r"$\theta$ = ({:.3f}, {:.3f})".format(*theta_hist[i]), fontsize=18)

        # Loss surface (3D)
        ax1 = fig.add_subplot(1, 3, 3, projection="3d")
        ax1.plot_wireframe(tx, ty, cost, linewidth=0.5)
        ax1.plot(th0, th1, cost_hist[: i + 1], "r-")
        ax1.plot([th0[-1]], [th1[-1]], [cost_hist[i]], "ro")
        ax1.set_xlim(tx.min(), tx.max())
        ax1.set_ylim(ty.min(), ty.max())
        ax1.set_zlim(0, cost.max())
        ax1.set_xlabel(r"$\theta_0$")
        ax1.set_ylabel(r"$\theta_1$")
        ax1.set_zlabel("loss")
        ax1.set_title(f"Loss at step {i} = {cost_hist[i]:.3f}", fontsize=18)

        fig.tight_layout()
        return fig

    return (plot_step,)


@app.cell
def _(
    controls,
    cost_hist,
    epochs,
    get_step,
    jump_btn,
    mo,
    next_btn,
    plot_step,
    prev_btn,
    reset_btn,
    theta_hist,
):
    _i = min(get_step(), epochs)
    mo.vstack(
        [
            controls,
            mo.hstack(
                [prev_btn, next_btn, jump_btn, reset_btn, mo.md(f"**Step {_i} / {epochs}**")],
                justify="start",
                align="center",
            ),
            plot_step(_i, theta_hist, cost_hist),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
