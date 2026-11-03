import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    x_data = np.array([1, 2, 3, 4])
    y_data = np.array([1, 3, 3, 5])
    w_slider = mo.ui.slider(start=-2, stop=3, step=0.1, value=1, label="w")
    b_slider = mo.ui.slider(start=-2, stop=5, step=0.1, value=0, label="b")
    return b_slider, mo, np, plt, sns, w_slider, x_data, y_data


@app.cell
def _(b_slider, mo, np, plt, sns, w_slider, x_data, y_data):
    w = w_slider.value
    b = b_slider.value
    mse = np.mean((y_data - (w * x_data + b)) ** 2)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = w * x_line + b
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots()
    ax.set_title(f"Linear model | MSE = {mse:.3f}")
    sns.scatterplot(
        x=x_data, y=y_data, color="tab:orange", label="Data", s=80, ax=ax
    )
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


if __name__ == "__main__":
    app.run()
