"""
XOR as a regression problem.

XOR is the classic example of a problem that is NOT linearly separable:
no single straight line can separate the 1s from the 0s. A network with
one hidden layer can solve it, because the hidden layer bends the input
space into a new representation where the problem becomes linear.

Here we treat the targets 0 and 1 as plain numbers and fit them with
mean squared error (MSE), just like ordinary regression. Compare with
S42_XOR_classification.py, which treats the same problem as binary
classification with a cross-entropy loss.
"""

import torch
from torch import nn

# ── 0. Random seed (optional) ──────────────────────────────────────────────
# No seed is set, so the initial weights (and the result) change each run.
# Most runs reach loss ≈ 0, but some get stuck in a local minimum:
#
#   loss ≈ 0.125   2 points predicted correctly, the other 2 stuck at 0.5.
#                  Loss = (2/4) * (0.5)^2 = 0.125
#
# Uncomment a seed to make a run reproducible:
#
# torch.manual_seed(0)  # reaches loss ≈ 0
# torch.manual_seed(1)  # gets stuck at loss ≈ 0.125

# ── 1. Data ────────────────────────────────────────────────────────────────
# All four possible inputs and their XOR outputs:
#   (0, 0) -> 0
#   (0, 1) -> 1
#   (1, 0) -> 1
#   (1, 1) -> 0
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# ── 2. Model ───────────────────────────────────────────────────────────────
# The output layer has no activation, so the model can output any real
# number. That is normal for regression: predictions are not restricted
# to (0, 1), and may land slightly below 0 or above 1.
model = nn.Sequential(
    nn.Linear(2, 2),  # hidden layer: 2 inputs -> 2 neurons
    nn.Sigmoid(),  # non-linearity (without it, the model is just linear)
    nn.Linear(2, 1),  # output layer: 2 neurons -> 1 real-valued prediction
)

# Count the number of parameters
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {pytorch_total_params}")

# ── 3. Train ───────────────────────────────────────────────────────────────
# MSELoss = mean of (prediction - target)^2 over all samples.
#   - Input : raw model outputs (any real number)
#   - Treats 0 and 1 as numbers to approximate, not as class labels
loss_fn = nn.MSELoss()

# Adam is a variant of gradient descent that adapts the step size for
# each weight. Try changing the learning rate (lr) to see its effect:
#   too small -> slow training; too large -> loss may jump around.
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(5000):
    y_pred = model(X)  # forward pass
    loss = loss_fn(y_pred, y)  # compare predictions with targets

    optimizer.zero_grad()  # clear gradients from the previous step
    loss.backward()  # backpropagation: compute gradients
    optimizer.step()  # update weights

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}: loss = {loss.item():.4f}")

# ── 4. Check ───────────────────────────────────────────────────────────────
# no_grad() turns off gradient tracking: we are only evaluating, not training.
# Predictions are real numbers; values close to 0 or 1 mean a good fit.
with torch.no_grad():
    y_pred = model(X)

print(f"\nFinal loss : {loss.item():.4f}")
print(f"Predictions: {[round(p, 3) for p in y_pred.squeeze().tolist()]}")
print(f"Targets    : {y.squeeze().tolist()}")
