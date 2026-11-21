"""
XOR as a binary classification problem.

XOR is the classic example of a problem that is NOT linearly separable:
no single straight line can separate the 1s from the 0s. A network with
one hidden layer can solve it, because the hidden layer bends the input
space into a new representation where the classes become separable.

This script also shows that training a neural network is a non-convex
optimization problem. With only 2 hidden neurons, gradient descent can
get stuck in a local minimum depending on the random initial weights.
"""

import torch
from torch import nn

# ── 0. Random seed ─────────────────────────────────────────────────────────
# The seed fixes the initial weights, so each run is reproducible.
# Try each seed below and compare the final loss:
#
#   seed  1 -> loss ≈ 0      Escapes local minima, all 4 points correct.
#   seed 10 -> loss ≈ 0.347  Local minimum: 2 points correct, the other 2
#                            stuck at p = 0.5.
#                            Loss = (2/4) * ln(2) ≈ 0.347
#   seed 40 -> loss ≈ 0.477  Local minimum: 1 corner isolated, the other 3
#                            points (labels 1, 1, 0) collapse together, so
#                            the best it can do is predict p = 2/3 for all 3.
#                            Loss = (3/4) * H(2/3) ≈ 0.477
#
# Widening the hidden layer (e.g. 2 -> 4 neurons) makes these traps much rarer.

torch.manual_seed(1)
# torch.manual_seed(10)
# torch.manual_seed(40)

# ── 1. Data ────────────────────────────────────────────────────────────────
# All four possible inputs and their XOR labels:
#   (0, 0) -> 0
#   (0, 1) -> 1
#   (1, 0) -> 1
#   (1, 1) -> 0
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# ── 2. Model ───────────────────────────────────────────────────────────────
# Note: there is NO sigmoid at the end. The model outputs raw scores
# (logits), and BCEWithLogitsLoss applies the sigmoid internally.
model = nn.Sequential(
    nn.Linear(2, 2),  # hidden layer: 2 inputs -> 2 neurons
    nn.Sigmoid(),  # non-linearity (without it, the model is just linear)
    nn.Linear(2, 1),  # output layer: 2 neurons -> 1 logit
)

# Count the number of parameters
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {pytorch_total_params}")

# ── 3. Train ───────────────────────────────────────────────────────────────
# BCEWithLogitsLoss = Sigmoid + BCELoss combined into one step.
#   - Input : raw logits (any real number), NOT probabilities
#   - Why   : more numerically stable than applying Sigmoid then BCELoss
#             (it avoids computing log(0) when p is very close to 0 or 1)
# If you use nn.BCELoss instead, you must add nn.Sigmoid() to the model output.
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(5000):
    logits = model(X)  # forward pass
    loss = loss_fn(logits, y)  # compare logits with true labels

    optimizer.zero_grad()  # clear gradients from the previous step
    loss.backward()  # backpropagation: compute gradients
    optimizer.step()  # update weights

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}: loss = {loss.item():.4f}")

# ── 4. Check ───────────────────────────────────────────────────────────────
# The model outputs logits, so apply sigmoid ourselves to get probabilities.
# Then threshold at 0.5 to get class predictions (0 or 1).
with torch.no_grad():
    logits = model(X)
    probs = torch.sigmoid(logits)  # logits -> probabilities in (0, 1)
    preds = (probs >= 0.5).float()  # probabilities -> class 0 or 1

print(f"\nFinal loss   : {loss.item():.4f}")
print(f"Probabilities: {[round(p, 3) for p in probs.squeeze().tolist()]}")
print(f"Predictions  : {preds.squeeze().tolist()}")
print(f"True labels  : {y.squeeze().tolist()}")
