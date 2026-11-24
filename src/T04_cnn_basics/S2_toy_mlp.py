"""
A toy MLP: telling vertical lines from horizontal lines.

Same data as S1_toy_cnn.py: each image is a tiny 8x8 grayscale picture that
contains ONE bright line, either vertical (label 0) or horizontal (label 1),
at a random position, plus some random noise.

This time we use a plain MLP (fully connected network) instead of a CNN.

Why does an MLP need more parameters?
  - The MLP first flattens the 8x8 image into 64 numbers, so it loses the
    2D layout of the pixels.
  - Every hidden neuron is connected to ALL 64 pixels, with its own weight
    for each pixel. Nothing is shared: a pattern learned in the top-left
    corner has to be learned again for every other position.
  - A CNN filter has only 3x3 = 9 weights, reused at every position.
Compare the parameter count printed below with the CNN in S1_toy_cnn.py (105).
"""

import matplotlib.pyplot as plt
import torch
from torch import nn

torch.manual_seed(0)

# ── 1. Data ────────────────────────────────────────────────────────────────
# Exactly the same data as S1_toy_cnn.py.
# X has shape (N, 1, 8, 8): (batch, channels, height, width).
N = 200
SIZE = 8


def make_data(n):
    X = torch.zeros(n, 1, SIZE, SIZE)
    y = torch.zeros(n, 1)
    for i in range(n):
        pos = int(torch.randint(0, SIZE, (1,)))  # where to draw the line
        if i % 2 == 0:
            X[i, 0, :, pos] = 1.0  # vertical line   -> label 0
            y[i] = 0.0
        else:
            X[i, 0, pos, :] = 1.0  # horizontal line -> label 1
            y[i] = 1.0
    X = X + 0.1 * torch.randn_like(X)  # add noise
    return X, y


X_train, y_train = make_data(N)
X_test, y_test = make_data(50)  # fresh images the model never sees in training

print(f"X_train shape: {tuple(X_train.shape)}  (batch, channels, height, width)")

# Plot the first 8 training images as pictures
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i, 0], cmap="gray")  # X_train[i, 0] is one 8x8 image
    label = int(y_train[i].item())
    ax.set_title(f"label {label} ({'horizontal' if label else 'vertical'})")
    ax.axis("off")
plt.tight_layout()
plt.show()

# ── 2. Model ───────────────────────────────────────────────────────────────
# Shape of one image as it flows through the model:
#   (1, 8, 8) --Flatten--> (64,) --Linear--> (32,) --Linear--> (1,)  one logit
HIDDEN = 32
model = nn.Sequential(
    nn.Flatten(),  # (1, 8, 8) -> 64 numbers: the 2D layout is gone
    nn.Linear(SIZE * SIZE, HIDDEN),  # every hidden neuron sees all 64 pixels
    nn.ReLU(),  # non-linearity
    nn.Linear(HIDDEN, 1),  # 32 features -> 1 logit
)

# Count the number of parameters
#   Linear 1 : 64 * 32 weights + 32 biases = 2080
#   Linear 2 : 32 weights + 1 bias         =   33
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal number of parameters in the model: {pytorch_total_params}")
print("(The CNN in S1_toy_cnn.py uses only 105 parameters.)")

# ── 3. Train ───────────────────────────────────────────────────────────────
# Same recipe as the CNN: logits + BCEWithLogitsLoss.
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    logits = model(X_train)  # forward pass
    loss = loss_fn(logits, y_train)  # compare logits with true labels

    optimizer.zero_grad()  # clear gradients from the previous step
    loss.backward()  # backpropagation: compute gradients
    optimizer.step()  # update weights

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss = {loss.item():.4f}")

# ── 4. Check ───────────────────────────────────────────────────────────────
# Evaluate on the TEST images to see if the model generalizes.
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)  # logits -> probabilities in (0, 1)
    preds = (probs >= 0.5).float()  # probabilities -> class 0 or 1
    accuracy = (preds == y_test).float().mean()

print(f"\nFinal training loss: {loss.item():.4f}")
print(f"Test accuracy      : {accuracy.item() * 100:.1f}%")
print(f"First 10 predictions: {preds[:10].squeeze().tolist()}")
print(f"First 10 true labels: {y_test[:10].squeeze().tolist()}")

# ── 5. Peek inside ─────────────────────────────────────────────────────────
# Each hidden neuron has one weight per pixel (64 weights).
# Reshaping them back to 8x8 shows what each neuron looks for.
# Compare: a CNN filter is just 3x3, while each MLP neuron is a full 8x8 image.
fc1 = model[1]  # [0] is Flatten, [1] is the first Linear layer
assert isinstance(fc1, nn.Linear)  # sanity check (also tells the type checker)
print(f"\nFirst Linear weight shape: {tuple(fc1.weight.shape)}  (hidden, pixels)")

fig, axes = plt.subplots(2, 4, figsize=(10, 4.5))
for k, ax in enumerate(axes.flat):
    w = fc1.weight.data[k].reshape(SIZE, SIZE)  # 64 weights -> 8x8 picture
    im = ax.imshow(w, cmap="bwr")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # weight value scale
    ax.set_title(f"neuron {k}")
    ax.axis("off")
plt.suptitle("Weights of the first 8 hidden neurons (each is 8x8)")
plt.tight_layout()
plt.show()
