"""
A toy CNN: telling vertical lines from horizontal lines.

Each image is a tiny 8x8 grayscale picture that contains ONE bright line,
either vertical (label 0) or horizontal (label 1), at a random position,
plus some random noise.

Why a CNN?
  - A convolution filter is a small window (here 3x3) that slides over the
    whole image. The SAME weights are used at every position, so a filter
    that learns "vertical edge" detects it wherever the line is.
  - Max pooling keeps the strongest response in each region, so the exact
    position of the line matters less.
"""

import matplotlib.pyplot as plt
import torch
from torch import nn

torch.manual_seed(0)

# ── 1. Data ────────────────────────────────────────────────────────────────
# Generate N images of size 8x8.
# PyTorch expects image tensors with shape (batch, channels, height, width).
# Grayscale images have 1 channel, so X has shape (N, 1, 8, 8).
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

# Look at one example of each class (rounded so it is easy to read)
print("\nExample vertical line (label 0):")
print(X_train[0, 0].round(decimals=1))
print("\nExample horizontal line (label 1):")
print(X_train[1, 0].round(decimals=1))

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
#   (1, 8, 8) --Conv2d--> (4, 8, 8) --MaxPool2d--> (4, 4, 4) --Flatten--> (64,)
#             --Linear--> (1,)  one logit
model = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1),  # 1 input channel -> 4 filters
    nn.ReLU(),  # non-linearity
    nn.MaxPool2d(2),  # halve height and width: 8x8 -> 4x4
    nn.Flatten(),  # (4, 4, 4) -> 64 numbers
    nn.Linear(4 * 4 * 4, 1),  # 64 features -> 1 logit
)

# Count the number of parameters
#   Conv2d : 4 filters * (3*3 weights) + 4 biases = 40
#   Linear : 64 weights + 1 bias                  = 65

# (Optional) Only 2 kernels -- uncomment to replace the model above.
# There are only two patterns (vertical / horizontal), so 2 kernels are enough.
#   (1, 8, 8) --Conv2d--> (2, 8, 8) --MaxPool2d--> (2, 4, 4) --Flatten--> (32,)
#             --Linear--> (1,)  one logit
# model = nn.Sequential(
#     nn.Conv2d(1, 2, kernel_size=3, padding=1),  # 1 input channel -> 2 filters
#     nn.ReLU(),  # non-linearity
#     nn.MaxPool2d(2),  # halve height and width: 8x8 -> 4x4
#     nn.Flatten(),  # (2, 4, 4) -> 32 numbers
#     nn.Linear(2 * 4 * 4, 1),  # 32 features -> 1 logit
# )
#
# Count the number of parameters
#   Conv2d : 2 filters * (3*3 weights) + 2 biases = 20
#   Linear : 32 weights + 1 bias                  = 33
#   Total                                         = 53

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal number of parameters in the model: {pytorch_total_params}")

# ── 3. Train ───────────────────────────────────────────────────────────────
# Same recipe as the XOR classification: logits + BCEWithLogitsLoss.
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
# Each 3x3 filter is what the conv layer learned to look for.
# Some should look like vertical or horizontal stripes.
conv = model[0]  # layers in a Sequential are indexed like a list: [0] is the Conv2d
assert isinstance(conv, nn.Conv2d)  # sanity check (also tells the type checker)
print(f"\nConv weight shape: {tuple(conv.weight.shape)}  (filters, in_channels, h, w)")

print("\nLearned 3x3 filters:")
for k, w in enumerate(conv.weight.data):
    print(f"Filter {k}:\n{w[0].round(decimals=2)}")

# Plot the kernels as pictures (one subplot per kernel).
# Compare with S2_toy_mlp.py: each MLP neuron is a full 8x8 image,
# while each CNN kernel is just 3x3 and is reused at every position.
n_kernels = conv.out_channels  # 4 (or 2 if you use the optional model)
fig, axes = plt.subplots(1, n_kernels, figsize=(2.5 * n_kernels, 2.5))
for k, ax in enumerate(axes.flat):
    w = conv.weight.data[k, 0]  # one 3x3 kernel (in_channels = 1)
    im = ax.imshow(w, cmap="bwr")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # weight value scale
    ax.set_title(f"kernel {k}")
    ax.axis("off")
plt.suptitle("Learned 3x3 kernels of the conv layer")
plt.tight_layout()
plt.show()
