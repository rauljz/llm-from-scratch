"""
00 - Setup: tensores y autograd.

Ajustamos a mano una recta y = w*x + b a unos puntos con ruido,
usando gradient descent manual. Es el mismo mecanismo (forward,
loss, backward, update) que luego usaremos para entrenar un GPT.
"""

import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"usando device: {device}")

torch.manual_seed(0)

# datos: y = 3x + 2, con algo de ruido
x = torch.linspace(-5, 5, 100, device=device)
y_true = 3 * x + 2 + torch.randn_like(x) * 0.5

# parámetros que queremos aprender, inicializados al azar
w = torch.randn(1, device=device, requires_grad=True)
b = torch.randn(1, device=device, requires_grad=True)

lr = 0.01
for step in range(200):
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()  # mean squared error

    loss.backward()  # calcula d(loss)/dw y d(loss)/db

    with torch.no_grad():  # el update en sí no debe formar parte del grafo
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if step % 40 == 0:
        print(f"step {step:3d} | loss {loss.item():.4f} | w {w.item():.3f} | b {b.item():.3f}")

print(f"\nresultado final: w={w.item():.3f} (objetivo 3.0), b={b.item():.3f} (objetivo 2.0)")
