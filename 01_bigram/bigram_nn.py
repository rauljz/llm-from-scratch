"""
01 - Bigram, version 2: la misma tarea, resuelta con una red neuronal
de una sola capa entrenada por gradient descent.

Objetivo: comprobar que backprop, sin saber nada a priori, converge
a (aprox.) lo mismo que calculamos a mano contando en bigram_counts.py.
"""

import torch
import torch.nn.functional as F

words = open("../data/names.txt").read().splitlines()

chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

# --- construir el dataset de entrenamiento: pares (char_entrada, char_siguiente) ---
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num_examples = xs.nelement()
print(f"dataset de entrenamiento: {num_examples} bigramas")

# --- parámetros: una matriz vocab_size x vocab_size ---
# W[i] es el "embedding"/fila de logits para el carácter i.
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

lr = 10.0  # lr alto porque no hay más capas ni normalización que amortigüen
num_steps = 200

for step in range(num_steps):
    # forward: one-hot @ W selecciona la fila -> logits
    xenc = F.one_hot(xs, num_classes=vocab_size).float()
    logits = xenc @ W
    # softmax manual: exp y normalizar (equivalente a F.cross_entropy pero explícito)
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    loss = -probs[torch.arange(num_examples), ys].log().mean()
    # regularización L2 pequeña: empuja W hacia 0 => probs hacia uniforme,
    # evita que el modelo memorice con logits extremos (equivalente al
    # +1 de Laplace smoothing en la versión de conteo)
    loss = loss + 0.01 * (W ** 2).mean()

    W.grad = None
    loss.backward()
    W.data -= lr * W.grad

    if step % 20 == 0 or step == num_steps - 1:
        print(f"step {step:3d} | loss {loss.item():.4f}")

# --- generar nombres con la W entrenada ---
print("\nnombres generados (red neuronal entrenada):")
g = torch.Generator().manual_seed(2147483647)
for _ in range(10):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits = xenc @ W
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print("  " + "".join(out))

print(f"\n(recuerda: bigram_counts.py llegó a loss ~2.4546 con conteo puro)")
