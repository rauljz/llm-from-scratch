"""
01 - Bigram, version 1: modelo por conteo puro (sin gradient descent).

Construye la tabla de probabilidades P(siguiente_char | char_actual)
contando frecuencias en el dataset, y la usa para generar nombres nuevos
y para medir la calidad del modelo con negative log-likelihood.
"""

import torch

words = open("../data/names.txt").read().splitlines()
print(f"dataset: {len(words)} nombres, ej: {words[:5]}")

# vocabulario: 26 letras + '.' como token de inicio/fin
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
print(f"vocab_size: {vocab_size} -> {itos}")

# --- 1. contar bigramas ---
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# --- 2. normalizar filas a probabilidades ---
# +1 (Laplace smoothing) para que ningún bigrama tenga probabilidad exacta 0
P = (N + 1).float()
P /= P.sum(dim=1, keepdim=True)

# --- 3. generar nombres nuevos muestreando de P ---
g = torch.Generator().manual_seed(2147483647)
print("\nnombres generados (conteo puro):")
for _ in range(10):
    out = []
    ix = 0  # empieza en '.'
    while True:
        row = P[ix]
        ix = torch.multinomial(row, num_samples=1, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print("  " + "".join(out))

# --- 4. evaluar el modelo: negative log-likelihood promedio ---
log_likelihood = 0.0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        prob = P[stoi[ch1], stoi[ch2]]
        log_likelihood += torch.log(prob).item()
        n += 1

nll = -log_likelihood / n
print(f"\nnegative log-likelihood promedio (loss): {nll:.4f}")
print("(un modelo perfecto tendría loss 0; un modelo al azar sobre 27 clases tendría ~3.30)")
