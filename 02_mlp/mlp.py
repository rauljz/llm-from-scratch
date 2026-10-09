"""
02 - MLP con ventana de contexto (estilo makemore part 2 / Bengio 2003).

Predice el siguiente caracter a partir de los `block_size` caracteres
anteriores, usando una tabla de embeddings real (densa) + una capa
oculta con tanh. Introduce train/dev/test split y minibatches.
"""

import torch
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"

words = open("data/names.txt").read().splitlines()

chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 3  # cuántos caracteres de contexto usamos para predecir el siguiente


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size  # empieza relleno de '.'
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # desliza la ventana
    return torch.tensor(X), torch.tensor(Y)


# --- train / dev / test split (80/10/10) ---
g = torch.Generator().manual_seed(42)
words_shuffled = [words[i] for i in torch.randperm(len(words), generator=g)]
n1 = int(0.8 * len(words_shuffled))
n2 = int(0.9 * len(words_shuffled))

Xtr, Ytr = build_dataset(words_shuffled[:n1])
Xdev, Ydev = build_dataset(words_shuffled[n1:n2])
Xtest, Ytest = build_dataset(words_shuffled[n2:])
Xtr, Ytr = Xtr.to(device), Ytr.to(device)
Xdev, Ydev = Xdev.to(device), Ydev.to(device)
Xtest, Ytest = Xtest.to(device), Ytest.to(device)
print(f"train: {Xtr.shape[0]}, dev: {Xdev.shape[0]}, test: {Xtest.shape[0]}")

# --- parámetros ---
emb_dim = 10
hidden_size = 200

g = torch.Generator(device=device).manual_seed(2147483647)
C = torch.randn((vocab_size, emb_dim), generator=g, device=device, requires_grad=True)
W1 = torch.randn((block_size * emb_dim, hidden_size), generator=g, device=device, requires_grad=True)
b1 = torch.randn(hidden_size, generator=g, device=device, requires_grad=True)
W2 = torch.randn((hidden_size, vocab_size), generator=g, device=device, requires_grad=True)
b2 = torch.randn(vocab_size, generator=g, device=device, requires_grad=True)
params = [C, W1, b1, W2, b2]
print(f"parametros totales: {sum(p.nelement() for p in params)}")


def forward(X):
    emb = C[X]  # (N, block_size, emb_dim) -- lookup de embeddings
    emb = emb.view(emb.shape[0], -1)  # concatenar contexto: (N, block_size*emb_dim)
    h = torch.tanh(emb @ W1 + b1)  # capa oculta con no-linealidad
    logits = h @ W2 + b2  # capa de salida: (N, vocab_size)
    return logits


@torch.no_grad()
def eval_loss(X, Y):
    logits = forward(X)
    return F.cross_entropy(logits, Y).item()


# --- entrenamiento con minibatches ---
num_steps = 20000
batch_size = 32

for step in range(num_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g, device=device)
    logits = forward(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in params:
        p.grad = None
    loss.backward()

    lr = 0.1 if step < 10000 else 0.01  # decay simple de learning rate
    for p in params:
        p.data -= lr * p.grad

    if step % 2000 == 0 or step == num_steps - 1:
        print(f"step {step:5d} | train batch loss {loss.item():.4f}")

print(f"\ntrain loss (completo): {eval_loss(Xtr, Ytr):.4f}")
print(f"dev loss:              {eval_loss(Xdev, Ydev):.4f}")
print(f"test loss:             {eval_loss(Xtest, Ytest):.4f}")
print("(compara con 01_bigram: ~2.45-2.52 -- aqui deberia bajar notablemente al ver mas contexto)")

# --- generar nombres nuevos ---
print("\nnombres generados:")
g2 = torch.Generator(device=device).manual_seed(2147483647)
for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        logits = forward(torch.tensor([context], device=device))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g2).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    print("  " + "".join(out))
