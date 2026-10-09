# 02 — MLP con contexto: embeddings de verdad y no-linealidad

El bigrama de `01_bigram` tiene un techo claro: solo mira **un** carácter de contexto. "¿Qué viene después de 'q'?" siempre tiene la misma respuesta, ignorando todo lo anterior. Para mejorar, el modelo necesita ver una ventana de varios caracteres. Ese es el salto de este módulo (estilo makemore part 2, Bengio et al. 2003).

## Context window (`block_size`)

En vez de predecir el siguiente carácter a partir de 1, lo predecimos a partir de los últimos `block_size` caracteres (aquí 3). `"emma"` con `block_size=3` genera ejemplos de entrenamiento:

```
...  -> e
..e  -> m
.em  -> m
emm  -> a
mma  -> .
```

(`.` rellena el contexto antes del inicio de la palabra). Cada ejemplo X ahora es un vector de 3 índices, no 1.

## Embeddings de verdad (no one-hot)

En `01_bigram`, "embedding" era indexar una fila de una matriz 27x27 — cada carácter tenía tantos números como el vocabulario entero. Aquí usamos una tabla de embeddings `C` de forma `(vocab_size, emb_dim)` con `emb_dim` mucho menor que vocab_size (p.ej. 10). Cada carácter se representa con un vector denso de 10 números, **aprendidos** — el modelo decide qué "significa" cada dimensión para resolver la tarea. Esta es la noción de embedding que se usa en cualquier LLM real.

Para los 3 caracteres de contexto, se busca su embedding y se **concatenan**: `(3, emb_dim) -> (3*emb_dim,)`. Esa concatenación es el input de la red.

## Por qué hace falta una no-linealidad

Si encadenas capas puramente lineales (`x @ W1 @ W2`), el resultado sigue siendo una función lineal de `x` — no gana ninguna capacidad extra respecto a una sola capa. Por eso metemos una función de activación no-lineal entre capas: aquí `tanh`. Con eso, la red puede aproximar funciones mucho más complejas (combinaciones de contextos, no solo promedios).

Arquitectura: `embeddings concatenados -> Linear -> tanh -> Linear -> logits -> softmax -> cross-entropy`.

## Train / dev / test split: el concepto de generalizar

Con más parámetros, el modelo puede empezar a **memorizar** el dataset de entrenamiento en vez de aprender el patrón general (overfitting). Para detectarlo, partimos el dataset en tres:

- **train** (80%): con lo que se ajustan los parámetros.
- **dev/validation** (10%): para medir el loss en datos que el modelo no ha visto durante el entrenamiento, y así decidir hiperparámetros (tamaño de embedding, de la capa oculta, learning rate...) sin "hacer trampa".
- **test** (10%): se mira solo al final, una vez decidido todo, para tener una estimación honesta de qué tan bien generaliza.

Si el loss de train sigue bajando pero el de dev se estanca o sube, el modelo está memorizando en vez de generalizar — señal para reducir capacidad, regularizar, o parar antes.

## Minibatches

Con >200k ejemplos, calcular el gradiente exacto sobre todo el dataset en cada paso es lento. En su lugar, en cada paso se toma un **minibatch** aleatorio (p.ej. 32 ejemplos), se calcula el loss y gradiente solo sobre ese subconjunto, y se actualiza. Es una estimación ruidosa del gradiente real, pero mucho más rápida de calcular, y en la práctica funciona muy bien (este es, literalmente, el "**S**" de **S**GD: *stochastic* gradient descent).
