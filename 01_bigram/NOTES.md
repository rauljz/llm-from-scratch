# 01 — Bigram: el modelo de lenguaje más simple

Dataset: `data/names.txt` (32032 nombres propios en inglés, uno por línea — dataset clásico de Karpathy/makemore). Se descarga con:

```bash
curl -sL https://raw.githubusercontent.com/karpathy/makemore/master/names.txt -o data/names.txt
```

## La tarea: modelar lenguaje = predecir el siguiente token

Un "modelo de lenguaje" es, en el fondo, una función que dado un trozo de texto predice una **distribución de probabilidad** sobre cuál es el siguiente token. Eso es todo. Un LLM enorme y un bigrama son la misma tarea; lo que cambia es cuánto contexto usan y cómo de compleja es la función.

## Tokenización a nivel de carácter

El "vocabulario" aquí son solo las letras a-z más un token especial `.` que marca **inicio y fin de palabra**. Así "emma" se convierte en la secuencia de pares (bigramas):

```
. e
e m
m m
m a
a .
```

Cada carácter se mapea a un entero (`stoi`, string-to-index) y viceversa (`itos`). Esto es literalmente lo mismo que hace el tokenizer de un LLM real, solo que el vocabulario ahí son sub-palabras (BPE, módulo `06_bpe_tokenizer`) en vez de caracteres sueltos.

## Versión 1: conteo puro (`bigram_counts.py`)

Un modelo bigrama dice: "la probabilidad del siguiente carácter depende solo del carácter actual" (ninguna memoria más atrás). Así que basta con:

1. Contar, en todo el dataset, cuántas veces sigue cada carácter B después de cada carácter A → matriz de cuentas 27x27 (`N[a, b]`).
2. Normalizar cada fila para que sume 1 → matriz de probabilidades.
3. Para generar texto: partir de `.` (inicio) y muestrear del vocabulario según la fila de probabilidades correspondiente, repitiendo hasta muestrear `.` (fin).

## Evaluar qué tan bueno es el modelo: negative log-likelihood

Para medir "qué tan bien predice" el modelo, no basta con mirarlo generar cosas — necesitamos un número. Usamos **negative log-likelihood** (NLL):

- Para cada bigrama del dataset, el modelo le asigna una probabilidad `p`.
- Un buen modelo asigna probabilidad alta (cerca de 1) a los bigramas que realmente ocurren.
- `log(p)` es negativo (porque `p < 1`) y tiende a `-infinito` cuando `p -> 0`. Tomamos `-log(p)` para que sea positivo, y lo promediamos sobre todo el dataset.
- **Este promedio de `-log(p)` es exactamente la "loss" (cross-entropy) que se usa para entrenar cualquier LLM.** Minimizar NLL = maximizar la probabilidad que el modelo asigna a los datos reales.

## Versión 2: red neuronal de una capa (`bigram_nn.py`)

Ahora resolvemos el mismo problema pero con gradient descent en vez de contar a mano, para introducir el flujo que usaremos siempre a partir de aquí:

1. **One-hot encoding**: el carácter de entrada se representa como un vector de 27 posiciones, todo ceros menos un 1 en su índice.
2. **Capa lineal sin bias**: `logits = one_hot @ W`, donde `W` es una matriz 27x27 de parámetros *entrenables*. Como el one-hot solo tiene un 1, esta multiplicación en realidad **selecciona una fila de W** — es decir, `W` funciona como una tabla de embeddings: cada fila de W es el "embedding" (vector de features aprendido) de un carácter. Esta es la idea de embedding que usaremos en todo el proyecto.
3. **Softmax**: convertimos los logits (números sin restricción) en una distribución de probabilidad válida (positivos, suman 1): `exp(logits) / sum(exp(logits))`.
4. **Cross-entropy loss**: `-log(p)` de la probabilidad asignada al carácter correcto, promediado sobre el batch. Es la misma métrica que en la versión de conteo.
5. **Backward + update**: `loss.backward()` calcula el gradiente de la loss respecto a `W`; actualizamos `W -= lr * W.grad`.

Al entrenar, `W` converge (aprox.) a `log(probabilidades de conteo)` — es decir, **la red neuronal aprende, por gradient descent, exactamente lo mismo que calculamos a mano contando**. Es la prueba de que gradient descent funciona, en el caso más simple posible donde ya sabemos cuál es la respuesta "correcta". A partir de aquí, seguimos usando el mismo mecanismo (embedding → transformación → softmax → cross-entropy → backward) pero con arquitecturas cada vez más potentes que sí pueden mirar más contexto.
