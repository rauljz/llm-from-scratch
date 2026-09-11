# 00 — Setup: tensores y autograd

Antes de tocar nada de "lenguaje", necesitamos entender la herramienta base: **tensores** y **autograd** de PyTorch. Todo LLM, por dentro, es solo esto: multiplicar y sumar tensores, y ajustar sus valores para minimizar una pérdida (loss).

## Tensor

Un tensor es un array multidimensional (como un array de numpy) que además puede vivir en GPU y llevar la cuenta de las operaciones que se le aplican, para poder derivar automáticamente.

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
```

## ¿Por qué "aprende" una red neuronal?

Una red neuronal es una función con parámetros (pesos). "Entrenar" significa:

1. Le damos un input, calcula un output (**forward pass**).
2. Comparamos el output con lo que debería ser, con una función de **loss** (número que dice "cuánto de mal lo hemos hecho").
3. Calculamos el gradiente del loss respecto a cada parámetro (**backward pass** / backpropagation): cuánto y en qué dirección cambiaría el loss si moviera un poco cada parámetro.
4. Movemos cada parámetro un poquito en la dirección que reduce el loss (**gradient descent**).
5. Repetimos miles/millones de veces.

## Autograd

PyTorch calcula el paso 3 automáticamente. Si marcas un tensor con `requires_grad=True`, PyTorch construye un grafo de todas las operaciones que le aplicas, y `.backward()` recorre ese grafo hacia atrás (regla de la cadena) para darte el gradiente en `.grad`.

Ver `tensors_intro.py` para un ejemplo mínimo: ajustamos a mano una recta `y = w*x + b` a unos puntos, usando gradient descent, sin ninguna capa de red neuronal de por medio — el mismo mecanismo que luego entrenará un GPT de millones de parámetros.

## MPS (GPU de Apple Silicon)

En vez de `.cuda()`, en Mac con chip Apple Silicon usamos `.to("mps")`. Lo comprobamos con `torch.backends.mps.is_available()`.
