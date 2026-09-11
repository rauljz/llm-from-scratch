# llm-from-scratch

Proyecto personal de aprendizaje: construir un LLM tipo GPT desde cero, en PyTorch, entendiendo cada pieza antes de usarla.

Filosofía: cada módulo tiene código + un `NOTES.md` que explica los conceptos nuevos que aparecen ahí, antes de escalar al siguiente. Nada de "magic imports" sin entender qué hacen.

Hardware: Mac con chip Apple M5 (sin GPU NVIDIA) → entrenamos con PyTorch + backend MPS, a escala pequeña (modelos de pocos millones de parámetros, datasets de texto pequeños). La arquitectura es la misma que un LLM real; lo que cambia es la escala.

## Roadmap

- [x] `00_setup` — entorno, tensores, autograd básico
- [ ] `01_bigram` — el modelo de lenguaje más simple posible (conteo de bigramas → versión con red neuronal), tokenización a nivel de carácter, embeddings, cross-entropy loss
- [ ] `02_mlp` — modelo de lenguaje con MLP y contexto de varios caracteres (estilo makemore)
- [ ] `03_attention` — self-attention implementada a mano (Q/K/V, softmax, escalado, máscara causal)
- [ ] `04_transformer` — bloque transformer completo (multi-head attention + feed-forward + residuales + layernorm) → GPT pequeño
- [ ] `05_train_gpt` — entrenar el GPT pequeño de principio a fin sobre un dataset de texto y generar texto nuevo
- [ ] `06_bpe_tokenizer` — tokenizer BPE (byte-pair encoding) propio, en vez de nivel de carácter
- [ ] `07_scale_up` — subir escala del modelo/dataset, checkpointing, mejores prácticas de entrenamiento
- [ ] `08_finetune` — fine-tuning / instruction tuning básico (a decidir más adelante)

Cada carpeta se añade cuando llegamos a ella, no todas de golpe.

## Entorno

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Referencia principal de inspiración: el curso "Neural Networks: Zero to Hero" de Andrej Karpathy (nanoGPT / makemore), adaptado y explicado paso a paso.
