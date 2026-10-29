# gpt-clone

A GPT language model built from first principles, one layer at a time: start with gradient descent and activation functions in NumPy, end with a decoder-only transformer in PyTorch that trains and generates text. Every component is written by hand and checked against a known answer before the next one builds on it.

Status: in progress. The math foundations are in; the network, tokenizer and transformer pieces are next.

## Progress

| Stage | Component | Status |
|---|---|---|
| Foundations | Gradient descent on f(x) = x² | done: `foundations/gradient_descent.py` |
| | Sigmoid and ReLU | done: `foundations/activations.py` |
| | Numerically stable softmax (max-subtraction before `exp`) | done: `foundations/softmax.py` |
| | Loss functions, linear regression, single neuron | next |
| Neural nets | Backprop by hand, multi-layer backprop, MLP, weight init | next |
| | Training loop, diagnostics, dead-ReLU detection | next |
| Language data | Character vocabulary, BPE tokenizer, batched loader | next |
| Transformer | Embeddings, positional encoding, self-attention, multi-head attention | next |
| | LayerNorm / RMSNorm, transformer block, GPT model | next |
| Inference | Training run, autoregressive generation, KV cache, grouped-query attention | next |

## Layout

```
foundations/   NumPy primitives: optimization, activations, softmax
model/         transformer components (next)
data/          tokenizer and data pipeline (next)
```

## Run

```bash
pip install -r requirements.txt
python -c "from foundations.softmax import Solution; import numpy as np; print(Solution().softmax(np.array([1000., 1001., 1002.])))"
# [0.09   0.2447 0.6652]  (no overflow, thanks to the max shift)
```

The exercises follow the sequence of the [NeetCode ML course](https://neetcode.io), which is also how each piece gets tested.
