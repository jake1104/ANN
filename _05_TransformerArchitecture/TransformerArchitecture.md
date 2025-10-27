# 🧠 Transformer Architecture

## 1️⃣ Overview

The **Transformer** is a model first proposed in the 2017 paper "Attention Is All You Need" by Vaswani et al.
Unlike traditional RNNs or LSTMs, it allows **parallel processing without sequential operations**,
and efficiently learns contextual dependencies through the **Self-Attention** mechanism.

---

## 2️⃣ Core Concepts

The Transformer consists of two main parts:

1. **Encoder** — Encodes the input sequence to generate an embedding representation.
2. **Decoder** — Generates a new sequence based on the encoder's output.

The overall structure is as follows:

```
Input → [Embedding + Positional Encoding] → Encoder Stack → Decoder Stack → Output
```

---

## 3️⃣ Mathematical Composition

### (1) Scaled Dot-Product Attention

Attention calculates context through the relationship between Query (Q), Key (K), and Value (V) vectors.

$$ 
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V 
$$ 

Where

* $ Q \in \mathbb{R}^{n \times d_k} $: Query matrix
* $ K \in \mathbb{R}^{n \times d_k} $: Key matrix
* $ V \in \mathbb{R}^{n \times d_v} $: Value matrix
* $ d_k $: Dimension of Key

This equation calculates **how much each word should attend** to other words.

---

### (2) Multi-Head Attention

Using only one attention head limits expressiveness, so multiple "heads" are used in parallel.

$$ 
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O 
$$ 
$$ 
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) 
$$ 

Each head learns context from a different perspective,
and combining the results creates a rich semantic representation.

---

### (3) Position-wise Feed Forward Network (FFN)

A fully connected layer applied independently to each token's position:

$$ 
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 
$$ 

This FFN non-linearly transforms the information obtained through Attention.

---

### (4) Positional Encoding

The Transformer does not recognize order, so positional information must be added.

$$ 
\text{PE}*{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d*{model}}}\right), \quad 
\text{PE}*{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d*{model}}}\right) 
$$ 

This assigns a unique periodic pattern to each position.

---

## 4️⃣ Overall Structure

### (1) Encoder Layer

One encoder consists of:

1. Multi-Head Self-Attention
2. Add & Layer Normalization
3. Feed Forward Network
4. Add & Layer Normalization

### (2) Decoder Layer

The decoder additionally includes:

1. Masked Multi-Head Self-Attention
2. Encoder-Decoder Attention
3. Feed Forward Network

---

## 5️⃣ Implementation Concept (PyTorch Style)

```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = math.sqrt(d_k)

    def forward(self, Q, K, V, mask=None):
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, V)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        out = self.attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)
        return self.W_o(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.mha(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
```

---

## 6️⃣ Summary of Advantages

✅ **Parallel processing possible** — Can process sequences at once, unlike RNNs.
✅ **Excellent long-term dependency learning** — Easily captures long-range relationships with Attention.
✅ **Scalability** — Base structure for various models like GPT, BERT, T5.

```