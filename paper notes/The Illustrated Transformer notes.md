# The Illustrated Transformer - Notes

> Source: https://jalammar.github.io/illustrated-transformer/
## Problem with RNNs/LSTMs
- Process input sequentially (one token at a time)
- Difficult to parallelize
- Long-range dependencies are hard to learn
- Training becomes slow for long sequences

## Transformer Solution
- Processes the entire sequence simultaneously
- Uses **Self-Attention** instead of recurrence
- Faster training
- Better long-range dependency learning

# 2. Overall Architecture
Input
   │
   ▼
Encoder Stack ×6
   │
   ▼
Encoded Representation
   │
   ▼
Decoder Stack ×6
   │
   ▼
Output

- Encoder learns representations.
- Decoder generates the output sequence.

# 3. Encoder Architecture
Each encoder block contains:
Input
   │
   ▼
Multi-Head Self-Attention
   │
   ▼
Add + LayerNorm
   │
   ▼
Feed Forward Network
   │
   ▼
Add + LayerNorm
   │
   ▼
Output

The encoder block is repeated **6 times** in the original paper.

# 4. Decoder Architecture
Each decoder block contains:
Masked Multi-Head Attention
        │
        ▼
Encoder-Decoder Attention
        │
        ▼
Feed Forward Network

Differences from encoder:
- Uses **masked attention**
- Uses **cross-attention** with encoder outputs

# 5. Word Embeddings
Words are converted into dense vectors.
Example

"cat"
↓
[0.24, -0.81, 1.43, ...]

Original embedding dimension:
- 512

# 6. Positional Encoding
Self-attention has no notion of order.
Therefore,
Token Embedding
+
Positional Encoding
↓
Final Input Embedding

Original Transformer uses:
- Sinusoidal Positional Encoding

Purpose:
- Preserve word order
- Encode token position

# 7. Self-Attention
Each token attends to every other token.
Example
The animal didn't cross the road because it was tired.
For the word
it
the attention mechanism learns
it → animal
instead of
it → road

# 8. Query, Key and Value
Each token is projected into
- Query (Q)
- Key (K)
- Value (V)

Meaning
- Query → What am I looking for?
- Key → What information do I contain?
- Value → Information passed forward

Attention pipeline:
Q × Kᵀ
↓
Softmax
↓
Attention Weights
↓
Weighted Sum of Values

# 9. Scaled Dot Product Attention
Formula
Attention(Q,K,V)
=
softmax(
QKᵀ
─────
√dk
)V

Why divide by √dk?
- Prevents very large attention scores
- Keeps gradients stable
- Improves training

# 10. Multi-Head Attention
Instead of computing one attention,
compute multiple attentions simultaneously.
Example
Head 1
- Grammar
Head 2
- Subject
Head 3
- Verb
Head 4
- Context

Each head learns different relationships.

Original Transformer
- 8 attention heads
Outputs from all heads are concatenated.

# 11. Feed Forward Network
Each token independently passes through

Linear
↓
ReLU
↓
Linear

Purpose
- Learn higher-level features
- Increase model capacity

# 12. Residual Connections

Instead of
Output

the transformer computes
Output
+
Original Input

Advantages
- Better gradient flow
- Easier optimization
- Enables deep transformers

# 13. Layer Normalization
Applied after each residual connection.

Purpose
- Stabilizes activations
- Faster convergence
- Improves training stability

# 14. Decoder Prediction

Decoder output
↓
Linear Layer
↓
Vocabulary Size
↓
Softmax
↓
Predicted Word
The word with the highest probability is selected.

# 15. Training
Uses Teacher Forcing.

Example
Input
I love transformers

Target
<START>
↓
I
↓
love
↓
transformers
The previous **correct** word is provided during training.

# 16. Inference
During testing

<START>
↓
Predict Word
↓
Feed Prediction
↓
Predict Next Word
↓
...
↓
<END>

Generation stops when `<END>` is predicted.

# 17. Original Transformer Hyperparameters
| Component | Value |
|-----------|-------|
| Encoder Layers | 6 |
| Decoder Layers | 6 |
| Embedding Size | 512 |
| Feed Forward Size | 2048 |
| Attention Heads | 8 |
| Dropout | 0.1 |

# 18. Key Concepts
- Self-Attention replaces recurrence.
- Entire sequence is processed in parallel.
- Multi-head attention captures different relationships.
- Positional encoding preserves token order.
- Feed Forward Network learns richer representations.
- Residual connections improve gradient flow.
- LayerNorm stabilizes training.
- Encoder learns representations.
- Decoder generates sequences.
