##Main Contributions
-Proposed MoLFormer, an encoder-only molecular transformer.
-Trained on 1.1 billion molecules (PubChem + ZINC).
-Introduced Linear Attention for scalability.
-Used Rotary Positional Embeddings (RoPE) instead of absolute positional embeddings.
-Achieved competitive or better performance than many GNNs on multiple molecular property prediction tasks.
-Showed that attention maps capture chemical substructures and some spatial relationships.

##Architecture
Input

SMILES
   |
Tokenizer
   |
Embedding Layer
   |
RoPE
   |
12 Encoder Layers
   |
Mean Pooling
   |
768-D Molecular Embedding
   |
Prediction Head

Architecture Details:
Encoder only
12 encoder layers
12 attention heads
Hidden size = 768
Feed Forward size = 3072
GELU activation
LayerNorm
Linear Attention
Mean Poolin

##Why Linear Attention?
Problem with vanilla transformer is its Attention Complexity
O(n²)
Very expensive for long SMILES.
Solution:
Linear Attention
O(n)

Benefits
Lower memory
Faster training
Enables billion-scale pretraining
Reduced GPU requirement from roughly 1000 GPUs to 16 GPUs with linear attention and adaptive bucketing.

##Why RoPE?
Instead of Absolute Position Embedding they used
Rotary Position Embedding

##Advantages
Better relative position encoding
Faster convergence
Lower validation loss
Better performance for very large datasets (>1B molecules)

##Training Data
Pretraining datasets
PubChem → 111 Million molecules
ZINC → >1 Billion molecules
Combined:
≈ 1.1 Billion molecules

##Tokenization
Molecules converted to canonical SMILES using RDKit.
Existing SMILES tokenizer used.
Vocabulary:
2357 unique tokens
+5 special tokens
Total = 2362 tokens
Maximum sequence length = 202 tokens (covers >99.4% of molecules).
Self-Supervised Learning

Uses Masked Language Modeling (MLM)

##Masking strategy
15% tokens selected
80% replaced with [MASK]
10% replaced with random token
10% left unchanged
Training Details
4 epochs
Learning rate = 1.6 × 10⁻⁴
Batch size = 1600 molecules/GPU
16 V100 GPUs
Distributed training
Adaptive bucketing based on sequence length
Fine-Tuning

Two strategies:
Frozen
Freeze transformer
Train only prediction head
Fine-Tuned
Update entire transformer

##Result
12-layer model generally performed better.
Rotary embeddings performed better for large-scale pretraining.
Frozen vs Fine-Tuned
Fine-tuned model consistently outperformed frozen embeddings.

##Downstream Tasks
Classification
BBBP
HIV
BACE
ClinTox
SIDER
Tox21

##Regression
QM9
QM8
ESOL
FreeSolv
Lipophilicity

##Results
Outperformed all baselines on 3/6 classification tasks and was a close second on the remaining three.
Outperformed several GNN baselines on multiple regression tasks such as ESOL, FreeSolv, and Lipophilicity.
Learned embeddings correlated well with molecular similarity.
Attention maps captured bond connectivity and medium-range spatial relationships between atoms.

##Limitations
Does not explicitly use 3D molecular geometry.
Requires very large-scale pretraining.
Specialized 3D GNNs (e.g., SchNet, DimeNet) still perform better on some quantum chemistry tasks.
