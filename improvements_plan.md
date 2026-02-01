# GPT-2 Reproduction - Possible Improvements

I will try to squeeze the most of this model, trying to keep the number of model's parameters as small as possible. I will mainly focus on architecture improvements, better datasets, etc. rather than direct scaling to see how far can we push this architecture with modern techniques while keeping parameter count as small as possible. Another important point: trying to minimize the training cost while keeping the model training efficient.

---

## Architecture Improvements

### 1. Weight Initialization

Switch from std=0.02 to Xavier/Glorot or Kaiming/He Initialization

Better weight initialization helps with training stability and convergence speed. Xavier works well for tanh/sigmoid, while Kaiming/He is better for ReLU-like activations. Can reduce vanishing/exploding gradients and lead to faster convergence especially in deeper networks.

### 2. c_attn Projection Optimization

3 * config.n_embd is an 'ugly' number which might be hard to parallelize. Maybe use 4 instead?

GPU operations are optimized for powers of 2 and multiples of certain numbers (like 64, 128). Using 4 * n_embd instead of 3 * n_embd can improve memory alignment and tensor core utilization, leading to faster matrix multiplications. Trade-off: slightly more parameters but better hardware efficiency.

### 3. SwiGLU Activation Function

Switch from GELU to more recent non-linear functions (e.g., SwiGLU activation)

SwiGLU (used in LLaMA, PaLM) has shown better performance than GELU in modern LLMs. Empirically shown to improve model quality with minimal computational overhead.

### 4. Mixture of Experts (MOE)

Implement MOE architecture

MOE allows scaling model capacity without proportionally increasing compute. Only a subset of experts are activated per token, enabling much larger models while keeping inference cost manageable. However this is a complex implementation that causes training instability, requires careful load balancing.

### 5. RoPE Positional Embeddings

Implement RoPE (rotary positional embeddings, LLaMA-style) instead of learned positional embeddings

RoPE encodes relative positions through rotation in embedding space, enabling better length extrapolation (can handle sequences longer than training length). More parameter-efficient than learned embeddings. Used in LLaMA, GPT-NeoX, and shown to improve long-context understanding.

### 6. Adding dropout layers

Current implementation from Karpathy does not have any dropout layers in comparison to the original GPT-2 paper. At this point we do not see any overfitting, but later we might consider to add dropout to prevent this (p=0.1).

---

## Training Optimization

### 7. Learning Rate Increase

Increase the lr up to x3

Higher learning rates can lead to faster convergence and sometimes better final performance. Recent research shows that models can tolerate higher LRs than traditionally thought, especially with proper warmup and decay schedules. Can significantly reduce training time.

### 8. Training Steps Increase

Increase the number of steps by e.g. 4 times (4 epochs)

More training steps = more exposure to data = better learning. The model hasn't fully converged after 1 epoch. Additional epochs allow the model to learn more nuanced patterns. Diminishing returns after a point, but going from 1 to 4 epochs typically shows significant improvement.

---

## Data Improvements

### 9. Data Shuffling

If we do several epochs instead of 1, the data comes in the exactly same order. It would be nice to shuffle/permutate data randomly, permute the documents around in every single shard in every single epoch for several epochs training. Potentially even permute the shards.

### 10. Larger Dataset

Take bigger FineWeb-edu chunk instead of 10 billion tokens. Maybe add multi-language datasets, wikipedia, books, etc.

### 11. Curriculum Learning

Try to do initial model tuning (e.g. during the warmup) using baby-books

### 12. Modern Tokenizer

Try to use more recent and more effective tokenizers.

Better tokenizers (like SentencePiece, BPE with larger vocab, or modern tiktoken variants) can reduce sequence lengths (fewer tokens for same text = faster training/inference), better handle multilingual text, and improve model efficiency.

---

## Evaluation Improvements

### 13. Additional Evaluation Benchmarks

Add more evals in addition to HellaSwag, e.g. Eleuther Evaluation Harness

Single benchmark can be misleading or overfit. Eleuther Evaluation Harness provides standardized evaluation across multiple tasks (MMLU, ARC, TruthfulQA, etc.) giving better signal about actual model capabilities.

---