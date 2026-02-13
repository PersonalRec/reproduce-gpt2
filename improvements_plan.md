# GPT-2 Reproduction - Possible Improvements

I will try to squeeze the most of this model, trying to keep the number of model's parameters as small as possible. I will mainly focus on architecture improvements, better datasets, etc. rather than direct scaling to see how far can we push this architecture with modern techniques while keeping parameter count as small as possible. Another important point: trying to minimize the training cost while keeping the model training efficient.

### Weight Initialization

Switch from std=0.02 to Xavier/Glorot or Kaiming/He Initialization

Better weight initialization helps with training stability and convergence speed. Xavier works well for tanh/sigmoid, while Kaiming/He is better for ReLU-like activations. Can reduce vanishing/exploding gradients and lead to faster convergence especially in deeper networks.


### SwiGLU Activation Function (done)

Switch from GELU to more recent non-linear functions (e.g., SwiGLU activation)

SwiGLU (used in LLaMA, PaLM) has shown better performance than GELU in modern LLMs. Empirically shown to improve model quality with minimal computational overhead.


### Mixture of Experts (MOE)

Implement MOE architecture

MOE allows scaling model capacity without proportionally increasing compute. Only a subset of experts are activated per token, enabling much larger models while keeping inference cost manageable. However this is a complex implementation that causes training instability, requires careful load balancing.


### RoPE Positional Embeddings (done)

Implement RoPE (rotary positional embeddings, LLaMA-style) instead of learned positional embeddings

RoPE encodes relative positions through rotation in embedding space, enabling better length extrapolation (can handle sequences longer than training length). More parameter-efficient than learned embeddings. Used in LLaMA, GPT-NeoX, and shown to improve long-context understanding.


### Adding dropout layers

Current implementation from Karpathy does not have any dropout layers in comparison to the original GPT-2 paper. At this point we do not see any overfitting, but later we might consider to add dropout to prevent this (p=0.1).

### Using RMSNorm instead of LayerNorm (done)

RMSNorm (Root Mean Square Layer Normalization) is a simpler and faster alternative to LayerNorm that's used in modern LLMs. ~40% faster than LayerNorm while having the same performance.

### Learning Rate Increase (done)

Increase the lr up to x3

Higher learning rates can lead to faster convergence and sometimes better final performance. Recent research shows that models can tolerate higher LRs than traditionally thought, especially with proper warmup and decay schedules. Can significantly reduce training time.

### Training Steps Increase (done)

Increase the number of steps by e.g. 4 times (4 epochs)

More training steps = more exposure to data = better learning. The model hasn't fully converged after 1 epoch. Additional epochs allow the model to learn more nuanced patterns. Diminishing returns after a point, but going from 1 to 4 epochs typically shows significant improvement.


### Data Shuffling (done)

If we do several epochs instead of 1, the data comes in the exactly same order. It would be nice to shuffle/permutate data randomly, permute the documents around in every single shard in every single epoch for several epochs training. Potentially even permute the shards.


### Larger Dataset

Take bigger FineWeb-edu chunk instead of 10 billion tokens. Maybe add multi-language datasets, wikipedia, books, etc.


### Modern Tokenizer

Try to use more recent and more effective tokenizers.

Better tokenizers (like SentencePiece, BPE with larger vocab, or modern tiktoken variants) can reduce sequence lengths (fewer tokens for same text = faster training/inference), better handle multilingual text, and improve model efficiency.


### Additional Evaluation Benchmarks

Add more evals in addition to HellaSwag, e.g. Eleuther Evaluation Harness

Single benchmark can be misleading or overfit. Eleuther Evaluation Harness provides standardized evaluation across multiple tasks (MMLU, ARC, TruthfulQA, etc.) giving better signal about actual model capabilities.

### Make the model's training more optimal according to the Chinchilla scaling laws

The dataset size should be x20 bigger than the model's parameters count.


# Further improvements (Experiments): 


### Curriculum Learning

Try to do initial model reining (e.g. during the warmup) using baby-books so the models will be able to learn language patterns (hopefully) faster.


### Dual PatchNorm (Post-Embedding Normalization) (done)

Add LayerNorm/RMSNorm right after (tok_embed + pos_embed)

The embedding layer often has disproportionately large gradient norms compared to deeper layers, causing training instability. Adding normalization after the embedding "sandwiches" the input and brings gradient magnitudes in line with the rest of the network. This is a trivial 2-line change that requires no additional hyperparameters, adds negligible compute overhead, and can enable training with higher learning rates.


### Implement realtime TensorBoard chart for Vast.ai (done)

Helps track the model's training progress in real time with nice visualization


### Model FLOPs Utilization (MFU) (done)

Measure MFU during training to find out how well my setup performs.


### Implement model post-training on the custom dataset

Take completely new dataset and post-train the pre-trained model on it.

### Make a chat-model
