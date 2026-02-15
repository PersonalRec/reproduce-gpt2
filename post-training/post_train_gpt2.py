"""
Post-Training Script for GPT-2 124M on OD Documentation

Loads a pre-trained checkpoint and continues training on a small domain-specific
dataset (OD documentation). Adapts the model to generate documentation-style text.

Checkpoint: ../logs/log124M_050226/model_38145.pt (pre-trained GPT-2 124M with RoPE + SwiGLU + RMSNorm)
Dataset:    ./data/ (tokenized .npy shards from dataset_test.ipynb)

Usage:
    Single GPU:   python post_train_gpt2.py
    Multi-GPU:    torchrun --standalone --nproc_per_node=2 post_train_gpt2.py
"""

from dataclasses import dataclass
import torch, math
import torch.nn as nn
from torch.nn import functional as F
import time
import inspect
import os
import csv
import tiktoken
import numpy as np


@dataclass
class GPTConfig():
    block_size: int = 1024 # maximum sequence length (context length)
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 tokens + 1 <|endoftext|> token = 50257 which is inefficient in terms of cuda processing, new vocab size 50304 could be easily divided by a lot of numbers
    n_layer: int = 12 # number of layers (transformer blocks: each block has attention + MLP + RMSNorms)
    n_head: int = 12 # number of heads per transformer blocks. Each head sees 768 ÷ 12 = 64 dimensions. Different heads can learn different attention patterns.
    n_embd: int = 768 # embedding dimension

    # additional options
    use_rope: bool = True      # use RoPE embedding for training
    rope_base: float = 10000.0
    mlp_type: str = "swiglu"   # MLP activation type: "gelu" or "swiglu"


# ========================================= Post-Training parameters ==================================================================

# Post-training hyperparameters (tuned for small domain-specific dataset ~141K tokens)
# The dataset is tiny compared to pre-training (141K vs 10B tokens), so we use:
#   - Much lower learning rate to avoid catastrophic forgetting
#   - Smaller batch size (dataset has ~124 train windows of 1024 tokens)
#   - Multiple epochs over the small dataset

torch.set_float32_matmul_precision('high')

# Paths (resolved relative to this script's location, so it works from any cwd)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(SCRIPT_DIR, "..", "logs", "log124M_050226", "model_38145.pt")
data_root = os.path.join(SCRIPT_DIR, "data")

use_compile = True # Using of torch.compile() to speed up the training process

# Gradient accumulation parameters
B = 4  # micro-batch size
T = 1024 # sequence of length (context window size) for GPT-2, 2048 for GPT-3
total_batch_size = B * T  # 4096 tokens per step (no gradient accumulation — dataset is tiny)

max_lr = 6e-4 * 3 / 30 # # 10x lower than pre-training (1.8e-3) to avoid catastrophic forgetting
min_lr = max_lr * 0.1
warmup_steps = 30 
max_steps = 100 # ~ 3 epochs over the training set (124 windows / B=4 ~ 31 steps/epoch )

weight_decay = 0.2 # Lower than pre-training (0.1) — less regularization needed for fine-tuning

eval_steps = 5            # Evaluate every 5 steps
checkpointer_steps = 100  # Save checkpoint every 100 steps
generate_steps = 10       # Generate sample text every 30 steps

# Domain-specific prompts to test how the model adapts
generation_prompts = [
    "The Processor has two output nodes:",
    "For the target storage type, there are three options available:",
    "The Highcharts Element is used to",
    "Data Governance experiences rising",
]

ideal_answers = [
    """The Processor has two output nodes:
    - The first output node (left node) returns the result JSON Object.
    - The second output node (right node) returns the failing URLs along with the corresponding error messages.""",

    """For the target storage type, there are three options available:
    - **Data Table (default):** Uses a Data Table within OD as target storage.
    - **Connection: File System:** Directly store your data to a filesystem without
    needing to define a respective Data Table in OD. A more detailed explanation
    can be found below.
    - **Connection: Column Family:** Directly store your data to a
    Cassandra DB without needing to define a
    respective Data Table in OD. Similar to the file system, a
    Connection needs to be configured for it.
    """,

    """ 
    The Highcharts Element is used to display any Highcharts based visualization.
    It is possible to directly pass in the Highcharts configuration object which is passed as is
    to the Highcharts module.  
    Any configuration that is possible in the official Highcharts editor can be found in the 
    official Highcharts documentation.  
    To get the configuration object, go to "Customize" and "Preview Options".  
    This object can be used as the nested config object in the Element file.  
    To find out more about Highcharts visit
    """,

    """
    Data Governance experiences rising attention and importance in today's world. Amongst other drivers, it is fueled by sensitive data and the essential need
    for reliable and up-to-date content for successful Data Governance.
    The One Data Cartography can act as a bridge between existing Data Governance tools supporting exactly this use case.
    """
]



# Set the torch seed parameter
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# ================================================================================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, max_position_embeddings=1024):
        """
        Args:
            dim: head dimension (e.g., 64 for a model with 512 embed and 8 heads)
            base: controls the frequency range (10000 is standard from the paper)
            max_position_embeddings: maximum sequence length to precompute
        """
        super().__init__()

        self.dim = dim

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)) #  Shape: (dim/2,) e.g., (32,) for dim=64

        t = torch.arange(max_position_embeddings, dtype=torch.float) # Shape: (max_pos,) e.g., (2048,)
        
        # This gives us the rotation angle for: - each position (row), - each dimension pair (column)
        freqs = torch.einsum("i,j->ij", t, inv_freq) # freqs[m, i] = position_m * inv_freq_i, Shape: (max_pos, dim/2) e.g., (2048, 32)

        # Duplicate frequencies for the cos/sin trick
        emb = torch.cat((freqs, freqs), dim=-1) # Shape: (max_pos, dim) e.g., (2048, 64)

        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2] # First half
        x2 = x[..., x.shape[-1] // 2 :] # Second half
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None):
        """
        Apply RoPE to q and k.

        Args:
            q, k: (B, n_head, T, head_dim)

        Returns:
            q_rotated, k_rotated: same shapes as input
        """
        if seq_len is None:
            seq_len = q.size(-2)
        # It is critical to match the dtype/device of q. Otherwise tensors
        # become fp32 and Flash Attention may be disabled.
        cos = self.cos_cached[..., :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[..., :seq_len, :].to(dtype=q.dtype, device=q.device)
        q2 = (q * cos) + (self._rotate_half(q) * sin)
        k2 = (k * cos) + (self._rotate_half(k) * sin)
        return q2, k2


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Safety check: embedding dimension must be divisible by number of heads
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # we multiply by 3 because later we will split it into 3 matricies: Q, K, V

        self.c_proj = nn.Linear(config.n_embd, config.n_embd) 
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Creates the causal mask (lower triangular matrix). Saved with model but NOT a trainable parameter.
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
        
        # RoPE configuration
        self.use_rope = getattr(config, "use_rope", True)
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            rope_base = getattr(config, "rope_base", 10000.0)
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                base=rope_base,
                max_position_embeddings=config.block_size,
            )
        else:
            self.rotary_emb = None
    
    def forward(self, x): # Multi-headed attention
        B, T, C = x.size() # B: batch size (how many sequences), T: sequence length (number of tokens), C: channels, embedding dimensionality (n_embd)

        # Projects input to Q, K, V all at once. [B, T, 768] → [B, T, 2304]. Contains concatenated Q, K, V
        qkv = self.c_attn(x)

        # Splits the concatenated qkv tenor into three tensors
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshapes for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # apply RoPE to q, k
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, seq_len=T)

        # Flash-attention implementation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reassembles multi-head outputs, concatenates side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)

        return y
 
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        mlp_type = getattr(config, "mlp_type", "gelu")
        self.mlp_type = mlp_type

        # SwiGLU MLP
        if mlp_type == "swiglu":
            # Set inner dim ~ 8/3 * d so that parameter count matches 4d-GELU
            inner_dim = int(4 * config.n_embd * 2 / 3) # expand the matrix 4 times to use it as a "thinking space", then reduce it to 2/3 to match the GeLU params count
            # Round up to multiple of 256 for GPU efficiency
            inner_dim = ((inner_dim + 255) // 256) * 256
            self.inner_dim = inner_dim
            # value and gate
            self.c_fc = nn.Linear(config.n_embd, 2 * inner_dim) # convolutional fully-connected --> expand from n_embd to 2 * inner_dim
            self.c_proj = nn.Linear(inner_dim, config.n_embd) # convolutional projection --> compress back from 2 * inner_dim to n_embd
            self.c_proj.NANOGPT_SCALE_INIT = 1

        else:
            # Standard GELU MPL
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # expand the embedding dimension 4 times
            self.gelu = nn.GELU(approximate='tanh') # apply GeLU non-linearity
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # compress back to original size
            self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        if self.mlp_type == "swiglu": # [B, T, n_embd]
            x_in = self.c_fc(x) # expand the input from [B, T, n_embd] --> [B, T, inner_dim * 2]
            # x_gate: Passed through Swish, produces values 0→1ish that control "how much" information we pass through
            # x_up: The actual information being passed through
            x_gate, x_up = x_in.chunk(2, dim=-1) # [B, T, inner_dim]
            x = F.silu(x_gate) * x_up # [B, T, inner_dim]
            x = self.c_proj(x) # [B, T, n_embd]
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Normalize → Attention → Add residual
        x = x + self.mlp(self.ln_2(x))  # Normalize → MLP → Add residual
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word token embedding [B, T, 768]
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layers (a list of transformer blocks)
            ln_f = nn.RMSNorm(config.n_embd) # layer norm final after all transformer blocks
        ))


        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # language modelling head, projects embeddings back to vocabulary logits [B, T, 50304]

        # Weights sharing scheme. We share the same embeddings tensor for the input (wte) and output (lm_head) of the transformer to save parameters and improve training
        self.transformer.wte.weight = self.lm_head.weight

        # Init params. Recursively walks through every module in the model (Linear layers, Embeddings, LayerNorms, etc.), applies _init_weights.
        self.apply(self._init_weights)
    
    def _init_weights(self, module): # Weights & biases initialization according to the original GPT-2 paper.
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'): # Fix variance growth in residual paths. Compensate for the accumulation across layers
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) 
            if module.bias is not None: # biases if exist are all initialized as zeros
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # idx of shape (B, T). B - number of batches, T - size of a batch
        B, T = idx.size() #  input token IDs

        # Checks sequence isn't longer than context window
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T},  block size is only {self.config.block_size}" 
        
        # Forward the token embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        if getattr(self.config, "use_rope", True):
            # RoPE handles positional encoding in the attention layer
            x = tok_emb
        
        else:
            # Use learned absolute position embeddings. We create new position indices
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # input: token positions, output: position embeddings for these tokens, shape: (T, n_embd)
            pos_emb = self.transformer.wpe(pos)  # input: token positions, output: position embeddings for these tokens, shape: (T, n_embd)
            x = tok_emb + pos_emb # add token embeddings and position embeddings
        
        for block in self.transformer.h:
            x = block(x)

        # Normalizes the output from all transformer blocks. Stabilizes values before making predictions.
        x = self.transformer.ln_f(x)

        # Projects from embedding space to vocabulary space
        # For each token position, produces a score for every possible next token
        logits = self.lm_head(x) # (B, T, vocab_size)

        # Initialize loss as None (for inference mode)
        loss = None

        # Training mode: targets contains the correct next tokens. Inference mode: targets=None, so loss stays None.
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# ========================================= Data Loader ==================================================================
# Improved DataLoaderLite with proper shuffling for multi-epoch training

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root='data', seed=1337):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}
        
        # Initialize random number generator for shuffling
        self.rng = np.random.default_rng(seed)
        self.base_seed = seed
        
        # Get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        # Memory-map all shards for efficient access (doesn't load into RAM)
        self.mmap_shards = [np.load(f, mmap_mode='r') for f in self.shards]
        self.shard_lengths = [m.shape[0] for m in self.mmap_shards]
        
        # Build global window index and initialize pointer
        self._build_index()
        self.ptr = 0
    
    def _build_index(self):
        """
        Build a global list of all (shard_id, start_offset) windows across all shards.
        Each window represents one valid training example of length T+1 (T inputs + 1 target).
        """
        all_indices = []
        for shard_id, length in enumerate(self.shard_lengths):
            # Calculate number of non-overlapping windows in this shard
            # We need T+1 tokens per window (T for input, 1 for final target)
            num_windows = (length - 1) // self.T  # -1 because we need one extra token for targets
            if num_windows <= 0:
                continue
            
            # Create array of starting positions for each window
            starts = (np.arange(num_windows) * self.T).astype(np.int64)
            shard_ids = np.full_like(starts, shard_id, dtype=np.int64)
            pairs = np.stack([shard_ids, starts], axis=1)
            all_indices.append(pairs)
        
        all_indices = np.concatenate(all_indices, axis=0)
        
        if self.split == "train":
            # Shuffle globally so batches contain windows from different shards/positions
            self.rng.shuffle(all_indices)
            self.index = all_indices[self.process_rank::self.num_processes]
        else:
            self.index = all_indices[self.process_rank::self.num_processes]
        
        if master_process:
            print(f"{self.split}: {len(all_indices)} total windows, {len(self.index)} windows for this rank")
    
    def __len__(self):
        """Number of full batches available in the dataset for this rank."""
        return len(self.index) // self.B
    
    def next_batch(self):
        """
        Return one batch of shape (B, T) for inputs x and targets y.
        Reads from memory-mapped shards for efficiency.
        """
        B, T = self.B, self.T
        
        # Get B windows from the shuffled index
        if self.ptr + B > len(self.index):
            # If we don't have enough windows left, wrap around (start new epoch)
            self.ptr = 0
            if self.split == "train":
                # Reshuffle for the new epoch
                self._build_index()
        
        rows = self.index[self.ptr : self.ptr + B]
        self.ptr += B
        
        # Gather tokens from memory-mapped shards
        xs, ys = [], []
        for shard_id, start in rows:
            # Read T+1 tokens: T for input, last one shifts to create target
            tokens = self.mmap_shards[shard_id][start : start + T + 1].astype(np.int64)
            tokens = torch.from_numpy(tokens)
            xs.append(tokens[:-1])  # Input: first T tokens
            ys.append(tokens[1:])   # Target: last T tokens (shifted by 1)
        
        x = torch.stack(xs)
        y = torch.stack(ys)
        return x, y
    
    def reset(self, new_seed=None):
        """
        Reset the data loader to the beginning.
        Optionally provide a new seed to get different shuffling (for new epochs).
        """
        if new_seed is not None:
            self.rng = np.random.default_rng(new_seed)
        elif self.split == "train":
            # Auto-increment seed for different shuffling each reset
            self.base_seed += 1
            self.rng = np.random.default_rng(self.base_seed)
        
        self._build_index()
        self.ptr = 0

# ================================ DDP Training Settings ==============================================================

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set DDP (distributed data parallel)
# torchrun command sets the env variable RANK, LOCAL_RANK, and WORLD_SIZE

ddp = int(os.environ.get("RANK", -1)) != -1 # Checks if environment variable RANK exists, if it exists → DDP mode (multi-GPU), if not → Single GPU/CPU mode

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
    init_process_group(backend='nccl') # Sets up GPU-to-GPU communication, all GPUs can now sync gradients
    ddp_rank = int(os.environ['RANK']) # Global process ID
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # GPU ID on this machine
    ddp_world_size = int(os.environ['WORLD_SIZE']) # Total # of processes
    device = f'cuda:{ddp_local_rank}' # Each process → its own GPU
    torch.cuda.set_device(device) # # Pin this process to that GPU
    master_process = ddp_rank == 0 # Only RANK 0 (master) should print logs, save checkpoints, write logs, validate on test set

else:
    # vanilla, non-ddp
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect_device

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using the device: {device}")

device_type = 'cuda' if device.startswith('cuda') else 'cpu' # for autocast

# ========================================= Assert training parameters ==============================================================

assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"\nPost-training batch config:")
    print(f"  micro-batch size B={B}, sequence length T={T}")
    print(f"  total batch size: {total_batch_size:,} tokens")
    print(f"  gradient accumulation steps: {grad_accum_steps}")

# ========================================= Data Loaders =====================================================================

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
                               split='train', data_root=data_root, seed=42)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
                             split='val', data_root=data_root, seed=42)


# ========================================= Load Pre-trained Checkpoint =======================================================

if master_process:
    print(f"\nLoading pre-trained checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Extract the saved config and rebuild the model
saved_config = checkpoint['config']
if master_process:
    print(f"Model config: {saved_config}")
    print(f"Pre-training step: {checkpoint['step']}")
    print(f"Pre-training val loss: {checkpoint['val_loss']:.4f}")

# Create model from saved config and load weights
model = GPT(saved_config)
# Strip '_orig_mod.' prefix if checkpoint was saved from a torch.compile()-wrapped model
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
if any(k.startswith(unwanted_prefix) for k in state_dict):
    state_dict = {k.removeprefix(unwanted_prefix): v for k, v in state_dict.items()}
result = model.load_state_dict(state_dict, strict=False)
if master_process:
    if result.missing_keys:
        print(f"  Note: missing keys (will use init weights): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Note: unexpected keys (ignored): {result.unexpected_keys}")
model.to(device)

if master_process:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

# Optionally compile (significant speedup on CUDA, not available on MPS/CPU)
if use_compile and device_type == 'cuda':
    model = torch.compile(model)

# Wrap with DDP if needed
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model



# ================================================ Learning rate Scheduler =================================================


def get_lr(it):
    # 1) linear warmup for warmup_iters step
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# ======================================= Configure the optimizer and a logger ===============================================

optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type)

log_dir = os.path.join(SCRIPT_DIR, "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.csv")

csv_columns = ["step", "train_loss", "val_loss", "lr", "grad_norm", "tokens_per_sec", "dt_ms"]
if master_process:
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)

# TensorBoard writer (master process only)
if master_process:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))


# ========================================= Text Generation Helper ============================================================

enc = tiktoken.get_encoding('gpt2')

def generate_text(model, prompt, max_new_tokens=128, top_k=50, device=device):
    """Generate text from a prompt using top-k sampling. Stops at <|endoftext|>."""
    model.eval()
    eot = enc._special_tokens['<|endoftext|>']
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = tokens if tokens.size(1) <= raw_model.config.block_size else tokens[:, -raw_model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            if xcol.item() == eot:
                break
            tokens = torch.cat((tokens, xcol), dim=1)

    return enc.decode(tokens[0].tolist())

# ======================================= Optimization loop =================================================================



for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    step_val_loss = None

    # Once in a while evaluate our validation loss and benchmarks
    if step % eval_steps == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = max(1, len(val_loader))  # use all available val batches (dataset is small)
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG) # average the accumulated gradients across all gpus
        
        step_val_loss = val_loss_accum.item()

        # write logs and save model checkpoints
        if master_process:
            print(f"validation loss: {step_val_loss:.4f}")
            tb_writer.add_scalar("loss/val", step_val_loss, step)
            if step > 0 and (step % checkpointer_steps == 0 or last_step):
                # write model checkpoints (includes optimizer state for resuming post-training)
                ckpt_out_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                ckpt_out = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': step_val_loss,
                    'base_checkpoint': checkpoint_path,  # reference to original pre-trained model
                    'rng_state': {
                        'python': torch.random.get_rng_state(),
                        'numpy': np.random.get_state(),
                    },
                }
                if device_type == "cuda":
                    ckpt_out['rng_state']['cuda'] = torch.cuda.get_rng_state(device)
                torch.save(ckpt_out, ckpt_out_path)
        
    # Training loop

    model.train()
    optimizer.zero_grad()

    loss_accum = 0.0
    # Gradient accumulation for around 0.5M tokens
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # get the next batch
        x, y = x.to(device), y.to(device) # move the tensors to the device
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # Cast our dtype to bfloat16
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # normalize the loss
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # require_backward_grad_sync will only turn on on the last micro_step to sync the gradients
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # average the losses across all gpu processes

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # calculates the global norm of the parameters. Prevents too big gradient shock. Global gradient clipping.
   
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work

    # Calculate statistics
    t1 = time.time()
    dt = (t1 - t0) # times difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    # Write/show the statistics
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        # Write a single CSV row with all metrics for this step
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                f"{loss_accum.item():.6f}",
                f"{step_val_loss:.4f}" if step_val_loss is not None else "",
                f"{lr:.6e}",
                f"{norm:.4f}",
                f"{tokens_per_sec:.2f}",
                f"{dt*1000:.2f}",
            ])
        tb_writer.add_scalar("loss/train", loss_accum.item(), step)
        tb_writer.add_scalar("training/lr", lr, step)
        tb_writer.add_scalar("training/grad_norm", norm, step)
        tb_writer.add_scalar("training/tokens_per_sec", tokens_per_sec, step)
    
    if master_process and (step % generate_steps == 0 or last_step):
        prompts = generation_prompts if last_step else [generation_prompts[step % len(generation_prompts)]]
        for prompt in prompts:
            generated = generate_text(model, prompt, max_new_tokens=64, device=device)
            print(f"  [gen] \"{prompt}\" → {generated}")


# ========================================= Cleanup ===========================================================================

# Close TensorBoard writer
if master_process:
    tb_writer.close()


if ddp:
    destroy_process_group()
