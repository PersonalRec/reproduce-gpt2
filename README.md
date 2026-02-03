# GPT-2 124M Reproduction

This project is a near complete reproduction of the original GPT-2 124M foundation model from OpenAI, trained from scratch on FineWeb-Edu 10BT dataset with some modern improvements:

## Modern Improvements Applied

* **AdamW optimizer** - More robust weight decay implementation
* **Flash Attention** - 2-3x faster than classical attention from the original GPT-2 paper
* **bfloat16 dtype** - Mixed precision training for better memory efficiency
* **Padded vocab size** (50304) - Better GPU utilization vs. original 50257
* **torch.compile** - Additional PyTorch optimization speedup

## Training Configuration

* **GPU:** 4x RTX 3090 (24 GB) rented on Vast.ai
* **Training time:** ~11 hours
* **Training cost:** ~$7 USD
* **Framework**: PyTorch with DDP (DistributedDataParallel)
* **Base implementation:** Andrej Karpathy's build-nanoGPT repo
* **Dataset:** FineWeb-Edu 10B tokens
* **Batch size:** 524,288 tokens per step (~0.5M)
* **Total steps:** 19,073 steps (1 epoch)

## Results Achieved

* **HellaSwag accuracy:** 0.28 (28%) vs. 0.294 (29.4%) original GPT-2
* **Validation loss:** 3.26 vs. ~3.29 original GPT-2

### Training & Validation Loss Curve

![Training and Validation Loss](results/230126/img/train-val_loss.png)

The plot shows stable training convergence over ~19K steps with the model closely matching the original GPT-2 performance despite using only 10B tokens (GPT-2 was trained on much more data).

## Detailed Documentation

For complete training parameters, architecture details, and hyperparameters, see:
- [`results/230126/training_params.md`](results/230126/training_params.md) - Full training configuration and results
- [`improvements_plan.md`](improvements_plan.md) - Future improvements to the project

## Project Structure

```
reproduce_gpt-2/
├── train_gpt2.py           # Main training script
├── fineweb.py              # Dataset preparation script
├── hellaswag.py            # HellaSwag evaluation utilities
├── show_results.ipynb      # Final results visualisation
├── improvements_plan.md    # Potential improvements analysis
├── results/
│   └── 230126/
│       ├── training_params.md   # Complete training parameters
│       └── img/
│           └── train-val_loss.png
```

## Reproduction Steps

1. **Prepare the dataset:**
   ```bash
   python fineweb.py
   ```

2. **Train the model (multi-GPU):**
   ```bash
   torchrun --standalone --nproc_per_node=4 train_gpt2.py
   ```

3. **Monitor training:**
   - Training/validation losses logged to `log/log.txt`
   - Model checkpoints saved every 5000 steps
   - HellaSwag evaluation every 250 steps


## Acknowledgments

Huge thanks to **Andrej Karpathy** for his excellent course and implementation:
- **YouTube Course:** [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- **Base Repository:** [build-nanogpt](https://github.com/karpathy/build-nanogpt)

Special thanks to **Josh Starmer (StatQuest)** for clear explanations of ML/DL concepts:
- **YouTube Channel:** [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)



