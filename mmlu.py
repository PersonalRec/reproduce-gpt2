"""
Downloads and evaluates MMLU (Massive Multitask Language Understanding) in Python.
https://github.com/hendrycks/test

Uses teacher-forcing (completion style) evaluation, matching the HellaSwag pattern:
for each question, we feed context + each of the 4 answer completions through the model,
compute loss over the completion region, and pick the answer with the lowest loss.

MMLU has 57 subjects across STEM, humanities, social sciences, and other domains.
For small models (124M), we evaluate on a curated subset of ~12 subjects where
performance can meaningfully exceed random chance (25%).

The test split has ~14,042 questions total across all 57 subjects.

gpt2 (124M)
- eleuther harness reports ~26% (near random on most subjects)
- curated subjects can show signal above random

Dataset format (CSV, no header):
  column 0: question text
  columns 1-4: four answer choices
  column 5: correct answer letter (A, B, C, or D)
"""

import os
import csv
import tarfile
import requests
import tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "mmlu")

enc = tiktoken.get_encoding("gpt2")

choices = ["A", "B", "C", "D"]

# Curated subjects where 124M models can score above random (~25%).
# These tend to be less technical / more language-pattern-based.
CURATED_SUBJECTS = [
    "high_school_psychology",
    "high_school_us_history",
    "high_school_world_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "management",
    "marketing",
    "miscellaneous",
    "sociology",
    "public_relations",
    "human_aging",
    "nutrition",
]

# Mapping from answer letter to index
ANSWER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads and extracts the MMLU dataset to DATA_CACHE_DIR."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    # Check if data already exists by looking for the test directory
    test_dir = os.path.join(DATA_CACHE_DIR, "test")
    if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        return  # already downloaded

    tar_path = os.path.join(DATA_CACHE_DIR, "data.tar")
    data_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

    if not os.path.exists(tar_path):
        print(f"Downloading MMLU dataset from {data_url}...")
        download_file(data_url, tar_path)

    print("Extracting MMLU dataset...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(DATA_CACHE_DIR)

    # The tar extracts to DATA_CACHE_DIR/data/{dev,val,test}/*.csv
    # Move contents up one level so we have DATA_CACHE_DIR/{dev,val,test}/
    extracted_dir = os.path.join(DATA_CACHE_DIR, "data")
    if os.path.exists(extracted_dir):
        for item in os.listdir(extracted_dir):
            src = os.path.join(extracted_dir, item)
            dst = os.path.join(DATA_CACHE_DIR, item)
            if not os.path.exists(dst):
                os.rename(src, dst)
        # Clean up
        if os.path.exists(extracted_dir):
            import shutil
            shutil.rmtree(extracted_dir)

    # Remove the tar file to save space
    if os.path.exists(tar_path):
        os.remove(tar_path)

    print("MMLU dataset ready.")


def format_subject(subject):
    """Converts subject slug to readable name: 'high_school_psychology' -> ' High School Psychology'"""
    return " " + " ".join(word.capitalize() for word in subject.split("_"))


def render_example(subject, row):
    """
    Given a subject and a CSV row, render it as torch tensors matching the HellaSwag interface:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)

    Uses teacher-forcing style: each of the 4 rows contains the full prompt + one answer completion.
    The completion region includes the answer letter and the full answer text, e.g., " A. answer_text".
    """
    question = row[0]
    answer_choices = row[1:5]
    correct_letter = row[5].strip()
    label = ANSWER_TO_IDX[correct_letter]

    # Build the prompt (zero-shot)
    ctx = (
        f"The following are multiple choice questions (with answers) about{format_subject(subject)}.\n\n"
        f"{question}\n"
    )
    for i, choice_text in enumerate(answer_choices):
        ctx += f"{choices[i]}. {choice_text}\n"
    ctx += "Answer:"

    # Tokenize context
    ctx_tokens = enc.encode(ctx)

    # Build 4 completions: " A. answer_text", " B. answer_text", etc.
    tok_rows = []
    mask_rows = []
    for i, choice_text in enumerate(answer_choices):
        end_text = f" {choices[i]}. {choice_text}"
        end_tokens = enc.encode(end_text)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    # Collate into tensors (rows may differ in length)
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label


def iterate_examples(split, subjects=None):
    """
    Yields (subject, row) pairs for the given split across the specified subjects.

    Args:
        split: "test", "val", or "dev"
        subjects: list of subject names, or None to use CURATED_SUBJECTS
    """
    download()

    if subjects is None:
        subjects = CURATED_SUBJECTS

    split_dir = os.path.join(DATA_CACHE_DIR, split)
    for subject in subjects:
        csv_path = os.path.join(split_dir, f"{subject}_{split}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping {subject}")
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:  # question + 4 choices + answer
                    yield subject, row


@torch.no_grad()
def evaluate(model_type, device):
    """Standalone evaluation using a HuggingFace GPT-2 model (for testing)."""
    from transformers import GPT2LMHeadModel

    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    num_correct = 0
    num_total = 0

    for subject, row in iterate_examples("test"):
        tokens, mask, label = render_example(subject, row)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get logits
        logits = model(tokens).logits

        # Evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # Get the average loss just for the completion region (where mask == 1)
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # Pick the completion with the lowest average loss
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred_norm == label)

        if num_total % 50 == 0:
            print(f"MMLU {num_total} acc={num_correct / num_total:.4f}")

    acc = num_correct / num_total
    print(f"\nMMLU accuracy: {num_correct}/{num_total} = {acc:.4f}")
    return acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2",
                        help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="the device to use")
    parser.add_argument("--subjects", type=str, default=None,
                        help="comma-separated list of subjects, or 'all' for all 57")
    args = parser.parse_args()

    subjects = None
    if args.subjects == "all":
        # Discover all subjects from the test directory
        download()
        test_dir = os.path.join(DATA_CACHE_DIR, "test")
        subjects = sorted(
            f.replace("_test.csv", "")
            for f in os.listdir(test_dir)
            if f.endswith("_test.csv")
        )
        print(f"Evaluating all {len(subjects)} subjects")
    elif args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]

    evaluate(args.model_type, args.device)
