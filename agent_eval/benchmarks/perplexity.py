from argparse import ArgumentParser

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig


class PerplexityBenchmark:
    def __init__(
        self,
        model,
        tokenizer,
        dataset_path="wikitext",
        dataset_name=None,
        split="test",
        text_column="text",
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._text_column = text_column
        self._text = self._prepare_data()

    def _prepare_data(self):
        if self._dataset_path == "wikitext":
            self._dataset_name = "wikitext-2-raw-v1"

        # Load the dataset
        data = load_dataset(self._dataset_path, self._dataset_name, split=self._split)
        # Format the text column of the dataset
        text_list = [" \n" if s == "" else s for s in data[self._text_column]]
        return "".join(text_list)

    @staticmethod
    def softmax(logits):
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0)

    def calculate_perplexity(self, n_ctx=512, n_batch=512):
        tokens = self._tokenizer(
            self._text, truncation=False, return_tensors="pt"
        ).input_ids.to(self._model.device)

        nll = 0.0  # Negative log likelihood
        count = 0  # Counter for processed tokens
        all_perplexity = []

        with tqdm(range(len(tokens[0]) // n_ctx), desc="Perplexity: - ") as progress:
            for i in progress:
                try:
                    # Process each batch of tokens
                    nll, count = self._process_batch(
                        i, n_ctx, n_batch, tokens, nll, count
                    )

                    # Calculate and display the current perplexity
                    curr_ppl = np.exp(nll / count)
                    all_perplexity.append(curr_ppl)
                    progress.set_description(f"Perplexity: {curr_ppl:.4f}")
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    continue

        # Calculate statistics
        avg_perplexity = np.mean(all_perplexity)
        median_perplexity = np.median(all_perplexity)
        min_perplexity = np.min(all_perplexity)
        max_perplexity = np.max(all_perplexity)
        percentile_90 = np.percentile(all_perplexity, 90)
        percentile_95 = np.percentile(all_perplexity, 95)

        print(f"\nAverage perplexity: {avg_perplexity:.4f}")
        print(f"Median perplexity: {median_perplexity:.4f}")
        print(f"Min perplexity: {min_perplexity:.4f}")
        print(f"Max perplexity: {max_perplexity:.4f}")
        print(f"90th percentile: {percentile_90:.4f}")
        print(f"95th percentile: {percentile_95:.4f}")

        return {
            "avg": avg_perplexity,
            "median": median_perplexity,
            "min": min_perplexity,
            "max": max_perplexity,
            "p90": percentile_90,
            "p95": percentile_95,
        }

    def _process_batch(self, i, n_ctx, n_batch, tokens, nll, count):
        start = i * n_ctx
        end = start + n_ctx

        num_batches = (n_ctx + n_batch - 1) // n_batch

        logits = []

        for j in range(num_batches):
            batch_start = start + j * n_batch
            batch_size = min(end - batch_start, n_batch)

            token_org = tokens[0][batch_start].item()

            if j == 0:
                # Replace the first token with the BOS token
                tokens[0][batch_start] = self._tokenizer.bos_token_id

            # Compute the logits for the current batch of tokens
            batch_logits = self._compute_batch_logits(tokens, batch_start, batch_size)

            tokens[0][batch_start] = token_org

            logits.append(batch_logits)

        for j in range(min(512, n_ctx // 2), n_ctx - 1):
            tok_logits = logits[0][0][j].cpu().numpy()
            # Compute the probability of the next token
            prob = self.softmax(tok_logits)[tokens[0][start + j + 1]]

            # Update the negative log likelihood and the count of processed tokens
            nll += -np.log(prob, where=prob > 0)
            count += 1

        return nll, count

    def _compute_batch_logits(self, tokens, batch_start, batch_size):
        # Compute the logits without keeping track of gradients
        with torch.no_grad():
            outputs = self._model(tokens[:, batch_start : batch_start + batch_size])
        return outputs.logits.detach()
