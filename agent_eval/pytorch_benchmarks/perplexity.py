import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
import tiktoken
import traceback
from pytorch_benchmarks.sample import Model_Generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerplexityBenchmark:
    def __init__(self, model, dataset_path="wikitext", dataset_name=None, split="test", text_column="text"):
        self.model = model
        self.sample = Model_Generate(self.model)
        self.encoder = tiktoken.get_encoding("gpt2")
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._text_column = text_column
        self._text = self._prepare_data()
        self.encode = lambda s: self.encoder.encode(s, allowed_special={"<|endoftext|>"})

    def _prepare_data(self):
        if self._dataset_path == "wikitext":
            self._dataset_name = "wikitext-2-raw-v1"
        data = load_dataset(self._dataset_path, self._dataset_name, split=self._split)
        text_list = [" \n" if s == "" else s for s in data[self._text_column]]
        return "".join(text_list)

    @staticmethod
    def softmax(logits):
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0)

    def calculate_perplexity(self, n_ctx=512):
        tokens = torch.tensor(self.encode(self._text), dtype=torch.long, device=device).unsqueeze(0)
        nll = 0.0
        count = 0
        all_perplexity = []

        with tqdm(range(len(tokens[0]) // n_ctx), desc="Perplexity: - ") as progress:
            for i in progress:
                try:
                    nll, count = self._process_batch(i, n_ctx, tokens, nll, count)
                    if count > 0:  # Ensure we don't divide by zero
                        count = count + 1
                        print(f"!!!!!! NLL is {nll}")
                        print(f"!!!!!! COUNT is {count}")
                        curr_ppl = np.exp(nll / count)
                        print(f"!!!!!! CURR_PPL is {curr_ppl}")
                        all_perplexity.append(curr_ppl)
                        progress.set_description(f"Perplexity: {curr_ppl:.4f}")
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"Error processing batch {i}: {e}\n{tb}")
                    continue

        if len(all_perplexity) == 0:
            raise ValueError("No valid perplexity calculations were performed.")

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

    def _process_batch(self, i, n_ctx, tokens, nll, count):
        # Calculate the start and end indices for the current context window
        start = i * n_ctx
        end = start + n_ctx

        # Extract the tokens for the current context window
        context_tokens = tokens[:, start:end]

        # Replace the first token of the context with the end-of-text token
        context_tokens[0][0] = self.encode("<|endoftext|>")[0]

        block_size = self.sample.block_size  # Model's block size

        for block_start in range(0, n_ctx, block_size):
            block_end = min(block_start + block_size, n_ctx)
            block_tokens = context_tokens[:, block_start:block_end]
            print("size of block_tokens")
            print(block_tokens.size(1))
            print("block_tokens")
            print(block_tokens)

            try:
                # Compute the logits for the current block
                text, y, logits = self.sample.return_response(block_tokens[0])

                # Process logits to calculate negative log likelihood
                for k in range(block_start, block_end - 1):
                    # Get the logits for the current token position
                    tok_logits = logits[0][k - block_start].cpu()
                    
                    print("size of tok_logits")
                    print(tok_logits.size())
                    print("tok_logits")
                    print(tok_logits)
                    print(start + k + 1)

                    # Compute the probability of the next token
                    #prob = self.softmax(tok_logits)[tokens[0][start + k + 1]] # this is the line that is causing the error
                    prob = self.softmax(tok_logits)[tokens[0][start + k + 1]]
                    

                    # Update the negative log likelihood with the log probability of the next token
                    nll += -np.log(prob, where=prob > 0)

                    # Increment the count of processed tokens
                    count += 1
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Error processing block starting at {block_start} in batch {i}: {e}\n{tb}")
                continue

        # Return the updated negative log likelihood and count of processed tokens
        return nll, count