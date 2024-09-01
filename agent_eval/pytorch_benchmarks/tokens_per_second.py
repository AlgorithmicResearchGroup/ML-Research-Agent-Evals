import timeit
from argparse import ArgumentParser
import tiktoken
import numpy as np
import torch
from pytorch_benchmarks.sample import Model_Generate
from tqdm import tqdm

class TokensPerSecondBenchmark:
    def __init__(self, model):
        self.model = model
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.sample = Model_Generate(self.model)
        self.max_new_tokens = 512
        self.temperature = 0.8
        self.top_k = 200
        self.prompt = "Hello my name is Matt. I am getting in touch with you because i didn't get a response from you."

    def measure_tokens_per_second(self, warmup_iterations=10, benchmark_iterations=30):
        tokens_per_second = []

        encode = lambda s: self.tokenizer.encode(s, allowed_special={"<|endoftext|>"})
        encoded_input = encode(self.prompt)

        # Warmup iterations
        print("Starting warmup iterations...")
        for _ in range(warmup_iterations):
            text, y, logits = self.sample.return_response(encoded_input)

        # Benchmark iterations
        print("Starting benchmark iterations...")
        for _ in tqdm(range(benchmark_iterations)):
            start_time = timeit.default_timer()
            text, y, logits = self.sample.return_response(encoded_input)
            end_time = timeit.default_timer()

            elapsed_time = end_time - start_time
            num_tokens = len(encode(text))
            tokens_per_second.append(num_tokens / elapsed_time)

        # Calculate statistics
        avg_tokens_per_second = np.mean(tokens_per_second)
        std_tokens_per_second = np.std(tokens_per_second)
        median_tokens_per_second = np.median(tokens_per_second)
        min_tokens_per_second = np.min(tokens_per_second)
        max_tokens_per_second = np.max(tokens_per_second)
        percentile_90 = np.percentile(tokens_per_second, 90)
        percentile_95 = np.percentile(tokens_per_second, 95)

        print(
            f"Average tokens/sec: {avg_tokens_per_second:.2f} ± {std_tokens_per_second:.2f}"
        )
        print(f"Median tokens/sec: {median_tokens_per_second:.2f}")
        print(f"Min tokens/sec: {min_tokens_per_second:.2f}")
        print(f"Max tokens/sec: {max_tokens_per_second:.2f}")
        print(f"90th percentile: {percentile_90:.2f}")
        print(f"95th percentile: {percentile_95:.2f}")

        return {
            "avg": avg_tokens_per_second,
            "std": std_tokens_per_second,
            "median": median_tokens_per_second,
            "min": min_tokens_per_second,
            "max": max_tokens_per_second,
            "p90": percentile_90,
            "p95": percentile_95,
        }