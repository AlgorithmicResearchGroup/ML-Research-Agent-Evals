import timeit
from argparse import ArgumentParser

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig


class TokensPerSecondBenchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = "Hello my name is Matt. I am getting in touch with you because i didn't get a response from you."

    def generate_text(self, encoded_input):
        with torch.no_grad():
            outputs = self.model.generate(
                encoded_input,
                max_length=200,
                temperature=0.8,
                top_k=300,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def measure_tokens_per_second(self, warmup_iterations=10, benchmark_iterations=30):
        encoded_input = self.tokenizer.encode(self.prompt, return_tensors="pt").to(
            self.model.device
        )
        tokens_per_second = []

        # Warmup iterations
        print("Starting warmup iterations...")
        for _ in range(warmup_iterations):
            _ = self.generate_text(encoded_input)

        # Benchmark iterations
        print("Starting benchmark iterations...")
        for _ in range(benchmark_iterations):
            start_time = timeit.default_timer()
            generated_text = self.generate_text(encoded_input)
            end_time = timeit.default_timer()

            elapsed_time = end_time - start_time
            num_tokens = len(self.tokenizer.encode(generated_text))
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

