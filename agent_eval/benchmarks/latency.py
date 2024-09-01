import gc
import time
from argparse import ArgumentParser

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForCausalLM
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import GenerationConfig


class LatencyBenchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1
        self.prompt_length = 512
        self.nb_tokens = 512

    def synchronize(self, device):
        if device.type == "cuda":
            torch.cuda.synchronize()

    def timing_event(self, device):
        if device.type == "cuda":
            return torch.cuda.Event(enable_timing=True)

        class CPUEvent:
            def __init__(self):
                self.time = None

            def record(self):
                self.time = time.time()

            def elapsed_time(self, other):
                assert self.time is not None
                assert other.time is not None
                return (other.time - self.time) * 1000

        return CPUEvent()

    def get_device_memory(self, device):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            return torch.cuda.memory_allocated()
        return None

    def latency(self, warmup_iterations=5, measurement_iterations=10):
        generation_config = GenerationConfig(
            max_new_tokens=self.nb_tokens,
            min_new_tokens=self.nb_tokens,
            use_cache=False,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            do_sample=True,
            eos_token_id=None,  # This is required for min_new_tokens to actually have an effect.
        )
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.eos_token_id = None  # greedy_search falls back on this eos_token_id that we need to set to None as well for min_new_tokens to have an effect.

        # self.synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        memory = self.get_device_memory(self.device)
        if memory is not None:
            print(f"Device memory: {memory / (2 ** 30):.4f} GB")

        latencies = []
        # input_ids = torch.randint(1, model.config.vocab_size, size=(batch_size, prompt_length)).to(device)
        # masks = torch.ones(batch_size, prompt_length, dtype=torch.int32).to(device)
        prompt_text = "Hello my name is Matt. I am getting in touch with you because i didn't get a response from you."
        # Encode the prompt text
        encoded_input = self.tokenizer.encode(prompt_text, return_tensors="pt").to(
            self.model.device
        )

        # Warmup iterations
        print("Starting warmup iterations...")
        for _ in range(warmup_iterations):
            _ = self.model.generate(
                encoded_input,
                max_length=200,
                temperature=0.8,
                top_k=300,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        print("Starting measurement iterations...")
        for _ in tqdm(range(measurement_iterations)):
            start_event = self.timing_event(self.device)
            end_event = self.timing_event(self.device)
            self.synchronize(self.device)
            start_event.record()

            # _ = model.generate(input_ids, attention_mask=masks, generation_config=generation_config)
            _ = self.model.generate(
                encoded_input,
                max_length=200,
                temperature=0.8,
                top_k=300,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            end_event.record()
            self.synchronize(self.device)

            latency_ms = start_event.elapsed_time(end_event)
            latencies.append(latency_ms)

        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
            print(f"Peak memory during benchmark: {peak_memory / (2 ** 30):.4f} GB")

        # Calculate statistics
        avg_latency = np.mean(latencies) / generation_config.min_new_tokens
        median_latency = np.median(latencies) / generation_config.min_new_tokens
        min_latency = np.min(latencies) / generation_config.min_new_tokens
        max_latency = np.max(latencies) / generation_config.min_new_tokens
        percentile_90 = np.percentile(latencies, 90) / generation_config.min_new_tokens
        percentile_95 = np.percentile(latencies, 95) / generation_config.min_new_tokens

        print(f"\nAverage latency per token: {avg_latency:.4f} ms")
        print(f"Median latency per token: {median_latency:.4f} ms")
        print(f"Min latency per token: {min_latency:.4f} ms")
        print(f"Max latency per token: {max_latency:.4f} ms")
        print(f"90th percentile latency per token: {percentile_90:.4f} ms")
        print(f"95th percentile latency per token: {percentile_95:.4f} ms")

        return {
            "avg": avg_latency,
            "median": median_latency,
            "min": min_latency,
            "max": max_latency,
            "p90": percentile_90,
            "p95": percentile_95,
        }


