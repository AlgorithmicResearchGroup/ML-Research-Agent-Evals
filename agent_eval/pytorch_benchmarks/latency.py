import gc
import time
from argparse import ArgumentParser
import numpy as np
import torch
import tiktoken
from tqdm.auto import tqdm
from pytorch_benchmarks.sample import Model_Generate


import gc
import time
from argparse import ArgumentParser
import numpy as np
import torch
from tqdm.auto import tqdm


class LatencyBenchmark:
    def __init__(self, model):
        self.model = model
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.sample = Model_Generate(self.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1
        self.prompt_length = 512
        self.nb_tokens = 512
        self.prompt = "Hello my name is Matt. I am getting in touch with you because i didn't get a response from you."

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

        # self.synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        memory = self.get_device_memory(self.device)
        if memory is not None:
            print(f"Device memory: {memory / (2 ** 30):.4f} GB")

        latencies = []

        encode = lambda s: self.tokenizer.encode(s, allowed_special={"<|endoftext|>"})
        encoded_input = encode(self.prompt)

        # Warmup iterations
        print("Starting warmup iterations...")
        for _ in range(warmup_iterations):
            text, y, logits = self.sample.return_response(encoded_input)

        print("Starting measurement iterations...")
        for _ in tqdm(range(measurement_iterations)):
            start_event = self.timing_event(self.device)
            end_event = self.timing_event(self.device)
            self.synchronize(self.device)
            start_event.record()

            text, y, logits = self.sample.return_response(encoded_input)
            end_event.record()
            self.synchronize(self.device)

            latency_ms = start_event.elapsed_time(end_event)
            latencies.append(latency_ms)

        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
            print(f"Peak memory during benchmark: {peak_memory / (2 ** 30):.4f} GB")

        # Calculate statistics
        avg_latency = np.mean(latencies) / self.nb_tokens
        median_latency = np.median(latencies) / self.nb_tokens
        min_latency = np.min(latencies) / self.nb_tokens
        max_latency = np.max(latencies) / self.nb_tokens
        percentile_90 = np.percentile(latencies, 90) / self.nb_tokens
        percentile_95 = np.percentile(latencies, 95) / self.nb_tokens

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