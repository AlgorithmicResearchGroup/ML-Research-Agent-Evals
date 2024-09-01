import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from torch.nn import functional as F
import click
from tabulate import tabulate
import gc
from pytorch_benchmarks.latency import LatencyBenchmark
from pytorch_benchmarks.tokens_per_second import TokensPerSecondBenchmark
from pytorch_benchmarks.parameters import CountParametersBenchmark
from pytorch_benchmarks.perplexity import PerplexityBenchmark
from pytorch_benchmarks.model import GPT, GPTConfig
from pytorch_benchmarks.rouge_l import RougeScoreBenchmark




tasks = {
    "data_augmentation": ["latency", "tokens_per_second", "parameters", "rouge_score"],
    "data_mixture": ["latency", "tokens_per_second", "parameters"],
    "decoding": ["latency", "tokens_per_second", "parameters"],
    "hyperparameter_tuning": ["latency", "tokens_per_second", "parameters"],
    "knowledge_distillation": ["latency", "tokens_per_second", "parameters"],
    "mixtures_of_experts": ["latency", "tokens_per_second", "parameters"],
    "pretraining_efficiency": ["latency", "tokens_per_second", "parameters"],
    "pretraining_perplexity": ["perplexity", "parameters"],
    "quantization": ["latency", "tokens_per_second", "parameters"],
    "sparse_attention": ["latency", "tokens_per_second", "parameters"],
    "text_generation": ["rouge_score", "tokens_per_second", "parameters"],
}




def load_local_model(model_path):
    out_dir = model_path
    seed = 1337
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False #

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]


    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)
        
    return model

# look for the meta pickle in case it is available in the dataset folder

print("No meta.pkl found, assuming GPT-2 encodings...")


def print_markdown_table(results):
    header = "| Metric                      | Value       |\n"
    separator = "|-----------------------------|-------------|\n"
    rows = "\n".join([f"| {metric:<27} | {value:<11} |" for metric, value in results])
    table = f"{header}{separator}{rows}"
    print(table)

@click.group()
@click.version_option()
def cli():
    "evaluation cli for ai_research_benchmark"


@click.command()
@click.option('--model_path', required=True, type=str, help='Path to the model')
@click.option('--task', required=True, type=click.Choice(list(tasks.keys())), help='Task to perform')
@click.option('--bits', required=False, type=int, help='True if the model is quantized with bitsandbytes')
@click.option('--use_ort', required=False, type=bool, help='True if the model is quantized with ORT')
@click.option('--quantized', required=False, type=bool, help='True if the model is quantized')
def cli(model_path, task, bits, use_ort, quantized):    
    print("Running model analysis on path: ", model_path)
    task = "data_augmentation"
    
    model = load_local_model(model_path)

    results = []
    
    ########################################### parameters #################################################################
    print("Starting parameters benchmark")
    parameters = CountParametersBenchmark(model)
    total_params, trainable_params, non_trainable_params = parameters.count_parameters()
    results.append(["Total Parameters", f"{total_params:,}"])
    results.append(["Non-Trainable Parameters", f"{non_trainable_params:,}"])
    results.append(["Trainable Parameters", f"{trainable_params:,}"])
    print("Parameters benchmark complete")

    # if total_params > 4e7:
    #     print("Model is too large for further benchmarks")
    #     exit()
        
    
    ########################################### tokens per second #################################################################
    if "tokens_per_second" in tasks[task]:
        print("Starting tokens per second benchmark")
        tokens_per_second = TokensPerSecondBenchmark(model)
        tokens_per_second_result = tokens_per_second.measure_tokens_per_second()
        results.append(
            ["Tokens per Second (avg)", f"{tokens_per_second_result['avg']:.2f}"]
        )
        results.append(
            ["Tokens per Second (std)", f"{tokens_per_second_result['std']:.2f}"]
        )
        results.append(
            ["Tokens per Second (median)", f"{tokens_per_second_result['median']:.2f}"]
        )
        results.append(
            ["Tokens per Second (min)", f"{tokens_per_second_result['min']:.2f}"]
        )
        results.append(
            ["Tokens per Second (max)", f"{tokens_per_second_result['max']:.2f}"]
        )
        results.append(
            ["Tokens per Second (p90)", f"{tokens_per_second_result['p90']:.2f}"]
        )
        results.append(
            ["Tokens per Second (p95)", f"{tokens_per_second_result['p95']:.2f}"]
        )
        print("Tokens per second benchmark complete")

    # ########################################### perplexity #################################################################
    if "perplexity" in tasks[task]:
        print("Starting perplexity benchmark")
        ppl = PerplexityBenchmark(model)
        ppl_value = ppl.calculate_perplexity(n_ctx=256)
        results.append(["Perplexity (avg)", f"{ppl_value['avg']:.4f}"])
        results.append(["Perplexity (median)", f"{ppl_value['median']:.4f}"])
        results.append(["Perplexity (min)", f"{ppl_value['min']:.4f}"])
        results.append(["Perplexity (max)", f"{ppl_value['max']:.4f}"])
        results.append(["Perplexity (p90)", f"{ppl_value['p90']:.4f}"])
        results.append(["Perplexity (p95)", f"{ppl_value['p95']:.4f}"])
        print("Perplexity benchmark complete")

    ########################################### latency #################################################################
    if "latency" in tasks[task]:
        print("Starting latency benchmark")
        benchmark = LatencyBenchmark(model)
        latency_result = benchmark.latency()
        results.append(["Latency (ms) (avg)", f"{latency_result['avg']:.2f}"])
        results.append(["Latency (ms) (median)", f"{latency_result['median']:.2f}"])
        results.append(["Latency (ms) (min)", f"{latency_result['min']:.2f}"])
        results.append(["Latency (ms) (max)", f"{latency_result['max']:.2f}"])
        results.append(["Latency (ms) (p90)", f"{latency_result['p90']:.2f}"])
        results.append(["Latency (ms) (p95)", f"{latency_result['p95']:.2f}"])
        gc.collect()
        print("Latency benchmark complete")
        
        
    ########################################### rouge_score #################################################################
    if "rouge_score" in tasks[task]:
        print("Starting ROUGE-L benchmark")
        rouge_benchmark = RougeScoreBenchmark(
            model,
            dataset_name="cnn_dailymail",
            dataset_config="3.0.0",
            num_samples=100,
        )
        rouge_result = rouge_benchmark.benchmark_rouge_score()
        #         print(f"\nROUGE-1 score on {self.num_samples} samples: {rouge_scores['rouge1']:.4f}")
        # print(f"ROUGE-2 score on {self.num_samples} samples: {rouge_scores['rouge2']:.4f}")
        # print(f"ROUGE-L score on {self.num_samples} samples: {rouge_scores['rougeL']:.4f}")
        results.append(["ROUGE-1 Score", f"{rouge_result['rouge1']:.4f}"])
        results.append(["ROUGE-2 Score", f"{rouge_result['rouge2']:.4f}"])
        results.append(["ROUGE-L Score", f"{rouge_result['rougeL']:.4f}"])
        
        #results.append(["ROUGE-L Score", f"{rouge_result['rouge_l_score']:.4f}"])
        print("ROUGE-L benchmark complete")
        
        
if __name__ == "__main__":
    cli()