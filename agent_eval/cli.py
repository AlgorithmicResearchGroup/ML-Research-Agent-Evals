import time
import functools
import torch
import numpy as np
import gc
import click
from tabulate import tabulate
import matplotlib.pyplot as plt
from agent_eval.benchmarks.latency import LatencyBenchmark
from agent_eval.benchmarks.mmlu import MMLUBenchmark
from agent_eval.benchmarks.parameters import CountParametersBenchmark
from agent_eval.benchmarks.perplexity import PerplexityBenchmark
from agent_eval.benchmarks.rouge_l import RougeScoreBenchmark
from agent_eval.benchmarks.tokens_per_second import TokensPerSecondBenchmark
from agent_eval.benchmarks.final_score import FinalScore
from agent_eval.load_model import load_test_model


model_metrics = [
    {
        'model': 'x-small',
        'total_params': 30044544,
        'tokens_per_second': 242.70,
        'perplexity': 95.2161,
        'latency': 2.25,
        'rouge_l': 0.4803,
    },
    {
        'model': 'medium',
        'total_params': 354823168,
        'tokens_per_second': 29.11,
        'perplexity': 23.7864,
        'latency': 14.38,
        'rouge_l': 0.4819,   
    },
    {
        'model': 'large',
        'total_params': 774030080,
        'tokens_per_second': 13.51,
        'perplexity': 20.7318,
        'latency': 29.48,
        'rouge_l': 0.4837, 
    },
    {
        'model': 'x-large',
        'total_params': 1557611200,
        'tokens_per_second': 8.42,
        'perplexity': 18.7528,
        'latency': 45.58,
        'rouge_l': 0.4843,
    },
]

tasks = {
    "data_augmentation": ["latency", "tokens_per_second", "parameters", "perplexity", "rouge_score"],
    "data_mixture": ["latency", "tokens_per_second", "parameters"],
    "decoding": ["latency", "tokens_per_second", "parameters"],
    "hyperparameter_tuning": ["latency", "tokens_per_second", "parameters"],
    "knowledge_distillation": ["latency", "tokens_per_second", "parameters"],
    "mixture_of_experts": ["latency", "tokens_per_second", "parameters"],
    "pretraining_efficiency": ["latency", "tokens_per_second", "parameters"],
    "pretraining_perplexity": ["perplexity", "parameters"],
    "quantization": ["latency", "tokens_per_second", "parameters"],
    "sparse_attention": ["latency", "tokens_per_second", "parameters"],
    "text_generation": ["rouge_score", "tokens_per_second", "parameters"],
}


def plot_results(metrics, model_name, task):
    plt.figure(figsize=(10, 5))
    for metric, values in metrics.items():
        plt.plot(values, label=metric)
    plt.title(f'Benchmark Results for {model_name} - {task}')
    plt.xlabel('Benchmarks')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_{task}_benchmark_results.png')
    plt.show()



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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load a non-quantized model
    model, tokenizer = load_test_model(model_path, bits=bits, use_ort=use_ort, quantized=quantized)

    results = []
    rubric_results = []
    
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
        tokens_per_second = TokensPerSecondBenchmark(model, tokenizer)
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
        
        rubric_results.append(["tokens_per_second", tokens_per_second_result['avg']])

    ########################################### perplexity #################################################################
    if "perplexity" in tasks[task]:
        print("Starting perplexity benchmark")
        ppl = PerplexityBenchmark(model, tokenizer)
        ppl_value = ppl.calculate_perplexity(n_ctx=256)
        results.append(["Perplexity (avg)", f"{ppl_value['avg']:.4f}"])
        results.append(["Perplexity (median)", f"{ppl_value['median']:.4f}"])
        results.append(["Perplexity (min)", f"{ppl_value['min']:.4f}"])
        results.append(["Perplexity (max)", f"{ppl_value['max']:.4f}"])
        results.append(["Perplexity (p90)", f"{ppl_value['p90']:.4f}"])
        results.append(["Perplexity (p95)", f"{ppl_value['p95']:.4f}"])
        print("Perplexity benchmark complete")
        
        rubric_results.append(["perplexity", ppl_value['avg']])

    ########################################### latency #################################################################
    if "latency" in tasks[task]:
        print("Starting latency benchmark")
        benchmark = LatencyBenchmark(model, tokenizer)
        latency_result = benchmark.latency()
        results.append(["Latency (ms) (avg)", f"{latency_result['avg']:.2f}"])
        results.append(["Latency (ms) (median)", f"{latency_result['median']:.2f}"])
        results.append(["Latency (ms) (min)", f"{latency_result['min']:.2f}"])
        results.append(["Latency (ms) (max)", f"{latency_result['max']:.2f}"])
        results.append(["Latency (ms) (p90)", f"{latency_result['p90']:.2f}"])
        results.append(["Latency (ms) (p95)", f"{latency_result['p95']:.2f}"])
        gc.collect()
        print("Latency benchmark complete")
        
        rubric_results.append(["latency", latency_result['avg']])

    ########################################### rouge_score #################################################################
    if "rouge_score" in tasks[task]:
        print("Starting ROUGE-L benchmark")
        rouge_benchmark = RougeScoreBenchmark(
            model,
            tokenizer,
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
        
        rubric_results.append(["rouge_score", rouge_result['rougeL']])

    if "mmlu" in tasks[task]:
        print("Starting MMLU benchmark")
        benchmark_name = (
            "high_school_mathematics"  # Replace with the desired MMLU benchmark name
        )
        mmlu_benchmark = MMLUBenchmark(
            model, tokenizer, benchmark_name, num_samples=100
        )
        accuracy = mmlu_benchmark.benchmark()
        results.append(["MMLU Accuracy", f"{accuracy:.4f}"])
        print("MMLU benchmark complete")
        
        rubric_results.append(["mmlu", accuracy])

    print("\n")
    print("Benchmark Results:")
    print(tabulate(results, headers=["Metric", "Value"]))
    print("\n")
    # (self, results, num_tokens, time_in_seconds, task, model_averages):
    model_averages = model_metrics[1]
    calc_score = FinalScore(rubric_results, num_tokens=1000, time_in_seconds=1800, task=tasks[task], model_averages=model_averages).calculate_final_score()
    final_score = calc_score.calculate_final_score()
    print(final_score)
    print("\n") 