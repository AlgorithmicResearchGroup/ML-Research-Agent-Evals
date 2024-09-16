# Agent-Eval

Agent-Eval is a tool for evaluating the performance of an agent on a set of tasks. The tool calculates evaluation metrics such as perplexity, accuracy, and latency, and generates evaluation reports to help visualize the results. The evaluation process can be performed manually or automatically using the evaluation library.

Evaluations can either be performed manually or automatically. The manual evaluation process involves running the agent on each task and manually executing the evaluation script to evaluate its performance based on predefined metrics. The automatic evaluation process involves using a library to calculate evaluation metrics, generate evaluation reports, and visualize the results.

## Setup

```
pip install agent-eval 
```

## Usage

```
agent-eval --model_args <path_to_your_model> --task <task_name>
```


## Available Tasks

The following tasks are now available:

- llm_efficiency
- baby_lm
- mini_pile
- budget_model_training
- budget_model_inference
- llm_merging
- edge_llm_compression
- edge_llm_training
- math_reasoning

Each task also has a "mini" version prefixed with "mini_".

## Evaluation Process

For each task, the evaluation now consists of two parts:

1. Custom `agent_eval` metrics:
   - Latency
   - Tokens per second
   - Parameter count
   - Perplexity

2. Task-specific `lm-eval-harness` benchmarks:
   - For `llm_efficiency`: hellaswag
   - For `baby_lm`: hellaswag, gsm8k, arc, mmlu, blimp
   - For `mini_pile`: glue
   - For `llm_merging`, `edge_llm_compression`, `edge_llm_training`: mmlu
   - For `mini_math_reasoning`: mmlu_high_school_mathematics, mmlu_high_school_statistics, mmlu_high_school_computer_science

## Results

The evaluation will output results in two formats:

1. Custom `agent_eval` metrics will be displayed in a markdown table format in the console.
2. `lm-eval-harness` results will be saved to the specified output path.

## Example

To run the `llm_efficiency` task on a model:


```
agent-eval --model_args /path/to/model --task llm_efficiency
```

This will:
1. Run the custom `agent_eval` metrics (latency, tokens per second, parameters, perplexity)
2. Run the hellaswag benchmark using `lm-eval-harness`
3. Display the custom metrics in the console