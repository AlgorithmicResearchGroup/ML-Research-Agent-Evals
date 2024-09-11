# Agent-Eval

Agent-Eval is a tool for evaluating the performance of an agent on a set of tasks. The tool calculates evaluation metrics such as perplexity, accuracy, and latency, and generates evaluation reports to help visualize the results. The evaluation process can be performed manually or automatically using the evaluation library.

Evaluations can either be performed manually or automatically. The manual evaluation process involves running the agent on each task and manually executing the evaluation script to evaluate its performance based on predefined metrics. The automatic evaluation process involves using a library to calculate evaluation metrics, generate evaluation reports, and visualize the results.

## Setup

1. Build the Docker image:
   ```
   docker build -t agent-eval .
   ```

2. Run the evaluation for a specific task:
   ```
   docker run -v [task_name] [model_path] [results_output_path]


   docker run -it agent-eval \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -e TASK_NAME="mini_mini_pile" \
    -e MODEL_PATH="gpt2" \ 
   ```

   example:
    ```
  docker run llm_efficiency gpt2 .
   ```

   Replace `[task_name]` with one of the available tasks, `[model_path]` with the path to your model, and `[results_output_path]` with the desired output location for results.

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
docker run -v /path/to/models:/models agent-eval llm_efficiency /models/my_model 
```

This will:
1. Run the custom `agent_eval` metrics (latency, tokens per second, parameters, perplexity)
2. Run the hellaswag benchmark using `lm-eval-harness`
3. Display the custom metrics in the console

## Note

- Ensure that you have sufficient GPU resources available when running GPU-based evaluations.
- The evaluation process now includes more comprehensive benchmarks, which may take longer to complete depending on the task and model size.
- Review the `run.sh` script for any task-specific configurations or additional benchmarks that may be run for each task.