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




GPT2 30M parameters model (CPU)
| Metric                      | Value      |
|-----------------------------|------------|
| Total Parameters            | 30,044,544 |
| Non-Trainable Parameters    | 0          |
| Trainable Parameters        | 30,044,544 |
| Tokens per Second (avg)     | 191.17     |
| Tokens per Second (std)     | 25.25      |
| Tokens per Second (median)  | 197.21     |
| Tokens per Second (min)     | 59.41      |
| Tokens per Second (max)     | 202.30     |
| Tokens per Second (p90)     | 201.78     |
| Tokens per Second (p95)     | 202.03     |
| Perplexity (avg)            | 95.2161    |
| Perplexity (median)         | 96.1996    |
| Perplexity (min)            | 77.8055    |
| Perplexity (max)            | 191.9815   |
| Perplexity (p90)            | 102.5676   |
| Perplexity (p95)            | 102.9616   |
| Latency (ms) (avg)          | 2.08       |
| Latency (ms) (median)       | 2.09       |
| Latency (ms) (min)          | 2.02       |
| Latency (ms) (max)          | 2.16       |
| Latency (ms) (p90)          | 2.13       |
| Latency (ms) (p95)          | 2.15       |
| ROUGE-1 Score               | 0.4891     |
| ROUGE-2 Score               | 0.4558     |
| ROUGE-L Score               | 0.4803     |


GPT2 30M parameters model (GPU)
| Metric                      | Value      |
|-----------------------------|------------|
| Total Parameters            | 30,044,544 |
| Non-Trainable Parameters    | 0          |
| Trainable Parameters        | 30,044,544 |
| Tokens per Second (avg)     | 242.70     |
| Tokens per Second (std)     | 270.66     |
| Tokens per Second (median)  | 175.58     |
| Tokens per Second (min)     | 161.59     |
| Tokens per Second (max)     | 1519.79    |
| Tokens per Second (p90)     | 181.98     |
| Tokens per Second (p95)     | 577.12     |
| Perplexity (avg)            | 95.2161    |
| Perplexity (median)         | 96.1996    |
| Perplexity (min)            | 77.8055    |
| Perplexity (max)            | 191.9814   |
| Perplexity (p90)            | 102.5676   |
| Perplexity (p95)            | 102.9616   |
| Latency (ms) (avg)          | 2.25       |
| Latency (ms) (median)       | 2.21       |
| Latency (ms) (min)          | 2.07       |
| Latency (ms) (max)          | 2.53       |
| Latency (ms) (p90)          | 2.43       |
| Latency (ms) (p95)          | 2.48       |
| ROUGE-1 Score               | 0.4894     |
| ROUGE-2 Score               | 0.4542     |
| ROUGE-L Score               | 0.4803     |


GPT2 30M parameters - 4 bit quantized model
| Metric                      | Value      |
|-----------------------------|------------|
| Total Parameters            | 24,736,128 |
| Non-Trainable Parameters    | 5,329,152  |
| Trainable Parameters        | 19,406,976 |
| Tokens per Second (avg)     | 190.06     |
| Tokens per Second (std)     | 7.81       |
| Tokens per Second (median)  | 191.15     |
| Tokens per Second (min)     | 175.75     |
| Tokens per Second (max)     | 202.37     |
| Tokens per Second (p90)     | 200.87     |
| Tokens per Second (p95)     | 201.26     |
| Perplexity (avg)            | 93.7483    |
| Perplexity (median)         | 94.8143    |
| Perplexity (min)            | 76.5667    |
| Perplexity (max)            | 192.5414   |
| Perplexity (p90)            | 100.9117   |
| Perplexity (p95)            | 101.3199   |
| Latency (ms) (avg)          | 1.91       |
| Latency (ms) (median)       | 1.91       |
| Latency (ms) (min)          | 1.89       |
| Latency (ms) (max)          | 1.94       |
| Latency (ms) (p90)          | 1.93       |
| Latency (ms) (p95)          | 1.93       |
| ROUGE-1 Score               | 0.4906     |
| ROUGE-2 Score               | 0.4555     |
| ROUGE-L Score               | 0.4795     |


GPT2 30M parameters - 8 bit quantized model
| Metric                      | Value      |
|-----------------------------|------------|
| Total Parameters            | 30,044,544 |
| Non-Trainable Parameters    | 10,637,568 |
| Trainable Parameters        | 19,406,976 |
| Tokens per Second (avg)     | 86.86      |
| Tokens per Second (std)     | 6.38       |
| Tokens per Second (median)  | 88.14      |
| Tokens per Second (min)     | 54.33      |
| Tokens per Second (max)     | 91.51      |
| Tokens per Second (p90)     | 90.53      |
| Tokens per Second (p95)     | 90.81      |
| Perplexity (avg)            | 91.7547    |
| Perplexity (median)         | 92.7215    |
| Perplexity (min)            | 75.0153    |
| Perplexity (max)            | 190.5842   |
| Perplexity (p90)            | 98.6888    |
| Perplexity (p95)            | 99.0690    |
| Latency (ms) (avg)          | 4.65       |
| Latency (ms) (median)       | 4.64       |
| Latency (ms) (min)          | 4.37       |
| Latency (ms) (max)          | 4.96       |
| Latency (ms) (p90)          | 4.89       |
| Latency (ms) (p95)          | 4.92       |
| ROUGE-1 Score               | 0.4887     |
| ROUGE-2 Score               | 0.4542     |
| ROUGE-L Score               | 0.4775     |


GPT2 124M parameters model (GPU): openai-community/gpt2
| Metric                      | Value       |
|-----------------------------|-------------|
| Total Parameters            | 124,439,808 |
| Non-Trainable Parameters    | 0           |
| Trainable Parameters        | 124,439,808 |
| Tokens per Second (avg)     | 65.86       |
| Tokens per Second (std)     | 4.01        |
| Tokens per Second (median)  | 64.93       |
| Tokens per Second (min)     | 60.11       |
| Tokens per Second (max)     | 80.91       |
| Tokens per Second (p90)     | 70.12       |
| Tokens per Second (p95)     | 70.97       |
| Perplexity (avg)            | 33.2258     |
| Perplexity (median)         | 32.6837     |
| Perplexity (min)            | 31.3960     |
| Perplexity (max)            | 57.8901     |
| Perplexity (p90)            | 35.1760     |
| Perplexity (p95)            | 35.5833     |
| Latency (ms) (avg)          | 6.01        |
| Latency (ms) (median)       | 5.99        |
| Latency (ms) (min)          | 5.74        |
| Latency (ms) (max)          | 6.21        |
| Latency (ms) (p90)          | 6.20        |
| Latency (ms) (p95)          | 6.20        |
| ROUGE-1 Score               | 0.4993      |
| ROUGE-2 Score               | 0.4538      |
| ROUGE-L Score               | 0.4819      |


GPT2 124M parameters model - 4 bit quantized model: openai-community/gpt2
| Metric                      | Value       |
|-----------------------------|-------------|
| Total Parameters            | 81,972,480  |
| Non-Trainable Parameters    | 42,550,272  |
| Trainable Parameters        | 39,422,208  |
| Tokens per Second (avg)     | 106.69      |
| Tokens per Second (std)     | 2.99        |
| Tokens per Second (median)  | 107.16      |
| Tokens per Second (min)     | 101.04      |
| Tokens per Second (max)     | 113.47      |
| Tokens per Second (p90)     | 110.27      |
| Tokens per Second (p95)     | 110.86      |
| Perplexity (avg)            | 38.6520     |
| Perplexity (median)         | 38.1042     |
| Perplexity (min)            | 36.6629     |
| Perplexity (max)            | 61.5566     |
| Perplexity (p90)            | 40.8819     |
| Perplexity (p95)            | 41.3970     |
| Latency (ms) (avg)          | 3.58        |
| Latency (ms) (median)       | 3.54        |
| Latency (ms) (min)          | 3.45        |
| Latency (ms) (max)          | 3.86        |
| Latency (ms) (p90)          | 3.76        |
| Latency (ms) (p95)          | 3.81        |
| ROUGE-1 Score               | 0.5039      |
| ROUGE-2 Score               | 0.4536      |
| ROUGE-L Score               | 0.4822      |


GPT2 124M parameters model - 8 bit quantized model: openai-community/gpt2
| Metric                      | Value       |
|-----------------------------|-------------|
| Total Parameters            | 124,439,808 |
| Non-Trainable Parameters    | 85,017,600  |
| Trainable Parameters        | 39,422,208  |
| Tokens per Second (avg)     | 41.51       |
| Tokens per Second (std)     | 1.59        |
| Tokens per Second (median)  | 41.54       |
| Tokens per Second (min)     | 39.22       |
| Tokens per Second (max)     | 45.86       |
| Tokens per Second (p90)     | 43.28       |
| Tokens per Second (p95)     | 43.95       |
| Perplexity (avg)            | 33.2936     |
| Perplexity (median)         | 32.7487     |
| Perplexity (min)            | 31.4940     |
| Perplexity (max)            | 58.4141     |
| Perplexity (p90)            | 35.2813     |
| Perplexity (p95)            | 35.7158     |
| Latency (ms) (avg)          | 8.97        |
| Latency (ms) (median)       | 8.97        |
| Latency (ms) (min)          | 8.64        |
| Latency (ms) (max)          | 9.27        |
| Latency (ms) (p90)          | 9.15        |
| Latency (ms) (p95)          | 9.21        |
| ROUGE-1 Score               | 0.5006      |
| ROUGE-2 Score               | 0.4528      |
| ROUGE-L Score               | 0.4805      |


GPT2 354M parameters model (GPU): openai-community/gpt2-medium
| Metric                      | Value      |
|-----------------------------|------------|
| Total Parameters            | 354,823,168 |
| Non-Trainable Parameters    | 0          |
| Trainable Parameters        | 354,823,168 |
| Tokens per Second (avg)     | 29.11      |
| Tokens per Second (std)     | 8.56       |
| Tokens per Second (median)  | 26.91      |
| Tokens per Second (min)     | 23.38      |
| Tokens per Second (max)     | 73.32      |
| Tokens per Second (p90)     | 30.60      |
| Tokens per Second (p95)     | 35.21      |
| Perplexity (avg)            | 23.7864    |
| Perplexity (median)         | 23.3280    |
| Perplexity (min)            | 22.5567    |
| Perplexity (max)            | 38.1757    |
| Perplexity (p90)            | 25.3799    |
| Perplexity (p95)            | 25.7811    |
| Latency (ms) (avg)          | 14.38      |
| Latency (ms) (median)       | 14.33      |
| Latency (ms) (min)          | 13.96      |
| Latency (ms) (max)          | 14.74      |
| Latency (ms) (p90)          | 14.68      |
| Latency (ms) (p95)          | 14.71      |
| ROUGE-1 Score               | 0.5006     |
| ROUGE-2 Score               | 0.4540     |
| ROUGE-L Score               | 0.4819     |


GPT2 354M parameters model - 4 bit quantized model: openai-community/gpt2-medium
| Metric                      | Value       |
|-----------------------------|-------------|
| Total Parameters            | 203,828,224 |
| Non-Trainable Parameters    | 151,216,128 |
| Trainable Parameters        | 52,612,096  |
| Tokens per Second (avg)     | 60.73       |
| Tokens per Second (std)     | 3.09        |
| Tokens per Second (median)  | 59.99       |
| Tokens per Second (min)     | 58.31       |
| Tokens per Second (max)     | 74.44       |
| Tokens per Second (p90)     | 61.10       |
| Tokens per Second (p95)     | 66.03       |
| Perplexity (avg)            | 25.7124     |
| Perplexity (median)         | 25.2522     |
| Perplexity (min)            | 24.3486     |
| Perplexity (max)            | 39.3701     |
| Perplexity (p90)            | 27.3985     |
| Perplexity (p95)            | 27.7908     |
| Latency (ms) (avg)          | 7.02        |
| Latency (ms) (median)       | 6.95        |
| Latency (ms) (min)          | 6.66        |
| Latency (ms) (max)          | 7.52        |
| Latency (ms) (p90)          | 7.46        |
| Latency (ms) (p95)          | 7.49        |
| ROUGE-1 Score               | 0.5030      |
| ROUGE-2 Score               | 0.4545      |
| ROUGE-L Score               | 0.4824      |


GPT2 354M parameters model - 8 bit quantized model: openai-community/gpt2-medium
| Metric                      | Value       |
|-----------------------------|-------------|
| Total Parameters            | 354,823,168 |
| Non-Trainable Parameters    | 302,211,072 |
| Trainable Parameters        | 52,612,096  |
| Tokens per Second (avg)     | 22.59       |
| Tokens per Second (std)     | 1.63        |
| Tokens per Second (median)  | 22.45       |
| Tokens per Second (min)     | 20.63       |
| Tokens per Second (max)     | 28.83       |
| Tokens per Second (p90)     | 23.17       |
| Tokens per Second (p95)     | 25.57       |
| Perplexity (avg)            | 23.8487     |
| Perplexity (median)         | 23.3734     |
| Perplexity (min)            | 22.6434     |
| Perplexity (max)            | 38.6032     |
| Perplexity (p90)            | 25.4763     |
| Perplexity (p95)            | 25.8514     |
| Latency (ms) (avg)          | 18.62       |
| Latency (ms) (median)       | 18.68       |
| Latency (ms) (min)          | 17.96       |
| Latency (ms) (max)          | 19.04       |
| Latency (ms) (p90)          | 18.86       |
| Latency (ms) (p95)          | 18.95       |
| ROUGE-1 Score               | 0.5013      |
| ROUGE-2 Score               | 0.4543      |
| ROUGE-L Score               | 0.4802      |


GPT2 774M parameters model (GPU): openai-community/gpt2-large
| Metric                      | Value       |
|-----------------------------|-------------|
| Total Parameters            | 774,030,080 |
| Non-Trainable Parameters    | 0           |
| Trainable Parameters        | 774,030,080 |
| Tokens per Second (avg)     | 13.51       |
| Tokens per Second (std)     | 0.50        |
| Tokens per Second (median)  | 13.51       |
| Tokens per Second (min)     | 12.15       |
| Tokens per Second (max)     | 15.16       |
| Tokens per Second (p90)     | 13.84       |
| Tokens per Second (p95)     | 14.08       |
| Perplexity (avg)            | 20.7318     |
| Perplexity (median)         | 20.2512     |
| Perplexity (min)            | 19.6610     |
| Perplexity (max)            | 35.5371     |
| Perplexity (p90)            | 22.0921     |
| Perplexity (p95)            | 22.4243     |
| Latency (ms) (avg)          | 29.48       |
| Latency (ms) (median)       | 29.47       |
| Latency (ms) (min)          | 28.59       |
| Latency (ms) (max)          | 30.67       |
| Latency (ms) (p90)          | 30.32       |
| Latency (ms) (p95)          | 30.50       |
| ROUGE-1 Score               | 0.5050      |
| ROUGE-2 Score               | 0.4568      |
| ROUGE-L Score               | 0.4837      |


GPT2 774M parameters model - 4 bit quantized model: openai-community/gpt2-large
| Metric                      | Value       |
|-----------------------------|-------------|
| Total Parameters            | 420,135,680 |
| Non-Trainable Parameters    | 354,309,120 |
| Trainable Parameters        | 65,826,560  |
| Tokens per Second (avg)     | 40.61       |
| Tokens per Second (std)     | 1.55        |
| Tokens per Second (median)  | 40.12       |
| Tokens per Second (min)     | 39.43       |
| Tokens per Second (max)     | 46.94       |
| Tokens per Second (p90)     | 41.76       |
| Tokens per Second (p95)     | 43.81       |
| Perplexity (avg)            | 21.6462     |
| Perplexity (median)         | 21.1214     |
| Perplexity (min)            | 20.4944     |
| Perplexity (max)            | 36.8384     |
| Perplexity (p90)            | 23.1241     |
| Perplexity (p95)            | 23.4670     |
| Latency (ms) (avg)          | 9.73        |
| Latency (ms) (median)       | 9.71        |
| Latency (ms) (min)          | 9.64        |
| Latency (ms) (max)          | 9.90        |
| Latency (ms) (p90)          | 9.83        |
| Latency (ms) (p95)          | 9.87        |
| ROUGE-1 Score               | 0.5043      |
| ROUGE-2 Score               | 0.4564      |
| ROUGE-L Score               | 0.4843      |


GPT2 774M parameters model - 8 bit quantized model: openai-community/gpt2-large
| Metric                      | Value       |
|-----------------------------|-------------|
| Total Parameters            | 774,030,080 |
| Non-Trainable Parameters    | 708,203,520 |
| Trainable Parameters        | 65,826,560  |
| Tokens per Second (avg)     | 12.94       |
| Tokens per Second (std)     | 0.94        |
| Tokens per Second (median)  | 12.70       |
| Tokens per Second (min)     | 12.10       |
| Tokens per Second (max)     | 17.37       |
| Tokens per Second (p90)     | 13.45       |
| Tokens per Second (p95)     | 14.17       |
| Perplexity (avg)            | 20.6859     |
| Perplexity (median)         | 20.1899     |
| Perplexity (min)            | 19.5724     |
| Perplexity (max)            | 35.7887     |
| Perplexity (p90)            | 22.0796     |
| Perplexity (p95)            | 22.4133     |
| Latency (ms) (avg)          | 29.71       |
| Latency (ms) (median)       | 29.72       |
| Latency (ms) (min)          | 29.41       |
| Latency (ms) (max)          | 29.96       |
| Latency (ms) (p90)          | 29.95       |
| Latency (ms) (p95)          | 29.96       |
| ROUGE-1 Score               | 0.5037      |
| ROUGE-2 Score               | 0.4557      |
| ROUGE-L Score               | 0.4829      |