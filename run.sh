#!/bin/bash
display_usage() {
    echo "Usage: $0 <task_name> <model_args>"
    echo "Available tasks:"
    echo "  llm_efficiency, mini_llm_efficiency"
    echo "  baby_lm, mini_baby_lm"
    echo "  mini_pile, mini_mini_pile"
    echo "  budget_model_training, mini_budget_model_training"
    echo "  budget_model_inference, mini_budget_model_inference"
    echo "  llm_merging, mini_llm_merging"
    echo "  edge_llm_compression, mini_edge_llm_compression"
    echo "  edge_llm_training, mini_edge_llm_training"
    echo "  math_reasoning, mini_math_reasoning"
}

# Check if arguments are provided
if [ "$#" -eq 0 ]; then
    echo "Error: No arguments provided"
    display_usage
    exit 1
fi

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments"
    display_usage
    exit 1
fi

# Update the python command to use the correct path
PYTHON_CMD="cd /app && python agent_eval/cli.py"

# Add other task conditions
if [[ "$1" == *"llm_efficiency"* || "$1" == *"mini_llm_efficiency"* ]]
then
    echo "Running llm_efficiency"
    $PYTHON_CMD --model_args "$2" --task llm_efficiency 

    python -m lm_eval --model hf \
    --model_args pretrained="$2" \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8

fi

if [[ "$1" == *"baby_lm"* || "$1" == *"mini_baby_lm"* ]]
then
    echo "Running baby_lm"
    python agent_eval/cli.py --model_args "$2" --task baby_lm

    lm_eval --model hf \
    --model_args pretrained="$2" \
    --tasks hellaswag,gsm8k,arc_easy,mmlu,blimp \
    --device cuda:0 \
    --batch_size 8
fi

if [[ "$1" == *"mini_pile"* || "$1" == *"mini_mini_pile"* ]]
then
    echo "Running mini_pile"
    python agent_eval/cli.py --model_args "$2" --task mini_pile

    lm_eval --model hf \
    --model_args pretrained="$2" \
    --tasks glue \
    --device cuda:0 \
    --batch_size 8
fi

if [[ "$1" == *"budget_model_training"* || "$1" == *"mini_budget_model_training"* ]]
then
    echo "Running budget_model_training"
    python agent_eval/cli.py --model_args "$2" --task budget_model_training
fi

if [[ "$1" == *"budget_model_inference"* || "$1" == *"mini_budget_model_inference"* ]]
then
    echo "Running budget_model_inference"
        python agent_eval/cli.py --model_args "$2" --task budget_model_inference
fi


if [[ "$1" == *"llm_merging"* || "$1" == *"mini_llm_merging"* ]]
then
    echo "Running llm_merging"
        python agent_eval/cli.py --model_args "$2" --task llm_merging

    lm_eval --model hf \
    --model_args pretrained="$2" \
    --tasks mmlu  \
    --device cuda:0 \
    --batch_size 8
fi

if [[ "$1" == *"edge_llm_compression"* || "$1" == *"mini_edge_llm_compression"* ]]
then
    echo "Running edge_llm_compression"
    python agent_eval/cli.py --model_args "$2" --task edge_llm_compression


    lm_eval --model hf \
    --model_args pretrained="$2" \
    --tasks mmlu  \
    --device cuda:0 \
    --batch_size 8
fi

if [[ "$1" == *"edge_llm_training"* || "$1" == *"mini_edge_llm_training"* ]]
then
    echo "Running edge_llm_training"
    python agent_eval/cli.py --model_args "$2" --task edge_llm_training

    lm_eval --model hf \
    --model_args pretrained="$2" \
    --tasks mmlu  \
    --device cuda:0 \
    --batch_size 8
fi


if [[ "$1" == *"math_reasoning"* || "$1" == *"mini_math_reasoning"* ]]
then
    echo "Running math_reasoning"
    python agent_eval/cli.py --model_args "$2" --task math_reasoning

    lm_eval --model hf \
    --model_args pretrained="$2" \
    --tasks mmlu_high_school_mathematics,mmlu_high_school_statistics,mmlu_high_school_computer_science \
    --device cuda:0 \
    --batch_size 8
fi

echo "Evaluation complete"


