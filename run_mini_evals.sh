########################################################
# 1. Mini BabyLM
########################################################

!git clone https://github.com/babylm/evaluation-pipeline-2024
%cd evaluation-pipeline-2024
!pip install -e .
!pip install minicons
!pip install --upgrade accelerate

!unzip evaluation_data.zip
!mkdir evaluation_data
!bash eval_blimp.sh /content/gpt2_babylm_model/checkpoint-3250/


########################################################
# 2. Mini Budget Model Train
########################################################

python eval_budget_model_train.py

########################################################
# 3. Mini Budget Model Inference
########################################################

python eval_budget_model_inference.py

########################################################
# 4. Mini Edge LLM Compression
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8

########################################################
# 5. Mini Edge LLM Compression
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8

########################################################
# 6. Mini LLM EFFICIENCY
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks hellaswag, gsm8k, arc, mmlu, blimp \
    --device cuda:0 \
    --batch_size 8

########################################################
# 7. Mini LLM Merging
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8


########################################################
# 8. Mini Math
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks math, mmlu_high_school_mathmatics, mmlu_college_mathmatics  \
    --device cuda:0 \
    --batch_size 8


########################################################
# 9. Mini MiniPile
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks glue \
    --device cuda:0 \
    --batch_size 8