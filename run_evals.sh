########################################################
# 1. BabyLM
########################################################

!unzip evaluation_data.zip
!mkdir evaluation_data

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks blimp_filtered,blimp_supplement, ewok_filtered \
    --device cuda:0 \
    --batch_size 8
    --log_samples \
    --output_path results/${MODEL_BASENAME}/baby_lm_results.json


########################################################
# 2. Budget Model Train
########################################################

python eval_budget_model_train.py

########################################################
# 3. Budget Model Inference
########################################################

python eval_budget_model_inference.py

########################################################
# 4. Edge LLM Compression
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path results/${MODEL_BASENAME}/budget_model_inference_results.json

########################################################
# 5. Edge LLM Compression
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path results/${MODEL_BASENAME}/edge_llm_compression_results.json

########################################################
# 6. LLM EFFICIENCY
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks hellaswag, gsm8k, arc, mmlu, blimp \
    --device cuda:0 \
    --batch_size 8
    --log_samples \
    --output_path results/${MODEL_BASENAME}/llm_efficiency_results.json

########################################################
# 7. LLM Merging
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path results/${MODEL_BASENAME}/llm_merging_results.json


########################################################
# 8. Math
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks ?  \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path results/${MODEL_BASENAME}/math_results.json


########################################################
# 9. MiniPile
########################################################

!lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks super_glue \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path results/${MODEL_BASENAME}/minipile_results.json



    






