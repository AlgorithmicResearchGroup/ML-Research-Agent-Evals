import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig


def load_test_model(model_name, bits=None, use_ort=False, quantized=True):
    # Setup the quantization config if the model is to be quantized
    if quantized:
        if bits == 4:
            print("Quantizing model to 4 bits.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        elif bits == 8:
            print("Quantizing model to 8 bits.")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            raise ValueError("Unsupported bit configuration. Choose either 4 or 8.")
    else:
        quantization_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Decide between a normal model, ORT model, or non-quantized model based on flags
    if quantized:
        if use_ort:
            print("Using ORT model.")
            model = ORTModelForCausalLM.from_pretrained(
                model_name, quantization_config=quantization_config
            )
        else:
            print("Using quantized model.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quantization_config
            )
    else:
        print("Using non-quantized model.")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Optional: Print the device being used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device.type)

    return model, tokenizer