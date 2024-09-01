from tqdm import tqdm
import torch
from datasets import load_dataset
from evaluate import load

class RougeScoreBenchmark:
    def __init__(
        self, model, tokenizer, dataset_name, dataset_config=None, num_samples=100
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.num_samples = num_samples
        self.rouge_scorer = load("rouge")

    def generate_text(self, encoded_input):
        with torch.no_grad():
            outputs = self.model.generate(
                encoded_input,
                max_new_tokens=50,  # Number of new tokens to generate
                temperature=0.8,
                top_k=300,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def calculate_rouge_score(self, reference_texts, generated_texts):
        scores = self.rouge_scorer.compute(
            predictions=generated_texts,
            references=reference_texts,
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
            use_aggregator=True  # Ensure we get the detailed score object
        )
        return {
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"],
        }

    def benchmark_rouge_score(self):
        dataset = load_dataset(self.dataset_name, self.dataset_config)

        reference_texts = []
        generated_texts = []

        print(
            f"Starting ROUGE benchmark on {self.num_samples} samples from the {self.dataset_name} dataset..."
        )
        for sample in tqdm(dataset["train"].select(range(self.num_samples))):
            reference_text = sample["article"]
            reference_texts.append(reference_text)

            # Truncate input text to fit within the model's token limit
            encoded_input = self.tokenizer.encode(
                reference_text, return_tensors="pt", truncation=True, max_length=200
            ).to(self.model.device)
            generated_text = self.generate_text(encoded_input)
            generated_texts.append(generated_text)

        rouge_scores = self.calculate_rouge_score(reference_texts, generated_texts)

        print(f"\nROUGE-1 score on {self.num_samples} samples: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2 score on {self.num_samples} samples: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L score on {self.num_samples} samples: {rouge_scores['rougeL']:.4f}")

        return rouge_scores