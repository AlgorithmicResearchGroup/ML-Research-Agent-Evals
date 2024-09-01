from tqdm import tqdm
import torch
from datasets import load_dataset
from evaluate import load
import tiktoken
from pytorch_benchmarks.sample import Model_Generate

class RougeScoreBenchmark:
    def __init__(
        self, model, dataset_name, dataset_config=None, num_samples=100
    ):
        self.model = model
        self.sample = Model_Generate(self.model)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.num_samples = num_samples
        self.rouge_scorer = load("rouge")

    def generate_text(self, encoded_input):
        text, y, logits = self.sample.return_response(encoded_input)
        return text
            

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
            encode = lambda s: self.tokenizer.encode(s, allowed_special={"<|endoftext|>"})
            encoded_input = encode(reference_text)
            generated_text = self.generate_text(encoded_input)
            generated_texts.append(generated_text)

        rouge_scores = self.calculate_rouge_score(reference_texts, generated_texts)

        print(f"\nROUGE-1 score on {self.num_samples} samples: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2 score on {self.num_samples} samples: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L score on {self.num_samples} samples: {rouge_scores['rougeL']:.4f}")

        return rouge_scores