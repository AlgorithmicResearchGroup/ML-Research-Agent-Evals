from tqdm import tqdm
import torch
from datasets import load_dataset
from evaluate import load

class BLEUScoreBenchmark:
    def __init__(
        self, model, tokenizer, dataset_path, num_samples=100
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.bleu_scorer = load("bleu")

    def generate_text(self, encoded_input):
        with torch.no_grad():
            outputs = self.model.generate(
                encoded_input,
                max_new_tokens=200,  # Increased for longer proofs
                temperature=0.7,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def calculate_bleu_score(self, reference_texts, generated_texts):
        tokenized_references = [[text.split()] for text in reference_texts]
        tokenized_predictions = [text.split() for text in generated_texts]
        
        scores = self.bleu_scorer.compute(
            predictions=tokenized_predictions,
            references=tokenized_references
        )
        return scores['bleu']

    def benchmark_bleu_score(self):
        dataset = load_dataset("json", data_files=self.dataset_path)["train"]

        reference_proofs = []
        generated_proofs = []

        print(f"Starting BLEU benchmark on {self.num_samples} samples from the dataset...")
        for sample in tqdm(dataset.select(range(self.num_samples))):
            problem_statement = sample["informal_statement"]
            reference_proof = sample["informal_proof"]
            reference_proofs.append(reference_proof)

            prompt = f"Problem: {problem_statement}\nSolution:"
            encoded_input = self.tokenizer.encode(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.model.device)
            generated_proof = self.generate_text(encoded_input)
            generated_proofs.append(generated_proof)

        bleu_score = self.calculate_bleu_score(reference_proofs, generated_proofs)

        print(f"\nBLEU score on {self.num_samples} samples: {bleu_score:.4f}")

        return bleu_score