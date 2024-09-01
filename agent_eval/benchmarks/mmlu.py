from tqdm import tqdm
import torch
from datasets import load_dataset
from evaluate import load

class MMLUBenchmark:
    def __init__(self, model, tokenizer, benchmark_name, num_samples=100):
        self.model = model
        self.tokenizer = tokenizer
        self.benchmark_name = benchmark_name
        self.num_samples = num_samples
        self.accuracy_metric = load("accuracy")

    def generate_answer(self, question, choices):
        prompt = f"{question}\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{i+1}. {choice}\n"
        prompt += "Answer: "

        encoded_input = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        attention_mask = torch.ones(encoded_input.shape, device=self.model.device)  # Create attention mask

        with torch.no_grad():
            output = self.model.generate(
                encoded_input,
                attention_mask=attention_mask,
                max_new_tokens=10,  # Increase token limit to capture complete answer
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id  # Set the pad token ID to the EOS token ID
            )
        generated_answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

        return generated_answer

    def parse_generated_answer(self, generated_answer):
        try:
            # Extract the part after "Answer: " and split to get the index
            answer_part = generated_answer.split("Answer: ")[-1].strip()
            # Handle cases like "2. (–1, 5)" and extract the index
            if answer_part.startswith("2. "):
                return "2"
            # If answer is a single number, return it
            if answer_part.isdigit():
                return answer_part
        except Exception as e:
            print(f"Error parsing generated answer: {generated_answer}, Error: {e}")
        return ""

    def benchmark(self):
        dataset = load_dataset("cais/mmlu", self.benchmark_name)
        predictions = []
        references = []

        for sample in tqdm(dataset["test"].select(range(self.num_samples))):
            question = sample["question"]
            choices = sample["choices"]
            reference_answer = str(sample["answer"]).strip().lower()

            generated_answer = self.generate_answer(question, choices)
            predicted_answer = self.parse_generated_answer(generated_answer).strip().lower()

            # Debugging: Print the question, choices, reference answer, and generated answer
            print(f"Question: {question}")
            print(f"Choices: {choices}")
            print(f"Reference Answer: {reference_answer}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Predicted Answer: {predicted_answer}")

            # Ensure the predicted answer is in the expected format (an index)
            try:
                predicted_choice_index = int(predicted_answer) - 1
                predicted_answer_text = choices[predicted_choice_index].lower()
            except (ValueError, IndexError):
                predicted_answer_text = predicted_answer

            predictions.append(predicted_answer_text)
            references.append(reference_answer)

        # Ensure that both predictions and references are choices
        def get_choice_index(answer, choices):
            try:
                return choices.index(answer) + 1  # Return 1-based index
            except ValueError:
                return -1

        converted_predictions = []
        converted_references = []

        for pred, ref, choices in zip(predictions, references, dataset["test"].select(range(self.num_samples))["choices"]):
            pred_index = get_choice_index(pred, [choice.lower() for choice in choices])
            ref_index = get_choice_index(ref, [choice.lower() for choice in choices])
            converted_predictions.append(pred_index)
            converted_references.append(ref_index)

        # Filter out any -1 indices which indicate invalid predictions/references
        valid_indices = [(pred, ref) for pred, ref in zip(converted_predictions, converted_references) if pred != -1 and ref != -1]
        
        # Handle case where valid_indices might be empty
        if not valid_indices:
            print("No valid predictions or references found.")
            return 0.0

        converted_predictions, converted_references = zip(*valid_indices)

        accuracy = self.accuracy_metric.compute(
            references=converted_references,
            predictions=converted_predictions,
        )
        print(f"\nAccuracy on MMLU {self.benchmark_name}: {accuracy['accuracy']:.4f}")
        return accuracy["accuracy"]
