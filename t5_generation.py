import torch

from tqdm import tqdm


class T5Labeler:
    def __init__(self, tokenizer, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = tokenizer
        self.model = model

        # Set up padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.to(self.device)

    def _get_final_scores(self, results, total):
        correct = 0
        for each in results:
            correct += 1 if str(each["original_answer"]) == each["generated_solution"] else 0
        return correct / total

    def generate_solution(self, question):
        input_text = f"sentiment: `{question}`"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to("cuda").input_ids

        outputs = self.model.generate(inputs)

        # Get only the newly generated tokens
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text in ["True", "positive", "1"]:
            return "1"
        elif generated_text in ["False", "negative", "0"]:
            return "0"
        else:
            raise ValueError(f"Unexpected return value: {generated_text}")

    def process_dataset(self, dataset):
        results = []
        total_entity_num = len(dataset)
        for item in tqdm(dataset, desc="Generating solutions"):
            question = item['text']
            original_answer = item['label']

            try:
                generated_solution = self.generate_solution(question)
                results.append({
                    'question': question,
                    'original_answer': original_answer,
                    'generated_solution': generated_solution
                })
            except Exception as e:
                print(f"Error processing question: {e}")
                continue

        return self._get_final_scores(results, total_entity_num)