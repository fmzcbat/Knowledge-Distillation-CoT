import torch

from tqdm import tqdm


class DataLabeler:
    BASE_PROMPT = (
        """
        Provide your sentiment analysis and the final sentiment label (0 or 1, 0 for negative and 1 for positive) in the
         format as the examples.
        
        Example 1:
        ```
        Let's analyze this movie review step by step: `A masterpiece of direction and writing that's both
         thought-provoking and deeply moving`
        Step 1: Emotional words identified: "masterpiece" (very positive), "thought-provoking" (positive), "deeply
         moving" (positive)
        Step 2: Context: These terms describe both technical quality and emotional impact
        Step 3: No contradictions; consistently positive terms
        
        **Final Sentiment: 1**
        ```
        
        Example 2:
        ```
        Let's analyze this movie review step by step: `contains no wit, only labored gags`
        Step 1: Emotional words identified: "no wit" (negative), "labored gags" (negative, suggests forced/unsuccessful
         humor)
        Step 2: Context: The review criticizes both the lack of genuine humor and the poor quality of attempted jokes
        Step 3: Consistently negative terms with no positive elements

        **Final Sentiment: 0**
        ```
        
        Example 3:
        ```
        Let's analyze this movie review step by step: `that loves its characters and communicates something rather
         beautiful about human nature`
        Step 1: Emotional words identified: "loves" (positive), "beautiful" (positive)
        Step 2: Context: The review praises both: The treatment of characters; The deeper meaning about humanity
        Step 3: Entirely positive sentiment with no negative qualifiers

        **Final Sentiment: 1**
        ```
        
        Let's analyze this movie review step by step: 
        """
    )

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

    def generate_solution(self, question):
        prompt = (
            f"""{self.BASE_PROMPT} `{question}`"""
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_length = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True
        )

        # Get only the newly generated tokens
        generated_tokens = outputs.sequences[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        texts = generated_text.strip().split("**Final Sentiment: ")
        if len(texts) != 2:
            raise ValueError("Unexpected format of generated text.")

        # (rationale, answer)
        answer = texts[1].replace("**", "")
        return texts[0], answer

    def process_dataset(self, dataset):
        results = []
        for item in tqdm(dataset, desc="Generating solutions"):
            question = item['text']
            original_answer = item['label']

            try:
                generated_rationale, generated_solution = self.generate_solution(question)
                results.append({
                    'question': question,
                    'original_answer': original_answer,
                    'generated_rationale': generated_rationale,
                    'generated_solution': generated_solution
                })
            except Exception as e:
                print(f"Error processing question: {e}")
                continue

        return results

