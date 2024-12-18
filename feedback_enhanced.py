import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler


class PrecomputedSentimentDataset(Dataset):
    def __init__(self, texts, rationales, labels, tokenizer, max_length=512):
        self.texts = texts
        self.rationales = rationales
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        rationale = self.rationales[idx]
        label = self.labels[idx]

        # Format input for T5
        input_text = f"sentiment analysis: {text} explanation: {rationale}"
        target_text = str(int(float(label) > 0.5))  # Convert to binary sentiment

        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=10,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
        }


class GemmaFeedbackGenerator:
    def __init__(self, model_name="google/gemma-1.1-2b-it", device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        print("Teacher model loaded successfully!")

    def generate_feedback(self, text, student_rationale, teacher_rationale):
        """Generate feedback comparing student's reasoning with teacher's reasoning"""
        prompt = (
            "Compare the following two explanations for sentiment analysis and provide specific feedback "
            "on how the student's explanation could be improved to match the teacher's quality.\n\n"
            f"Text: {text}\n"
            f"Student explanation: {student_rationale}\n"
            f"Teacher explanation: {teacher_rationale}\n"
            "Feedback:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return feedback[len(prompt):].strip()


class FeedbackEnhancedT5Student(nn.Module):
    def __init__(self, base_model, feedback_weight=0.3):
        super().__init__()
        self.model = base_model
        self.feedback_weight = feedback_weight

    def forward(self, input_ids, attention_mask, labels=None, feedback_signal=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        if feedback_signal is not None:
            # Incorporate feedback into the loss
            feedback_loss = (1 - feedback_signal).mean()
            outputs.loss = (1 - self.feedback_weight) * outputs.loss + self.feedback_weight * feedback_loss

        return outputs


class T5StudentTrainerWithFeedback:
    def __init__(self, model_name="t5-base", feedback_model_name="google/gemma-1.1-2b-it", device='cuda'):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        base_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model = FeedbackEnhancedT5Student(base_model).to(device)
        self.feedback_generator = GemmaFeedbackGenerator(feedback_model_name, device)

    def _generate_student_rationale(self, text):
        """Generate student's explanation for a given text"""
        input_text = f"Explain the sentiment of this text: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def train_with_feedback(self, train_data, val_data, shard_number, batch_size=8, epochs=3,
                            learning_rate=2e-5, feedback_frequency=10, save_dir='checkpoints'):
        """Train with periodic feedback from Gemma"""

        train_texts, train_rationales, train_labels = train_data
        val_texts, val_rationales, val_labels = val_data

        # Prepare data loaders
        train_loader = self.prepare_data(train_texts, train_rationales, train_labels, batch_size)
        val_loader = self.prepare_data(val_texts, val_rationales, val_labels, batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs
        )
        scaler = GradScaler()

        best_val_acc = 0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
                try:
                    optimizer.zero_grad()

                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    # Periodically get Gemma feedback
                    if batch_idx % feedback_frequency == 0:
                        feedback_signals = []
                        for idx in range(len(input_ids)):
                            # Get original text and teacher rationale from batch
                            text = self.tokenizer.decode(input_ids[idx], skip_special_tokens=True)
                            text = text.replace("sentiment analysis:", "").split("explanation:")[0].strip()
                            teacher_rationale = text.split("explanation:")[-1].strip()

                            # Generate student's current explanation
                            student_rationale = self._generate_student_rationale(text)

                            # Get feedback from Gemma
                            feedback = self.feedback_generator.generate_feedback(
                                text, student_rationale, teacher_rationale
                            )

                            # Convert feedback to signal
                            feedback_signal = torch.tensor(
                                0.5 + (0.5 * ("good" in feedback.lower() or "correct" in feedback.lower())),
                                device=self.device
                            )
                            feedback_signals.append(feedback_signal)

                        feedback_tensor = torch.stack(feedback_signals)
                    else:
                        feedback_tensor = None

                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            feedback_signal=feedback_tensor
                        )
                        loss = outputs.loss
                        total_loss += loss.item()

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print('| WARNING: ran out of memory, skipping batch')
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e

            # End of epoch evaluation
            avg_loss = total_loss / len(train_loader)
            val_metrics = self.evaluate(val_loader)
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Average training loss: {avg_loss:.4f}")
            print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")

            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(save_dir, shard_number, epoch, optimizer, scheduler, scaler)

            torch.cuda.empty_cache()

    def evaluate(self, data_loader):
        """Evaluate student model"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=2
                )

                predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                true_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                for pred, true in zip(predictions, true_labels):
                    pred = int(pred.strip() == "1")
                    true = int(true.strip() == "1")
                    correct += (pred == true)
                    total += 1

        return {
            'accuracy': correct / total if total > 0 else 0
        }

    def save_checkpoint(self, save_dir, shard_number, epoch, optimizer, scheduler, scaler):
        """Save training checkpoint with additional components"""
        checkpoint_dir = os.path.join(save_dir, f'shard_{shard_number}_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None, scaler=None):
        """Load training checkpoint with additional components"""
        checkpoint = torch.load(checkpoint_path)

        # Get the state dict
        state_dict = checkpoint['model_state_dict']

        # Create new state dict with correct structure
        new_state_dict = {}
        for key, value in state_dict.items():
            # Add 'model.' prefix if it doesn't exist
            if not key.startswith('model.'):
                new_key = f'model.{key}'
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # Load the modified state dict
        self.model.load_state_dict(new_state_dict)

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint

    def prepare_data(self, texts, rationales, labels, batch_size):
        dataset = PrecomputedSentimentDataset(
            texts, rationales, labels, self.tokenizer
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
