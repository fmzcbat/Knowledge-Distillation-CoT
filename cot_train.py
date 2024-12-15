import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
from torch.cuda.amp import autocast, GradScaler


class SentimentDatasetWithRationale(Dataset):
    def __init__(self, questions, rationales, labels, tokenizer, max_length=1024):
        self.questions = questions
        self.rationales = rationales
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        rationale = self.rationales[idx]
        # Use numpy for consistent float handling
        label = "1" if np.float32(self.labels[idx]) > 0.5 else "0"

        # Combine question and rationale in the input
        input_text = f"sentiment analysis - Question: {question} Rationale: {rationale}"

        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            label,
            max_length=10,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

class T5SentimentAnalyzerCustom:
    def __init__(self, model_name="t5-base", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model_name = model_name
        print(f"Using device: {device}")

    def save_checkpoint(self, save_dir, shard_number, epoch, step, optimizer, scheduler, scaler, loss, test_metrics):
        """Save training checkpoint with additional components"""
        checkpoint_dir = os.path.join(save_dir, f'shard_{shard_number}_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_epoch_{epoch}_step_{step}.pt'
        )

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'loss': loss,
            'test_metrics': test_metrics
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None, scaler=None):
        """Load training checkpoint with additional components"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint

    def train_shard(self, train_data, test_data, shard_number, save_dir,
                    batch_size=4, epochs=3, learning_rate=2e-5,
                    checkpoint_steps=100, gradient_accumulation_steps=4,
                    max_length=128, warmup_steps=0, weight_decay=0.01,
                    max_grad_norm=1.0):
        """Enhanced training with additional optimizations"""
        print(f"\nTraining on shard {shard_number}")

        # Prepare datasets
        train_questions, train_rationales, train_labels = train_data
        test_questions, test_rationales, test_labels = test_data

        train_dataset = SentimentDatasetWithRationale(
            train_questions, train_rationales, train_labels,
            self.tokenizer, max_length=max_length
        )
        test_dataset = SentimentDatasetWithRationale(
            test_questions, test_rationales, test_labels,
            self.tokenizer, max_length=max_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

        # Initialize training components
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        num_training_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        scaler = GradScaler()

        # Training loop
        self.model.train()
        global_step = 0
        best_accuracy = 0
        metrics_history = {
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': []
        }

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    # Mixed precision training
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss / gradient_accumulation_steps

                    # Scale loss and backward pass
                    scaler.scale(loss).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Clip gradients
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                        # Optimizer and scheduler step
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                    current_loss = loss.item() * gradient_accumulation_steps
                    total_loss += current_loss

                    # Evaluation and checkpointing
                    if global_step > 0 and global_step % checkpoint_steps == 0:
                        test_metrics = self.evaluate_batch(test_loader)

                        # Update metrics history
                        metrics_history['train_loss'].append(total_loss / (batch_idx + 1))
                        metrics_history['test_loss'].append(test_metrics['test_loss'])
                        metrics_history['test_accuracy'].append(test_metrics['test_accuracy'])

                        progress_bar.set_postfix({
                            'loss': current_loss,
                            'test_acc': f"{test_metrics['test_accuracy']:.4f}",
                            'lr': scheduler.get_last_lr()[0]
                        })

                        # Save best model
                        if test_metrics['test_accuracy'] > best_accuracy:
                            best_accuracy = test_metrics['test_accuracy']
                            self.save_checkpoint(
                                save_dir, shard_number, epoch, global_step,
                                optimizer, scheduler, scaler, current_loss, test_metrics
                            )

                    # Clean up memory
                    del outputs, loss
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print('| WARNING: ran out of memory, skipping batch')
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e

            # End of epoch evaluation
            avg_train_loss = total_loss / len(train_loader)
            test_metrics = self.evaluate_batch(test_loader)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")
            print(f"Test loss: {test_metrics['test_loss']:.4f}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")

            torch.cuda.empty_cache()

        return metrics_history

    def evaluate_batch(self, test_loader):
        """Evaluate model on test data with improved error handling"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    # Get model outputs
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    total_loss += outputs.loss.item()

                    # Generate predictions
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=2,
                        num_beams=2,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        min_length=1
                    )

                    # Decode predictions and labels
                    predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                    true_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                    # Process predictions and update metrics
                    for pred, true in zip(predictions, true_labels):
                        try:
                            # Clean and validate prediction
                            pred = pred.strip()
                            if pred == '':
                                pred = '0'
                            elif pred.lower() in ['true', '1', 'positive']:
                                pred = '1'
                            elif pred.lower() in ['false', '0', 'negative']:
                                pred = '0'
                            else:
                                print(f"Warning: Unexpected prediction '{pred}', defaulting to '0'")
                                pred = '0'

                            # Clean and validate true label
                            true = true.strip()
                            if true == '':
                                print(f"Warning: Empty true label, skipping example")
                                continue

                            # Convert to binary values
                            pred_val = int(float(pred))
                            true_val = int(float(true))

                            # Update metrics
                            correct += (pred_val == true_val)
                            total += 1

                            # Store for detailed metrics
                            all_predictions.append(pred_val)
                            all_labels.append(true_val)

                        except (ValueError, TypeError) as e:
                            print(f"Warning: Error processing prediction '{pred}' or label '{true}': {e}")
                            continue

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print('| WARNING: ran out of memory during evaluation')
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

                # Clear cache after each batch
                torch.cuda.empty_cache()

        # Calculate metrics
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
        accuracy = correct / total if total > 0 else 0

        # Calculate additional metrics if we have predictions
        metrics = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'correct': correct,
            'total': total,
            'empty_predictions': len([p for p in predictions if p.strip() == '']),
            'invalid_predictions': len([p for p in predictions if p.strip() not in ['0', '1', 'true', 'false', '']])
        }

        return metrics
