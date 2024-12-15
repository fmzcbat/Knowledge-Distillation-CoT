from data_loader import DataLoader as CustomizedDataLoader
from transformers import logging
from cot_train import T5SentimentAnalyzerCustom

data_loader = CustomizedDataLoader(
    data_dir="training_data_shuffled",
    test_size=0.1  # 10% for validation
)

logging.set_verbosity_error()

# Initialize model
analyzer = T5SentimentAnalyzerCustom()

# Train on each shard
for shard_number, train_data, val_data in data_loader.read_all_shards():
    print(f"\nTraining on shard {shard_number}")

    # Train the model
    metrics = analyzer.train_shard(
        train_data=train_data,
        test_data=val_data,
        shard_number=shard_number,
        save_dir="sentiment_models_cot",
        batch_size=2,
        gradient_accumulation_steps=8,
        epochs=3,
        checkpoint_steps=200
    )

    print(f"Completed training on shard {shard_number}")
    print("Final metrics:", metrics)