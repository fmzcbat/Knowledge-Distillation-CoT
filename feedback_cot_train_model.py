from data_loader import DataLoader as CustomizedDataLoader
from feedback_enhanced import T5StudentTrainerWithFeedback

data_loader = CustomizedDataLoader(
    data_dir="training_data_shuffled",
    test_size=0.1  # 10% for validation
)

# Initialize the feedback-enhanced model
model = T5StudentTrainerWithFeedback(
    model_name="t5-base",
    feedback_model_name="google/gemma-1.1-2b-it"
)

# Train on 3 shards
# for shard_number in range(3):

# Train the last 2 shards
for shard_number in range(3, 5):
    train_data, test_data = data_loader.read_shard(shard_number)
    print(f"\nTraining on shard {shard_number}")
    # Train on shard
    shard_metrics = model.train_with_feedback(
        train_data,
        test_data,
        batch_size=2,
        epochs=3,
        shard_number=shard_number,
        feedback_frequency=10,  # Get feedback every 10 batches
        save_dir='feedback_models'
    )
    print(f"Completed training on shard {shard_number}")
    print("Final metrics:", shard_metrics)
