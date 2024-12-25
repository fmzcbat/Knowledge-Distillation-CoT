from t5_generation import T5Labeler
from datasets import load_dataset
from cot_train import T5SentimentAnalyzerCustom
from datetime import datetime
from write_to_csv import append_results_to_csv

# Initialize model
analyzer = T5SentimentAnalyzerCustom()

# Generate for test set
test_dataset = load_dataset("stanfordnlp/imdb")["test"]
test_shard_start_idx = shard_idx = 0
total_test_shard_num = 25
cur_shard = test_dataset.shuffle(seed=42).shard(total_test_shard_num, shard_idx)

# run_id = 0 # shard_0_checkpoints/checkpoint_epoch_2.pt
# run_id = 1 # shard_1_checkpoints/checkpoint_epoch_2.pt
# run_id = 2 # shard_2_checkpoints/checkpoint_epoch_1.pt
# run_id = 3 # shard_3_checkpoints/checkpoint_epoch_2.pt
run_id = 4 # shard_4_checkpoints/checkpoint_epoch_0.pt
analyzer.load_checkpoint("feedback_models/shard_4_checkpoints/checkpoint_epoch_0.pt")

t5_labeler = T5Labeler(analyzer.tokenizer, analyzer.model)

print(f"Processing test dataset shard {shard_idx}")
score = t5_labeler.process_dataset(cur_shard)
print(f"Final score for shard-{shard_idx}: {score}")


time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
row = [run_id, time_stamp, score]
append_results_to_csv("score_fe_cot.csv", row)