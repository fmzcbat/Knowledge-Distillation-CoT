import pandas as pd
from sklearn.model_selection import train_test_split
import os


class DataLoader:
    def __init__(self, data_dir, test_size=0.1, random_state=42):
        """
        Initialize DataLoader

        Args:
            data_dir (str): Directory containing the CSV files
            test_size (float): Proportion of data to use for validation
            random_state (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state


    def _binarize_labels(self, labels):
        """
        Convert numeric labels to binary (0 or 1)
        First ensures labels are numeric, then applies binarization
        """
        # Convert to numeric, handle any errors
        numeric_labels = pd.to_numeric(labels, errors='coerce')

        # Check for any NaN values that resulted from conversion
        if numeric_labels.isna().any():
            print(f"Warning: {numeric_labels.isna().sum()} labels could not be converted to numbers")

        # Convert to binary
        return [1 if label >= 0.5 else 0 for label in numeric_labels]

    def read_shard(self, shard_number):
        """
        Read a single shard of data

        Args:
            shard_number (int): Shard number (0-4)

        Returns:
            tuple: (train_data, val_data) where each is a tuple of (questions, rationales, labels)
        """
        file_path = os.path.join(self.data_dir, f"train-{shard_number}.csv")

        try:
            # Read CSV file
            df = pd.read_csv(file_path, dtype={
                'question': str,
                'generated_rationale': str,
                'generated_solution': str
            })

            # Convert label column to numeric
            df['generated_solution'] = pd.to_numeric(df['generated_solution'], errors='coerce')

            # Check for any conversion issues
            if df['generated_solution'].isna().any():
                print(f"Warning: {df['generated_solution'].isna().sum()} labels were invalid and converted to NaN")
                # Drop rows with invalid labels
                df = df.dropna(subset=['generated_solution'])
                print(f"Dropped rows with invalid labels. Remaining rows: {len(df)}")

            # Convert to binary
            df['label'] = self._binarize_labels(df['generated_solution'])

            # Split into train and validation sets
            train_df, val_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state
            )

            # Prepare training data
            train_data = (
                train_df['question'].tolist(),
                train_df['generated_rationale'].tolist(),
                train_df['label'].tolist()
            )

            # Prepare validation data
            val_data = (
                val_df['question'].tolist(),
                val_df['generated_rationale'].tolist(),
                val_df['label'].tolist()
            )

            print(f"Shard {shard_number} loaded:")
            print(f"Training samples: {len(train_data[0])}")
            print(f"Validation samples: {len(val_data[0])}")

            return train_data, val_data

        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
            return None, None
        except Exception as e:
            print(f"Error loading shard {shard_number}: {str(e)}")
            return None, None

    def read_all_shards(self):
        """
        Read all shards and return a generator

        Yields:
            tuple: (shard_number, train_data, val_data)
        """
        for shard_number in range(5):  # 0 to 4
            train_data, val_data = self.read_shard(shard_number)
            if train_data is not None and val_data is not None:
                yield shard_number, train_data, val_data

    def get_shard_stats(self, shard_number):
        """
        Get statistics for a specific shard

        Args:
            shard_number (int): Shard number to analyze

        Returns:
            dict: Statistics about the shard
        """
        file_path = os.path.join(self.data_dir, f"train-{shard_number}.csv")

        try:
            df = pd.read_csv(file_path)
            stats = {
                'total_samples': len(df),
                'label_distribution': df['generated_solution'].value_counts().to_dict(),
                'avg_question_length': df['question'].str.len().mean(),
                'avg_rationale_length': df['generated_rationale'].str.len().mean(),
                'num_unique_labels': df['generated_solution'].nunique()
            }
            return stats
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
            return None
