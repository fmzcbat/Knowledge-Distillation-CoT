import csv
import os


def initialize_csv(filename, headers):
    """
    Create CSV file with headers if it doesn't exist

    Args:
        filename (str): Name of the CSV file
        headers (list): List of column headers
    """
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def append_results_to_csv(filename, results):
    """
    Append a row of results to the CSV file

    Args:
        filename (str): Name of the CSV file
        results (list): List of values to write as a row
    """
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results)


initialize_csv("score_original.csv", ["run_id", "time", "result"])