import json
from collections import defaultdict
import glob

def combine_journal_stats(input_dir, output_file):
    """
    Combine multiple journal stats files into one, aggregating counts and unifying Category_Counts.

    Parameters:
        input_dir (str): Directory containing the journal_tag_stats_i.jsonl files.
        output_file (str): Output file path to store the concatenated JSONL.
    """
    # Create a dictionary to store combined stats
    combined_stats = defaultdict(lambda: {
        "Journal_Count": 0,
        "AI_Count": 0,
        "Category_Counts": defaultdict(int)
    })

    # Loop over all files in the input directory
    for file_path in glob.glob(f"{input_dir}/journal_tag_stats_*.jsonl"):
        print(f"Processing file: {file_path}")
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                issn_eissn = data.get("ISSN_EISSN")
                year = data.get("year")

                if not issn_eissn or not year:
                    continue  # Skip invalid entries without ISSN_EISSN or year

                # Initialize the record for this ISSN_EISSN and year if not present
                key = (issn_eissn, year)

                # Aggregate Journal_Count and AI_Count
                combined_stats[key]["Journal_Count"] += data.get("Journal_Count", 0)
                combined_stats[key]["AI_Count"] += data.get("AI_Count", 0)

                # Merge Category_Counts
                for category, count in data.get("Category_Counts", {}).items():
                    combined_stats[key]["Category_Counts"][category] += count

    # Write the combined stats to the output JSONL file
    with open(output_file, "w") as f:
        for (issn_eissn, year), stats in combined_stats.items():
            stats_data = {
                "ISSN_EISSN": issn_eissn,
                "year": year,
                "Journal_Count": stats["Journal_Count"],
                "AI_Count": stats["AI_Count"],
                "Category_Counts": dict(stats["Category_Counts"])  # Convert to a regular dict for JSON compatibility
            }
            json.dump(stats_data, f)
            f.write("\n")

    print(f"Combined stats saved to {output_file}")

# Example usage
input_dir = "/users/rwan388/result"  # Replace with the directory containing your journal_tag_stats_*.jsonl files
concatenate_stat_output_file = "/users/rwan388/journal_result/concatenated_journal_stats.jsonl"
combine_journal_stats(input_dir, concatenate_stat_output_file)


import pandas as pd

def calculate_ai_percentage(input_file, output_file):
    """
    Calculate the AI percentage (AI_Perc) for each ISSN_EISSN and year, and save the result in an xsc file.

    Parameters:
        input_file (str): The JSONL file containing combined journal stats.
        output_file (str): Output file path to save the AI percentages as an xsc file.
    """
    ai_percentage_data = []

    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            issn_eissn = data.get("ISSN_EISSN")
            year = data.get("year")
            journal_count = data.get("Journal_Count", 0)
            ai_count = data.get("AI_Count", 0)

            # Calculate AI percentage if Journal_Count is not zero
            if journal_count > 0:
                ai_perc = ai_count / journal_count
            else:
                ai_perc = 0.0  # If there are no journals, set AI_Perc to 0

            ai_percentage_data.append({
                "ISSN_EISSN": issn_eissn,
                "year": year,
                "AI_Perc": ai_perc
            })

    # Convert the data to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(ai_percentage_data)

    # Save the data as an xsc file (e.g., CSV or Excel format)
    df.to_csv(output_file, index=False)  # Change to .xlsx if you want an Excel file
    print(f"AI percentages saved to {output_file}")

# Example usage
ai_percentage_output = "/users/rwan388/journal_result/ai_percentage.csv"
calculate_ai_percentage(concatenate_stat_output_file, ai_percentage_output)
