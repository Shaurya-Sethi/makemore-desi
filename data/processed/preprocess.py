import pandas as pd
import re

# Path to the existing CSV dataset
input_csv = "data/raw/indian_first_names.csv"
output_csv = "data/processed/indian_first_names_cleaned.csv"

# 1. Load the dataset
df = pd.read_csv(input_csv)

# 2. Identify and drop missing values in the 'name' column
df.dropna(subset=["name"], inplace=True)

# 3. Define a function to extract the first name (the first sequence of letters)
def extract_first_name(name):
    """
    Given a full name, returns the first sequence of alphabetic characters.
    This effectively removes surnames and any special characters.
    """
    name = name.strip()  # remove leading/trailing whitespace
    # Use regex to match the first sequence of letters (upper or lower case)
    match = re.match(r"([A-Za-z]+)", name)
    if match:
        return match.group(1)
    else:
        return name  # fallback if no match

# 4. Apply the function to the 'name' column
df["name"] = df["name"].apply(extract_first_name)

# 5. Remove any duplicate names
df.drop_duplicates(subset=["name"], inplace=True)

# 6. Save the cleaned dataset to a new CSV file
df.to_csv(output_csv, index=False)
print(f"Cleaned dataset saved to {output_csv}")
