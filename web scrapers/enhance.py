# The data crawled using firecrawl needs to be added to the dataset

import json
import pandas as pd

# Define file paths
json_file = "data/firecrawl_names.json"
csv_file = "data/indian_first_names.csv"

# 1. Load the existing dataset (if it exists) into a set
try:
    df_existing = pd.read_csv(csv_file)
    existing_names = set(df_existing['name'].astype(str))
    print(f"Loaded {len(existing_names)} names from the existing CSV.")
except FileNotFoundError:
    print("Existing CSV not found. Starting with an empty dataset.")
    existing_names = set()

# 2. Read the JSON file and extract the list of names
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# The JSON structure is a list with one dictionary; names are under "indian_forenames"
new_names = data[0].get("indian_forenames", [])
print(f"Found {len(new_names)} names in the JSON file.")

# 3. Filter names: only add names that are 3 or more characters long and not already in existing_names
filtered_names = {name for name in new_names if len(name) >= 3 and name not in existing_names}
print(f"{len(filtered_names)} new names will be added to the dataset.")

# 4. Merge the new names with the existing ones
updated_names = existing_names.union(filtered_names)

# 5. Save the updated dataset back to CSV
df_updated = pd.DataFrame({"name": sorted(updated_names)})
df_updated.to_csv(csv_file, index=False)
print(f"Updated CSV saved to {csv_file}. Total names now: {len(updated_names)}")
