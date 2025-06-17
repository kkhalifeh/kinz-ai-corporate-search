import pandas as pd

# Read the CSV file with UTF-8 encoding
df = pd.read_csv('partner_nationality.csv', encoding='utf-8')

# Combine name_en and name_ar into a new column with a separator
df['combined_name'] = df['name_en'] + ' | ' + df['name_ar']

# Save the updated CSV with UTF-8 encoding
# utf-8-sig adds BOM for Excel compatibility
df.to_csv('partner_nationalities.csv', index=False, encoding='utf-8-sig')

print("Updated CSV saved as partner_nationalities.csv")
