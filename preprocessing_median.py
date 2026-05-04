import pandas as pd

# Load your dataset
df = pd.read_csv('data/cip_dose_response_raw.csv')

# Define the columns that uniquely identify an experiment and the specific well
group_cols = ['rep', 'dosisAb', 'dosis', 'Ab', 'medium', 'well']

# Sort the dataset by the grouping columns and a time component to ensure proper rolling calculation
df_sorted = df.sort_values(by=group_cols + ['UNIXTimestamp'])

# Define your rolling window size (e.g., 5 data points)
window_size = 5

# Calculate the rolling median
df_sorted['ODsmooth'] = (
    df_sorted.groupby(group_cols, dropna=False)['OD']
    .rolling(window=window_size, center=True, min_periods=1)
    .median()
    .reset_index(level=group_cols, drop=True)
)

# If you want to put the dataframe back into its original chronological order
df_sorted = df_sorted.sort_index()

# Save the updated dataframe to a new CSV file
df_sorted.to_csv('data/cip_dose_response_all.csv', index=False)

print("Smoothing complete. New file saved!")
