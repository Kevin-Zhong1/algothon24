import pandas as pd

# Step 1: Read CSV with a single column, no headers
df_raw = pd.read_csv("./prices.csv", header=None)

# Step 2: Split each space-separated row into 50 separate columns
df = df_raw[0].str.split(expand=True)

# Step 3: Convert all values to float (optional but recommended)
df = df.astype(float)

# Step 4: Rename stock columns as STOCK1 ... STOCK50
df.columns = [f"STOCK{i+1}" for i in range(df.shape[1])]

# Step 5: Create date range from 1/1/2000, daily
dates = pd.date_range(start="2000-01-01", periods=df.shape[0], freq="D")

# Step 6: Insert dates column to the left
df.insert(0, "dates", dates)

# * Step 7: Keep only the first 400 rows, change this
# df = df.head(400)

# (Optional) Save to CSV
df.to_csv("temp.csv", index=False)

# Preview
print(df.head())
