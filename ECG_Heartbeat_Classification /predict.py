import pandas as pd
from sklearn.model_selection import train_test_split 

# Load the test data
x_test = pd.read_csv("data/processed/x_test.csv", header=None)
y_test = pd.read_csv("data/processed/y_test.csv", header=None)

# Check the shape of the loaded data
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Split the test set into holdout and validation sets (50-50 split)
x_val, x_holdout, y_val, y_holdout = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

print(f"Validation set size: {x_val.shape[0]}")
print(f"Holdout set size: {x_holdout.shape[0]}")