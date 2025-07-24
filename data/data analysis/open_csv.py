import pandas as pd

# Path to your .csv file
csv_file_path = '/Users/ratnamb.ojha/Downloads/uber.csv.zip'  # Replace with the actual path to your .csv file

# Read the .csv file
data = pd.read_csv(csv_file_path)

# Display the first few rows of the data
print(data.head())