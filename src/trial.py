
# # read csv file
# with open('listings.csv', 'r', encoding='utf-8') as f:
#     reader = csv.reader(f)
#     data = list(reader)

# convert to pandas DataFrame
# df = pd.DataFrame(data[1:], columns=data[0])
# df.head(10)

# Read the CSV file directly into a DataFrame
import pandas as pd

df = pd.read_csv('data/raw/listings.csv', encoding='utf-8')

# Display the first 10 rows
print(df.head(10))
print(df.shape)
print(df.columns)
