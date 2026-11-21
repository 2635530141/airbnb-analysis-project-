import pandas as pd

# Load data
mar = pd.read_csv("data/raw/listings_mar.csv")
jun = pd.read_csv("data/raw/listings_jun.csv")
sep = pd.read_csv("data/raw/listings_sep.csv")

# Combine data
df = pd.concat([mar, jun, sep], ignore_index=True)

# Filter listings
df = df[
    (df["host_identity_verified"] == "t") &
    (df["price_quote_price_per_night"].notna()) &
    (df["has_availability"] == "t") &
    ((df["estimated_occupancy_l365d"] != 0) |
     (df["availability_365"] != 0)) &
    (df["number_of_reviews"] > 0)
]

# Select columns
cols = ["id", "neighbourhood_group_cleansed",
        "price_quote_price_per_night", "review_scores_rating",
        "calendar_last_scraped"]

df = df[cols].drop_duplicates()

# Calculate average price
avg_price = df.groupby("id")["price_quote_price_per_night"].mean()

# Keep the most recent listing
df = df.sort_values("calendar_last_scraped")
df = df.drop_duplicates("id", keep="last").copy()
df["ave_price_quote_price_per_night"] = df["id"].map(avg_price)

# Save data
df.to_csv("data/processed/listings_combined_cleaned.csv", index=False)
print(df.shape)

