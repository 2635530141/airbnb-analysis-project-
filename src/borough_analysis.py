import pandas as pd

# Load data
listings = pd.read_csv("data/processed/listings_combined_cleaned.csv")
reviews = pd.read_csv("data/raw/reviews.csv")

# Filter reviews by date
reviews = reviews[reviews["date"].between("2025-10-01", "2026-09-14")]
reviews = reviews.dropna(subset=["comments"])
reviews = reviews.rename(columns={"id": "review_id"})

# Merge reviews with listings
df = reviews.merge(listings, left_on="listing_id", right_on="id")

# Randomly select 5 reviews from each borough
sample = df.groupby("neighbourhood_group_cleansed").sample(n=5, random_state=42)

# Select columns
cols = ["neighbourhood_group_cleansed", "listing_id", "review_id",
        "comments", "ave_price_quote_price_per_night", "review_scores_rating"]

sample = sample[cols].copy()

# Add labeling columns
sample["topic"] = ""
sample["sentiment"] = ""

# Save data
sample.to_csv("outputs/borough_25_reviews.csv", index=False, encoding="utf-8-sig")
print(sample.groupby("neighbourhood_group_cleansed").size())


