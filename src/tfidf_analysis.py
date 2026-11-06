# TF-IDF analysis of NYC Airbnb listing descriptions and guest reviews
# Data: Inside Airbnb NYC, 4 quarterly snapshots (Dec 2025 - Sep 2026)
#
# Put the 4 listings files and reviews.csv in data/raw/, then run:
#   python src/tfidf_analysis.py

import glob
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]


# ---------- load the 4 listings snapshots ----------
files = glob.glob("data/raw/listings*.csv*")
listings = pd.concat([pd.read_csv(f, low_memory=False) for f in files])
listings["quarter"] = pd.to_datetime(listings["last_scraped"]).dt.to_period("Q").astype(str)

# drop listings with no description and copy-pasted descriptions
listings = listings.dropna(subset=["description"])
listings = listings.drop_duplicates(subset=["quarter", "description"])

# keep max 5 listings per host per quarter so big companies don't take over
listings = listings.groupby(["quarter", "host_id"]).head(5)


# ---------- load reviews (last 12 months) ----------
reviews = pd.read_csv("data/raw/reviews.csv")
reviews = reviews.drop(columns=["reviewer_id", "reviewer_name"])  # privacy
reviews["date"] = pd.to_datetime(reviews["date"])
reviews = reviews[reviews["date"] >= "2025-10-01"].dropna(subset=["comments"])
reviews["quarter"] = reviews["date"].dt.to_period("Q").astype(str)

# add the borough from the listings data
boroughs = listings[["id", "neighbourhood_group_cleansed"]].drop_duplicates("id")
reviews = reviews.merge(boroughs, left_on="listing_id", right_on="id")


# ---------- stop words ----------
# remove place names and host names so the results show how places are described
extra = ["nyc", "new", "york", "city", "br", "jfk", "lga", "airport", "midtown",
         "downtown", "uptown", "staten", "island", "bronx", "uws", "ues", "les",
         "stuy", "bedstuy", "soho", "hell", "kitchen", "rumc", "siuh"]
for name in listings["neighbourhood_cleansed"].unique():
    extra += re.findall(r"[a-z]+", name.lower())
for name in listings["host_name"].dropna().unique():
    extra += re.findall(r"[a-z]+", name.lower())
# some host names are also normal words, so keep these
keep = ["home", "place", "stay", "clean", "nice", "great", "cozy", "quiet", "park",
        "garden", "ferry", "subway", "train", "parking", "view", "location", "staff"]
extra = [w for w in extra if w not in keep]
stop_words = list(ENGLISH_STOP_WORDS.union(extra))


# ---------- TF-IDF ----------
def top_words(df, text_col, group_col, n=10):
    # TF-IDF score for every word in every listing/review
    tfidf = TfidfVectorizer(stop_words=stop_words, min_df=20, sublinear_tf=True,
                            token_pattern=r"\b[a-z]{3,}\b")
    X = tfidf.fit_transform(df[text_col].str.lower())
    words = tfidf.get_feature_names_out()

    # a word is distinctive for a group if its average score in that group
    # is much higher than its average score everywhere else
    results = {}
    for group in sorted(df[group_col].unique()):
        in_group = (df[group_col] == group).values
        diff = X[in_group].mean(axis=0) - X[~in_group].mean(axis=0)
        best = np.asarray(diff).ravel().argsort()[::-1][:n]
        results[group] = ", ".join(words[best])
    return pd.Series(results, name="top_words")


desc_borough = top_words(listings, "description", "neighbourhood_group_cleansed").reindex(BOROUGHS)
desc_quarter = top_words(listings, "description", "quarter")

# median price per night by borough and quarter (the Dec 2025 file has no prices)
prices = listings.pivot_table(index="neighbourhood_group_cleansed", columns="quarter",
                              values="price_quote_price_per_night", aggfunc="median")
prices = prices.reindex(BOROUGHS).round(0)
rev_borough = top_words(reviews, "comments", "neighbourhood_group_cleansed").reindex(BOROUGHS)
rev_quarter = top_words(reviews, "comments", "quarter")

print("LISTING DESCRIPTIONS BY BOROUGH\n", desc_borough.to_string(), "\n")
print("LISTING DESCRIPTIONS BY QUARTER\n", desc_quarter.to_string(), "\n")
print("REVIEWS BY BOROUGH\n", rev_borough.to_string(), "\n")
print("REVIEWS BY QUARTER\n", rev_quarter.to_string(), "\n")
print("MEDIAN PRICE PER NIGHT\n", prices.to_string())

# ---------- save ----------
desc_borough.to_csv("outputs/descriptions_by_borough.csv")
desc_quarter.to_csv("outputs/descriptions_by_quarter.csv")
rev_borough.to_csv("outputs/reviews_by_borough.csv")
rev_quarter.to_csv("outputs/reviews_by_quarter.csv")
prices.to_csv("outputs/median_price_by_borough.csv")
