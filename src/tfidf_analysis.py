import glob
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

files = glob.glob("data/raw/listings*.csv*")
listings = pd.concat([pd.read_csv(f, low_memory=False) for f in files])
listings = listings.dropna(subset=["description"])
listings = listings.drop_duplicates(subset=["description"])
listings = listings.groupby("host_id").head(5)

reviews = pd.read_csv("data/raw/reviews.csv")
reviews = reviews.drop(columns=["reviewer_id", "reviewer_name"])
reviews = reviews[reviews["date"] >= "2025-10-01"]
reviews = reviews.dropna(subset=["comments"])
reviews["quarter"] = pd.to_datetime(reviews["date"]).dt.to_period("Q").astype(str)
boroughs = listings[["id", "neighbourhood_group_cleansed"]].drop_duplicates("id")
reviews = reviews.merge(boroughs, left_on="listing_id", right_on="id")

names = ["nyc", "new", "york", "br", "manhattan", "brooklyn", "queens", "bronx", "staten", "island",
         "stuy", "lga", "rumc"]
for name in listings["neighbourhood_cleansed"].unique():
    names += re.findall(r"[a-z]+", name.lower())
for name in listings["host_name"].dropna().unique():
    names += re.findall(r"[a-z]+", name.lower())
stop_words = list(ENGLISH_STOP_WORDS.union(names))


def top_words(df, text, group):
    tfidf = TfidfVectorizer(stop_words=stop_words, min_df=20, token_pattern=r"\b[a-z]{3,}\b")
    X = tfidf.fit_transform(df[text].str.lower())
    words = tfidf.get_feature_names_out()
    for g in sorted(df[group].unique()):
        in_group = (df[group] == g).values
        diff = np.asarray(X[in_group].mean(axis=0) - X[~in_group].mean(axis=0)).ravel()
        print(g, ":", ", ".join(words[diff.argsort()[::-1][:10]]))
    print()


top_words(listings, "description", "neighbourhood_group_cleansed")
top_words(reviews, "comments", "neighbourhood_group_cleansed")
top_words(reviews, "comments", "quarter")
