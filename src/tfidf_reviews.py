"""TF-IDF analysis of NYC Airbnb guest reviews, Oct 2025 - Sep 2026.

Questions
---------
1. What do guests talk about in each borough?
2. How does review language change across the four quarters of the year?

Data
----
- data/raw/reviews.csv: the full-text reviews file from the Sep 2026 Inside
  Airbnb snapshot. It contains every review ever posted, with its date, so one
  file covers the whole year.
- data/raw/listings*: all listings snapshots, used only to look up each
  listing's borough and the host first names.

Method
------
1. Keep reviews dated 2025-10-01 to 2026-09-14 (four quarters).
2. Privacy: drop reviewer_name and reviewer_id immediately, and remove host
   first names from the text so they don't show up as "distinctive" words.
3. Keep mostly-English reviews (at least 20% of words are common English
   stop words) and reviews of at least 5 words.
4. At most MAX_PER_LISTING reviews per listing per quarter, so a handful of
   very busy listings can't dominate.
5. TF-IDF (unigrams + bigrams, English stop words and place names removed,
   terms in < 30 reviews dropped). Distinctiveness of a term for a group =
   mean TF-IDF inside the group - mean TF-IDF outside it.

Run from the repo root:
    python src/tfidf_reviews.py
Outputs go to outputs/.
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TOP_N = 10
MAX_PER_LISTING = 10
START, END = "2025-10-01", "2026-09-14"
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

# ---------------------------------------------------------------- listings lookup
lcols = ["id", "host_name", "neighbourhood_cleansed", "neighbourhood_group_cleansed"]
files = sorted(p for p in RAW.glob("listings*") if p.name.endswith((".csv", ".csv.gz")))
listings = pd.concat([pd.read_csv(p, usecols=lcols, low_memory=False) for p in files])
listings = listings.drop_duplicates("id", keep="last")

# ---------------------------------------------------------------- reviews
rev = pd.read_csv(RAW / "reviews.csv", usecols=["listing_id", "date", "comments"])
rev["date"] = pd.to_datetime(rev["date"])
rev = rev[(rev["date"] >= START) & (rev["date"] <= END)].dropna(subset=["comments"])
rev["quarter"] = rev["date"].dt.to_period("Q").astype(str)
rev = rev.merge(listings, left_on="listing_id", right_on="id", how="inner")
rev = rev[rev["neighbourhood_group_cleansed"].isin(BOROUGHS)]
print(f"Reviews in window with a known borough: {len(rev):,}")

rev["text"] = (rev["comments"].str.replace(r"<[^>]+>", " ", regex=True)
               .str.lower())
tokens = rev["text"].str.findall(r"[a-z']+")
stop = set(ENGLISH_STOP_WORDS)
eng_share = tokens.map(lambda t: sum(w in stop for w in t) / max(len(t), 1))
rev = rev[(eng_share >= 0.2) & (tokens.str.len() >= 5)]
rev = (rev.sample(frac=1, random_state=0)
       .groupby(["quarter", "listing_id"]).head(MAX_PER_LISTING)
       .reset_index(drop=True))
print(f"After English filter and per-listing cap: {len(rev):,}")
print(rev["quarter"].value_counts().sort_index().to_dict())

# ---------------------------------------------------------------- stop words
place_words = set()
for name in pd.concat([listings["neighbourhood_cleansed"],
                       listings["neighbourhood_group_cleansed"]]).dropna().unique():
    place_words.update(re.findall(r"[a-z]+", name.lower()))
place_words.update({"nyc", "new", "york", "city", "ny", "manhattan", "brooklyn",
                    "queens", "bronx", "staten", "island", "jfk", "lga",
                    "laguardia", "airport", "midtown", "downtown", "uptown",
                    "stuy", "bk", "les", "ues", "uws", "soho"})
host_names = set()
for n in listings["host_name"].dropna().unique():
    host_names.update(re.findall(r"[a-z]+", n.lower()))
# Keep ordinary English words that also happen to be host names
host_names -= {"great", "home", "stay", "place", "room", "rooms", "host", "hosts",
               "clean", "nice", "good", "best", "city", "park", "love", "family",
               "studio", "apartment", "apartments", "suites", "house", "comfortable",
               "quiet", "cozy", "location", "space", "beautiful", "bed", "bath",
               "subway", "train", "parking", "bus", "ferry", "view", "views",
               "stadium", "yankee", "zoo", "garden", "hospital", "medical",
               "safe", "clean", "easy", "friendly", "helpful", "responsive"}
stop_words = list(stop | place_words | host_names)

vec = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), min_df=30,
                      sublinear_tf=True, token_pattern=r"(?u)\b[a-z]{3,}\b")
X = vec.fit_transform(rev["text"])
terms = vec.get_feature_names_out()
print(f"Vocabulary: {len(terms):,} terms")


def distinctive(labels, group, n=TOP_N):
    inside = labels == group
    diff = (X[inside].mean(axis=0) - X[~inside].mean(axis=0)).A1
    idx = diff.argsort()[::-1][:n]
    return [(terms[i], round(float(diff[i]), 4)) for i in idx]


boro = rev["neighbourhood_group_cleansed"].to_numpy()
qtr = rev["quarter"].to_numpy()
QUARTERS = sorted(rev["quarter"].unique())

top_b = pd.DataFrame([{"borough": b, "rank": r, "term": t, "distinctiveness": s}
                      for b in BOROUGHS for r, (t, s) in enumerate(distinctive(boro, b), 1)])
top_q = pd.DataFrame([{"quarter": q, "rank": r, "term": t, "distinctiveness": s}
                      for q in QUARTERS for r, (t, s) in enumerate(distinctive(qtr, q), 1)])
top_b.to_csv(OUT / "reviews_tfidf_top_terms_by_borough.csv", index=False)
top_q.to_csv(OUT / "reviews_tfidf_top_terms_by_quarter.csv", index=False)
counts = rev.groupby(["neighbourhood_group_cleansed", "quarter"]).size().unstack().reindex(BOROUGHS)
counts.to_csv(OUT / "reviews_count_by_borough_quarter.csv")
print("\nBorough:\n", top_b.groupby("borough", sort=False)["term"].apply(", ".join).to_string())
print("\nQuarter:\n", top_q.groupby("quarter")["term"].apply(", ".join).to_string())
print("\nSaved tables to outputs/")
