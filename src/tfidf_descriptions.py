"""TF-IDF analysis of NYC Airbnb listing descriptions, Dec 2025 - Sep 2026.

Questions
---------
1. What language do hosts use to market listings in each borough, and does
   it line up with price differences?
2. Is that language stable across the year's four quarterly snapshots?
3. How does city-wide listing language change from quarter to quarter?

Data
----
Every Inside Airbnb NYC listings file in data/raw/ (``listings*.csv`` or
``listings*.csv.gz``). The snapshot date is read from ``last_scraped``, so
file names don't matter.

Method
------
1. Cleaning (same for every quarter): verified host, has availability, and
   active (occupancy or availability in the last year). We do NOT require a
   price here, because the Dec 2025 snapshot has no price data at all.
2. Drop duplicate descriptions within each snapshot, and keep at most
   MAX_PER_HOST (5) descriptions per host per snapshot, so large multi-listing
   hosts can't dominate the results. For the pooled-year analysis, keep one
   copy of each description, since most listings appear in several snapshots.
3. One TF-IDF vocabulary fitted on the pooled year (unigrams + bigrams,
   English stop words and place names removed, terms in < 20 listings
   dropped), so scores are comparable between quarters.
4. Distinctiveness of a term for a group =
       mean TF-IDF inside the group - mean TF-IDF outside the group.
   Groups are boroughs (Q1, Q2) or snapshots (Q3).

Run from the repo root:
    python src/tfidf_descriptions.py
Outputs go to outputs/.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TOP_N = 10
MAX_PER_HOST = 5   # max descriptions any one host contributes per snapshot
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
COLS = ["id", "host_id", "description", "neighbourhood_cleansed", "neighbourhood_group_cleansed",
        "host_identity_verified", "has_availability", "estimated_occupancy_l365d",
        "availability_365", "last_scraped", "price", "price_quote_price_per_night"]


# ---------------------------------------------------------------- load
def load_snapshot(path):
    df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    df["snapshot"] = pd.to_datetime(df["last_scraped"]).min().strftime("%Y-%m")
    # Newer files have price_quote_price_per_night; fall back to "$1,234" price
    if "price_quote_price_per_night" in df:
        df["price_num"] = pd.to_numeric(df["price_quote_price_per_night"], errors="coerce")
    else:
        df["price_num"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    return df


files = sorted(p for p in RAW.glob("listings*") if p.name.endswith((".csv", ".csv.gz")))
data = pd.concat([load_snapshot(p) for p in files], ignore_index=True)
# Keep one file per snapshot date, in case the same snapshot was saved twice
data = data.drop_duplicates(subset=["snapshot", "id"])
SNAPSHOTS = sorted(data["snapshot"].unique())
print("Snapshots:", ", ".join(SNAPSHOTS))

# ---------------------------------------------------------------- clean
keep = ((data["host_identity_verified"] == "t")
        & (data["has_availability"] == "t")
        & ((data["estimated_occupancy_l365d"] != 0) | (data["availability_365"] != 0))
        & data["description"].notna()
        & data["neighbourhood_group_cleansed"].isin(BOROUGHS))
data = data[keep].copy()
data["text"] = data["description"].str.replace(r"<[^>]+>", " ", regex=True).str.lower()

per_q = data.drop_duplicates(subset=["snapshot", "description"])
# Large multi-listing hosts (e.g. one corporate host with ~1,000 Manhattan units)
# would otherwise dominate a borough's or quarter's vocabulary. That host's
# host_identity_verified flag also flips between snapshots, so without a cap it
# enters and leaves the cleaned data and fakes a change over time.
per_q = (per_q.sample(frac=1, random_state=0)
         .groupby(["snapshot", "host_id"]).head(MAX_PER_HOST)
         .reset_index(drop=True))
print("Unique descriptions per snapshot:", per_q["snapshot"].value_counts().sort_index().to_dict())

# ---------------------------------------------------------------- stop words
place_words = set()
for name in pd.concat([data["neighbourhood_cleansed"], data["neighbourhood_group_cleansed"]]).unique():
    place_words.update(re.findall(r"[a-z]+", str(name).lower()))
place_words.update({"nyc", "new", "york", "city", "ny", "manhattan", "brooklyn",
                    "queens", "bronx", "staten", "island", "jfk", "lga",
                    "laguardia", "airport", "midtown", "downtown", "uptown",
                    "stuy", "bk", "les", "ues", "uws", "soho", "rumc"})
stop_words = list(ENGLISH_STOP_WORDS | place_words)

# ---------------------------------------------------------------- TF-IDF
vec = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), min_df=20,
                      sublinear_tf=True, token_pattern=r"(?u)\b[a-z]{3,}\b")
year = per_q.drop_duplicates(subset="description")       # pooled year, one copy each
vec.fit(year["text"])
terms = vec.get_feature_names_out()
X_q = vec.transform(per_q["text"])
print(f"Vocabulary: {len(terms):,} terms; pooled year: {len(year):,} unique descriptions")


def distinctive(X, labels, group, n=TOP_N):
    inside = labels == group
    diff = (X[inside].mean(axis=0) - X[~inside].mean(axis=0)).A1
    idx = diff.argsort()[::-1][:n]
    return [(terms[i], float(diff[i])) for i in idx]


# Q1: borough terms, pooled year
X_year = vec.transform(year["text"])
b_year = year["neighbourhood_group_cleansed"].to_numpy()
rows = [{"borough": b, "rank": r, "term": t, "distinctiveness": round(s, 4)}
        for b in BOROUGHS for r, (t, s) in enumerate(distinctive(X_year, b_year, b), 1)]
top_year = pd.DataFrame(rows)
top_year.to_csv(OUT / "tfidf_top_terms_by_borough.csv", index=False)

# Q2: borough terms within each quarter + overlap with the yearly top 10
rows, stab = [], []
for q in SNAPSHOTS:
    m = (per_q["snapshot"] == q).to_numpy()
    labels = per_q.loc[m, "neighbourhood_group_cleansed"].to_numpy()
    for b in BOROUGHS:
        top_q = distinctive(X_q[m], labels, b)
        rows += [{"snapshot": q, "borough": b, "rank": r, "term": t,
                  "distinctiveness": round(s, 4)} for r, (t, s) in enumerate(top_q, 1)]
        yearly = set(top_year.loc[top_year["borough"] == b, "term"])
        stab.append({"snapshot": q, "borough": b,
                     "overlap_with_year_top10": len(yearly & {t for t, _ in top_q})})
pd.DataFrame(rows).to_csv(OUT / "tfidf_top_terms_by_borough_quarter.csv", index=False)
stability = pd.DataFrame(stab).pivot(index="borough", columns="snapshot",
                                     values="overlap_with_year_top10").reindex(BOROUGHS)
stability.to_csv(OUT / "tfidf_borough_term_stability.csv")
print("\nOverlap of each quarter's top 10 with the yearly top 10:\n", stability)

# Q3: city-wide terms distinctive to each quarter
snap = per_q["snapshot"].to_numpy()
rows = [{"snapshot": q, "rank": r, "term": t, "distinctiveness": round(s, 4)}
        for q in SNAPSHOTS for r, (t, s) in enumerate(distinctive(X_q, snap, q), 1)]
top_quarter = pd.DataFrame(rows)
top_quarter.to_csv(OUT / "tfidf_top_terms_by_quarter.csv", index=False)
print("\nDistinctive terms by quarter:\n",
      top_quarter.groupby("snapshot")["term"].apply(lambda s: ", ".join(s.head(8))).to_string())

# Median quoted price by borough x quarter (Dec 2025 has no prices)
prices = (data.dropna(subset=["price_num"])
          .pivot_table(index="neighbourhood_group_cleansed", columns="snapshot",
                       values="price_num", aggfunc="median")
          .reindex(index=BOROUGHS, columns=SNAPSHOTS).round(0))
prices.to_csv(OUT / "median_price_by_borough_quarter.csv")
print("\nMedian quoted price per night:\n", prices)

summary = (top_year.groupby("borough")["term"].apply(lambda s: ", ".join(s.head(6)))
           .reindex(BOROUGHS).to_frame("top_terms_year")
           .join(year["neighbourhood_group_cleansed"].value_counts().rename("unique_descriptions"))
           .join(prices.add_prefix("median_price_")))
summary.to_csv(OUT / "tfidf_summary_table.csv")

# ---------------------------------------------------------------- figures
INK, MUTED, BAR, GRID = "#1f2328", "#6e7781", "#3b6fb6", "#eaeef2"
latest_price = prices.ffill(axis=1).iloc[:, -1]


def style(ax):
    ax.tick_params(axis="y", labelsize=9.5, colors=INK, length=0)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d0d7de")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 5, figsize=(16, 4.8))
for ax, b in zip(axes, BOROUGHS):
    d = top_year[top_year["borough"] == b].iloc[::-1]
    ax.barh(d["term"], d["distinctiveness"], color=BAR, height=0.6)
    stable = int(stability.loc[b].min())
    ax.set_title(f"{b}\n${latest_price[b]:.0f}/night · {stable}+/10 stable",
                 fontsize=10.5, color=INK, loc="left")
    style(ax)
fig.suptitle("Most distinctive words in listing descriptions by borough, Dec 2025 – Sep 2026 (TF-IDF)",
             fontsize=13, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.01, f"Score = mean TF-IDF in borough minus mean in other boroughs. "
         f"{len(year):,} unique descriptions from 4 quarterly snapshots (max 5 per host per snapshot); place names removed. "
         f"Median price from {SNAPSHOTS[-1]} snapshot. Source: Inside Airbnb.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(OUT / "tfidf_top_terms_by_borough.png", dpi=200)

fig, axes = plt.subplots(1, len(SNAPSHOTS), figsize=(13, 4.4))
for ax, q in zip(axes, SNAPSHOTS):
    d = top_quarter[top_quarter["snapshot"] == q].iloc[::-1]
    ax.barh(d["term"], d["distinctiveness"], color=BAR, height=0.6)
    ax.set_title(pd.Period(q).strftime("%b %Y"), fontsize=11, color=INK, loc="left")
    style(ax)
fig.suptitle("Words that distinguish each quarter's listing descriptions (all NYC, TF-IDF)",
             fontsize=13, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.01, "Score = mean TF-IDF in snapshot minus mean in the other snapshots. Source: Inside Airbnb.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(OUT / "tfidf_top_terms_by_quarter.png", dpi=200)
print("\nSaved figures to outputs/")
