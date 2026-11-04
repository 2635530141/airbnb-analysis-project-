"""TF-IDF analysis of NYC Airbnb listing descriptions by borough.

Question: what language do hosts use to market listings in each borough,
and does it line up with differences in price?

Method
------
1. Apply the team's cleaning rules (verified hosts, non-missing price,
   available, active listings).
2. Drop duplicate descriptions (templates reused by multi-listing hosts).
3. Build one TF-IDF vector per listing description (unigrams + bigrams,
   English stop words, words in fewer than 20 listings dropped).
4. Remove place names (boroughs, neighbourhoods, "nyc", etc.) so the
   results show *how* hosts describe listings, not just *where* they are.
5. For each borough, score each term by
       mean TF-IDF inside the borough - mean TF-IDF in all other boroughs
   and keep the top 10. A high score = a word that is distinctive for
   that borough's listings.

Run from the repo root:
    python src/tfidf_descriptions.py
Outputs go to outputs/.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "New York City.csv"   # Sept 2026 snapshot
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TOP_N = 10
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

# ---------------------------------------------------------------- load + clean
df = pd.read_csv(DATA, low_memory=False)
verified = df["host_identity_verified"] == "t"
has_price = df["price_quote_price_per_night"].notna()
available = df["has_availability"] == "t"
active = (df["estimated_occupancy_l365d"] != 0) | (df["availability_365"] != 0)
df = df[verified & has_price & available & active & df["description"].notna()].copy()
print(f"Listings after cleaning: {len(df):,}")

# Multi-listing hosts often paste one template into many listings, which
# would let a single host dominate a borough's terms. Keep one copy of each
# identical description.
n_before = len(df)
df = df.drop_duplicates(subset="description")
print(f"Dropped {n_before - len(df):,} duplicate descriptions -> {len(df):,} listings")

# Descriptions contain HTML tags such as <br />
df["text"] = (df["description"].str.replace(r"<[^>]+>", " ", regex=True)
              .str.lower())

# ---------------------------------------------------------------- stop words
place_words = set()
for name in pd.concat([df["neighbourhood_cleansed"], df["neighbourhood_group_cleansed"]]).unique():
    place_words.update(re.findall(r"[a-z]+", str(name).lower()))
place_words.update({"nyc", "new", "york", "city", "ny", "manhattan", "brooklyn",
                    "queens", "bronx", "staten", "island", "jfk", "lga",
                    "laguardia", "airport", "midtown", "downtown", "uptown",
                    "stuy", "bk", "les", "ues", "uws", "soho", "rumc"})
stop_words = list(ENGLISH_STOP_WORDS | place_words)

# ---------------------------------------------------------------- TF-IDF
vec = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), min_df=20,
                      sublinear_tf=True, token_pattern=r"(?u)\b[a-z]{3,}\b")
X = vec.fit_transform(df["text"])
terms = vec.get_feature_names_out()
print(f"Vocabulary size: {len(terms):,}")

rows = []
groups = df["neighbourhood_group_cleansed"].to_numpy()
for b in BOROUGHS:
    inside = groups == b
    diff = (X[inside].mean(axis=0) - X[~inside].mean(axis=0)).A1
    for rank, i in enumerate(diff.argsort()[::-1][:TOP_N], start=1):
        rows.append({"borough": b, "rank": rank, "term": terms[i],
                     "distinctiveness": round(float(diff[i]), 4)})
top = pd.DataFrame(rows)
top.to_csv(OUT / "tfidf_top_terms_by_borough.csv", index=False)

prices = (df.groupby("neighbourhood_group_cleansed")["price_quote_price_per_night"]
          .agg(listings="count", median_price="median").reindex(BOROUGHS))
prices.to_csv(OUT / "median_price_by_borough.csv")

summary = (top.groupby("borough")["term"].apply(lambda s: ", ".join(s.head(6)))
           .reindex(BOROUGHS).to_frame("top_terms").join(prices))
summary.to_csv(OUT / "tfidf_summary_table.csv")
print(summary.to_string())

# ---------------------------------------------------------------- figure
INK, MUTED, BAR = "#1f2328", "#6e7781", "#3b6fb6"
fig, axes = plt.subplots(1, 5, figsize=(16, 4.6), sharex=False)
for ax, b in zip(axes, BOROUGHS):
    d = top[top["borough"] == b].iloc[::-1]
    ax.barh(d["term"], d["distinctiveness"], color=BAR, height=0.6)
    ax.set_title(f"{b}\nmedian ${prices.loc[b, 'median_price']:.0f}/night",
                 fontsize=11, color=INK, loc="left")
    ax.tick_params(axis="y", labelsize=9.5, colors=INK, length=0)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d0d7de")
    ax.grid(axis="x", color="#eaeef2", linewidth=0.8)
    ax.set_axisbelow(True)
fig.suptitle("Most distinctive words in listing descriptions, by borough (TF-IDF)",
             fontsize=13, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.01, "Score = mean TF-IDF in borough minus mean in other boroughs. "
         "Place names removed. Inside Airbnb NYC, Sept 2026 snapshot.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(OUT / "tfidf_top_terms_by_borough.png", dpi=200)
print("Saved outputs/tfidf_top_terms_by_borough.png")
