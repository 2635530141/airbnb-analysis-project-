# How Airbnb Prices Vary by Neighbourhood in New York City

## Research Question

How do Airbnb prices vary by neighbourhood in New York City and over time? We also ask how hosts and guests describe listings in each borough, and whether that language lines up with price.

## Team

| Member | Role |
|---|---|
| Justin | Proposal; results & interpretation |
| Stephen | Data cleaning; results & interpretation |
| David | Exploratory data analysis; documentation; results & interpretation |
| Deanne | Statistical analysis (TF-IDF on listings and reviews); results & interpretation |
| Anne | Data visualization; results & interpretation |

## Data

- **Source:** [Inside Airbnb](https://insideairbnb.com/), New York City
- **Listings:** 4 quarterly snapshots (Dec 2025, Mar 2026, Jun 2026, Sep 2026)
- **Reviews:** all reviews from Oct 2025 to Sep 2026
- **Text used:** listing descriptions (`description`) and review text (`comments`)
- **Note:** the Dec 2025 snapshot has no price data
- **Privacy:** reviewer names are dropped, and host names are removed from the text

## Methods

**TF-IDF.** Every listing description and review gets a TF-IDF score for each word. For each borough (and each quarter), we find the words whose average score is much higher there than everywhere else. Those are the words that make that borough stand out.

Before running TF-IDF we:
- removed copy-pasted descriptions
- kept at most 5 listings per host per quarter, so big companies don't take over the results
- removed common English words, place names and host names

Code: `src/tfidf_analysis.py`

## Preliminary Results

**How hosts describe each borough (listing descriptions):**

| Borough | Top TF-IDF words | Median price/night (Mar → Jun → Sep 2026) |
|---|---|---|
| Manhattan | building, park, elevator, location, doorman | $199 → $221 → $236 |
| Brooklyn | train, garden, bars, backyard, cafes | $151 → $165 → $166 |
| Queens | minutes, bus, parking, LIRR, drive | $121 → $131 → $133 |
| Bronx | Yankee Stadium, zoo, botanical, Montefiore, hospital | $105 → $108 → $108 |
| Staten Island | ferry, bus, free, parking, medical | $101 → $108 → $110 |

Manhattan, the most expensive borough, is described by its buildings and location (elevator, doorman). Brooklyn is described by its neighbourhood feel (garden, backyard, cafes). The cheaper boroughs are described by how to get somewhere else (bus, ferry, LIRR, parking) or by nearby places (Yankee Stadium, the zoo, hospitals).

**What guests say (reviews):** Manhattan reviews mention *location*, *staff* and *small*, so guests seem to pay for location and accept smaller rooms. Reviews in the other boroughs talk more about the *home*, being *quiet*, and *parking*.

**Over the year:** reviews change with the seasons. They mention *christmas* in Q4 2025, *cold* and *blizzard* in Q1 2026, and *air conditioning* in Q3 2026. Prices went up in every borough from March to September, most in Manhattan (about +19%).

## Limitations

- Prices are listed prices, not what guests actually paid, and Dec 2025 has no prices.
- Some TF-IDF words are really about location (e.g. "Yankee Stadium"), not how a place is described.
- Some reviews are not in English (e.g. "excelente", "muy"), and those words show up in the results.
- Listing descriptions by quarter don't show a clear pattern yet, so we need to look into this more.

## Next Steps

- Test whether TF-IDF words help predict price within each borough.
- Decide as a team how to clean the data the same way for every quarter.

## Repository

Code is in `src/` and result tables are in `outputs/`. The raw data is too large for GitHub. Download the 4 NYC listings files and `reviews.csv` from Inside Airbnb into `data/raw/`, then run `python src/tfidf_analysis.py`.
