# How Airbnb Prices Vary by Neighbourhood in New York City

## Research Question

How do Airbnb prices vary by neighbourhood in New York City and over time? As a sub-question, how do hosts describe their listings differently across boroughs, and does that language line up with price differences?

## Team

| Member | Role |
|---|---|
| Justin | Proposal; results & interpretation |
| Stephen | Data cleaning; results & interpretation |
| David | Exploratory data analysis; documentation; results & interpretation |
| Deanne | Statistical analysis (TF-IDF text analysis of listings and reviews); results & interpretation |
| Anne | Data visualization; results & interpretation |

## Data

- **Source:** [Inside Airbnb](https://insideairbnb.com/), New York City.
- **Listings:** four quarterly snapshots covering one year: Dec 2025, Mar 2026, Jun 2026 and Sep 2026. Each row is one listing as it appeared on the scrape date.
- **Reviews:** the full-text reviews file from the Sep 2026 snapshot. It contains every review with its date, so we use reviews dated Oct 2025 to Sep 2026.
- **Key variables:** borough and neighbourhood (`neighbourhood_group_cleansed`, `neighbourhood_cleansed`), quoted nightly price (`price_quote_price_per_night`), room type, host type, availability, listing text (`description`) and review text (`comments`).
- **Cleaning:** verified host, has availability, and some activity in the past year. The Dec 2025 snapshot has **no price data**, so price comparisons cover Mar to Sep 2026 only.
- **Ethics:** host names, reviewer names and exact coordinates are not published on this site. Reviewer names and IDs are dropped on load, and host first names are removed from review text. Results are reported only at the borough level.

## Methods

**TF-IDF text analysis** of listing descriptions and guest reviews.

1. Each description or review becomes a TF-IDF vector (unigrams and bigrams; English stop words, place names and host names removed).
2. A term's *distinctiveness* for a group is its mean TF-IDF inside the group minus its mean everywhere else. We compare boroughs, and quarters across the year.
3. To stop a few large actors from dominating: duplicate descriptions are dropped, each host contributes at most 5 descriptions per snapshot, and each listing contributes at most 10 reviews per quarter.
4. **Stability check:** we recompute each borough's top 10 terms within every quarter and count how many match the yearly top 10.

Code: `src/tfidf_descriptions.py` and `src/tfidf_reviews.py`. We also compare median quoted price by borough across snapshots.

## Preliminary Results

### How hosts describe listings (22,085 unique descriptions, Dec 2025 – Sep 2026)

| Borough | Most distinctive description terms | Median $/night (Mar → Jun → Sep 2026) | Terms stable across quarters |
|---|---|---|---|
| Manhattan | central, building, times, elevator, hudson, doorman | 208 → 234 → 245 | 9–10 / 10 |
| Brooklyn | train, brownstone, garden, barclays, cafes, backyard | 139 → 152 → 155 | 9–10 / 10 |
| Queens | minutes, parking, LIRR, bus, house, drive | 111 → 120 → 124 | 8–10 / 10 |
| Bronx | Yankee Stadium, zoo, botanical garden, Montefiore, hospital | 101 → 106 → 106 | 9–10 / 10 |
| Staten Island | ferry, bus, free ferry, parking, medical, students | 100 → 107 → 105 | 7–9 / 10 |

Manhattan listings are sold on centrality and building features such as elevators and doormen. Brooklyn listings are sold on character (brownstone, garden, backyard). Queens, the Bronx and Staten Island are sold on transport and travel time (minutes, LIRR, ferry, parking) and on specific destinations (Yankee Stadium, the zoo, hospitals). The pattern holds in every quarter. Prices rose in every borough from March to September, most steeply in Manhattan (+18%).

### What guests say (84,226 English reviews, Oct 2025 – Sep 2026)

Manhattan reviews stand out for *location*, *staff*, *small* and *room*, which suggests hotel-like, professionally run units where guests pay for location and accept less space. Brooklyn reviews focus on *home*, *neighborhood* and *hosts*. Outer-borough reviews focus on the *host*, the *house*, *parking* and *clean*. Guests repeat the selling points hosts write about: location in Manhattan, transport and destinations elsewhere.

### Change over the year

The quarter-to-quarter differences in language are seasonal and event-driven. Reviews mention *christmas* in Q4 2025, *cold*, *heater* and *blizzard* in Q1 2026, *graduation* in Q2 2026, and *air conditioning* in Q3 2026. In listing descriptions, the June 2026 snapshot is distinguished by *world cup* and *metlife*, as hosts marketed to visitors for the 2026 FIFA World Cup.

These results are preliminary and descriptive. We have not yet tested whether these terms predict price within a borough.

## Limitations

- **Cleaning filter instability:** one corporate host with about 1,000 Manhattan units is marked identity-verified in the Dec and Sep snapshots but not in Mar and Jun. The verified-host filter therefore adds and removes about 1,000 listings between quarters. Before we capped listings per host, this created a false change in Manhattan's vocabulary over time. Any over-time comparison (including prices) needs to check whether it is driven by listings entering and leaving the cleaned data.
- Prices are quoted, not booked, and each snapshot captures only one day. Dec 2025 has no prices.
- TF-IDF partly picks up location through landmark and transit words, and some place references (e.g. "central") survive the place-name filter.
- Bronx and Staten Island have few listings and reviews, so their term lists are less stable.
- Non-English reviews are excluded with a simple heuristic, which under-represents international guests.

## Next Steps

- Decide as a team whether to keep the verified-host filter, or to apply it using a host's status in any snapshot.
- Test whether distinctive description terms predict price after controlling for borough and room type.
- Track the same listings (matched by `id`) across snapshots to separate real price changes from changes in which listings are included.

## Repository

Analysis code is in `src/` and outputs (tables and figures) are in `outputs/`. The raw data is too large for GitHub. To reproduce the results, download the four NYC `listings.csv.gz` snapshots and the Sep 2026 `reviews.csv.gz` (unzipped as `reviews.csv`) from Inside Airbnb into `data/raw/`, then run:

```bash
python src/tfidf_descriptions.py
python src/tfidf_reviews.py
```
