# How Airbnb Prices Vary by Neighbourhood in New York City

## Research Question

How do Airbnb prices vary by neighbourhood in New York City and over time?

## Team

- Justin Kaye, Stephen Wu, David He, Deanne Balsbaugh, Anne Kim

## Data

- Inside Airbnb, New York City
- Listings: Dec 2025, Mar 2026, Jun 2026, Sep 2026
- Reviews: Oct 2025 to Sep 2026

## Methods

TF-IDF on listing descriptions and reviews, by borough, and by quarter.

## Preliminary Results

Top TF-IDF words in listing descriptions by borough:

| Borough | Top words (score) |
|---|---|
| Manhattan | building (0.024), location (0.013), elevator (0.013), doorman (0.012), heart (0.012) |
| Brooklyn | train (0.013), bars (0.009), backyard (0.008), stop (0.008), cafes (0.008) |
| Queens | airport (0.043), minutes (0.032), mins (0.019), parking (0.016), bus (0.015) |
| Bronx | yankee (0.043), stadium (0.037), zoo (0.028), botanical (0.018), montefiore (0.014) |
| Staten Island | ferry (0.100), bus (0.040), free (0.036), parking (0.030), medical (0.022) |

Top TF-IDF words in reviews by borough:

| Borough | Top words (score) |
|---|---|
| Manhattan | location (0.043), staff (0.025), small (0.008), perfect (0.006), building (0.006) |
| Brooklyn | home (0.011), neighborhood (0.011), lovely (0.010), comfortable (0.008), hosts (0.008) |
| Queens | airport (0.014), nice (0.013), clean (0.012), home (0.007), quiet (0.006) |
| Bronx | nice (0.012), stadium (0.009), yankee (0.008), good (0.007), parking (0.006) |
| Staten Island | ferry (0.047), home (0.017), wonderful (0.011), quiet (0.011), thank (0.011) |

Top TF-IDF words in reviews by quarter:

| Quarter | Top words (score) |
|---|---|
| Q4 2025 | subway (0.005), cozy (0.005), station (0.003), safe (0.002), warm (0.002), christmas (0.002), close (0.002), home (0.002), felt (0.002), restaurants (0.002) |
| Q1 2026 | staff (0.003), cozy (0.003), good (0.003), cold (0.003), warm (0.002), winter (0.002), price (0.002), location (0.002), snow (0.002), definitely (0.002) |
| Q2 2026 | neighborhood (0.002), quiet (0.002), location (0.001), beautiful (0.001), easy (0.001), value (0.001), lovely (0.001), access (0.001), train (0.001), subway (0.001) |
| Q3 2026 | staff (0.003), wonderful (0.002), air (0.002), clean (0.002), definitely (0.002), conditioning (0.002), nice (0.002), home (0.002), good (0.001), parking (0.001) |

Score = how much more the word is used in that group than everywhere else.

## Limitations

- Dec 2025 has no price data.
- Non-English reviews were removed with a simple word check, so a few may be missed.

## Next Steps

- Compare TF-IDF words with price.

## Repository

Code: `src/tfidf_analysis.py`
