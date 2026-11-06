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

Top TF-IDF words in listing descriptions by burrough:

Manhattan: building, location, elevator, doorman, views. 
Brooklyn: train, bars, backyard, cafes, shops. 
Queens: minutes, bus, parking, lirr, drive. 
Bronx: yankee, stadium, zoo, botanical, hospital. 
Staten Island: ferry, bus, free, parking, medical. 

Top TF-IDF words in reviews by burrough:

Manhattan: location, staff, small, building, price. 
Brooklyn: neighborhood, lovely, comfortable, responsive, subway. 
Queens: nice, clean, quiet, bus, peaceful. 
Bronx: nice, stadium, yankee, parking, metro. 
Staten Island: ferry, bus, wonderful, quiet, parking. 

Top TF-IDF words in reviews by quarter:

Q4 2025: subway, cozy, warm, christmas. 
Q1 2026: cozy, cold, warm, winter, snow. 
Q2 2026: neighborhood, quiet, beautiful, lovely. 
Q3 2026: staff, wonderful, clean, air. 

## Limitations

- Dec 2025 has no price data.
- Some reviews are not in English.

## Next Steps

- Compare TF-IDF words with price.

## Repository

Code: `src/tfidf_analysis.py`
