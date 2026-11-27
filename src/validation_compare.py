from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load CSV files


def load_csv(filename):
    return pd.read_csv(
        PROJECT_ROOT / "outputs" / filename,
        dtype={"review_id": str},
        keep_default_na=False
    )


human = load_csv("validation_250_reviews.csv")
v1 = load_csv("validation_250_ai.csv")
v2 = load_csv("validation_100_ai_v2.csv")

# Check for duplicate review IDs
assert human["review_id"].is_unique
assert v1["review_id"].is_unique
assert v2["review_id"].is_unique

# Select the same 100 reviews for comparison
assert len(v2) == 100

sample_ids = set(v2["review_id"])

human = human[human["review_id"].isin(sample_ids)]
v1 = v1[v1["review_id"].isin(sample_ids)]

assert len(human) == len(v1) == len(v2) == 100
assert set(human["review_id"]) == sample_ids
assert set(v1["review_id"]) == sample_ids

# Convert topics into sets to ignore order


def topic_set(value):
    return {
        t.strip()
        for t in value.split(";")
        if t.strip()
    }

# Calculate accuracy


def evaluate(ai):
    df = human.merge(
        ai,
        on="review_id",
        validate="one_to_one"
    )

    topic_accuracy = (
        df["topic"].apply(topic_set)
        == df["llm_topic"].apply(topic_set)
    ).mean()

    sentiment_accuracy = (
        df["sentiment"].str.strip()
        == df["llm_sentiment"].str.strip()
    ).mean()

    return topic_accuracy, sentiment_accuracy


# Load V3 results
v3 = load_csv("validation_100_ai_v3.csv")

assert len(v3) == 100
assert v3["review_id"].is_unique
assert set(v3["review_id"]) == sample_ids

# Compare V1, V2, and V3
v1_topic, v1_sentiment = evaluate(v1)
v2_topic, v2_sentiment = evaluate(v2)
v3_topic, v3_sentiment = evaluate(v3)

print("Number of reviews: 100")

print("\nTopic Accuracy")
print(f"V1: {v1_topic:.2%}")
print(f"V2: {v2_topic:.2%}")
print(f"V3: {v3_topic:.2%}")

print("\nSentiment Accuracy")
print(f"V1: {v1_sentiment:.2%}")
print(f"V2: {v2_sentiment:.2%}")
print(f"V3: {v3_sentiment:.2%}")

print("\nImprovement from V2 to V3")
print(f"Topic: {(v3_topic - v2_topic)*100:.2f} pp")
print(f"Sentiment: {(v3_sentiment - v2_sentiment)*100:.2f} pp")
