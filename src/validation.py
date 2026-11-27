from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load human and AI labels
human = pd.read_csv(
    PROJECT_ROOT / "outputs/validation_250_reviews.csv",
    dtype={"review_id": str},
    keep_default_na=False
)

ai = pd.read_csv(
    PROJECT_ROOT / "outputs/validation_250_ai.csv",
    dtype={"review_id": str},
    keep_default_na=False
)

# Check data completeness
assert len(human) == 250
assert len(ai) == 250
assert human["review_id"].is_unique
assert ai["review_id"].is_unique
assert set(human["review_id"]) == set(ai["review_id"])

# Rename human label columns
human = human.rename(columns={
    "topic": "human_topic",
    "sentiment": "human_sentiment"
})

# Merge human and AI labels
df = human.merge(
    ai,
    on="review_id",
    validate="one_to_one"
)

# Convert topics into sets to ignore topic order


def topic_set(value):
    return {
        t.strip()
        for t in value.split(";")
        if t.strip()
    }


# Compare topic labels
df["topic_match"] = [
    topic_set(h) == topic_set(a)
    for h, a in zip(df["human_topic"], df["llm_topic"])
]

# Compare sentiment labels
df["sentiment_match"] = (
    df["human_sentiment"].str.strip()
    == df["llm_sentiment"].str.strip()
)

# Calculate accuracy
topic_accuracy = df["topic_match"].mean()
sentiment_accuracy = df["sentiment_match"].mean()

print(f"Topic Accuracy: {topic_accuracy:.2%}")
print(f"Sentiment Accuracy: {sentiment_accuracy:.2%}")

# Extract disagreements
disagreements = df[
    (~df["topic_match"]) |
    (~df["sentiment_match"])
].copy()

disagreements = disagreements[[
    "review_id",
    "comments",
    "human_topic",
    "llm_topic",
    "human_sentiment",
    "llm_sentiment",
    "llm_reason",
    "topic_match",
    "sentiment_match"
]]

# Save disagreements
disagreements.to_csv(
    PROJECT_ROOT / "outputs/disagreements.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Topic disagreements:", (~df["topic_match"]).sum())
print("Sentiment disagreements:", (~df["sentiment_match"]).sum())
print("Total disagreements:", len(disagreements))
print("Disagreements saved.")
