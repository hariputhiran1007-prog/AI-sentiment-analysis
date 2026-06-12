import argparse
import csv
import random
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "data" / "reviews_500k.csv"
LABELS = ("Positive", "Negative", "Neutral")

SUBJECTS = (
    "product",
    "service",
    "delivery",
    "mobile app",
    "website",
    "course",
    "restaurant",
    "hotel",
    "support team",
    "shopping experience",
    "phone",
    "laptop",
    "booking process",
    "movie",
    "food",
)

POSITIVE_ADJECTIVES = (
    "excellent",
    "amazing",
    "helpful",
    "reliable",
    "smooth",
    "comfortable",
    "fast",
    "friendly",
    "impressive",
    "valuable",
    "clear",
    "professional",
)

NEGATIVE_ADJECTIVES = (
    "terrible",
    "poor",
    "slow",
    "confusing",
    "broken",
    "unhelpful",
    "disappointing",
    "bad",
    "uncomfortable",
    "buggy",
    "rude",
    "frustrating",
)

NEUTRAL_ADJECTIVES = (
    "average",
    "standard",
    "normal",
    "basic",
    "acceptable",
    "ordinary",
    "usual",
    "moderate",
    "expected",
    "regular",
    "simple",
    "fair",
)

POSITIVE_OUTCOMES = (
    "I am happy with it",
    "I would recommend it",
    "it worked better than expected",
    "the experience was pleasant",
    "it was worth the money",
    "everything felt easy to use",
)

NEGATIVE_OUTCOMES = (
    "I am unhappy with it",
    "I would not recommend it",
    "it worked worse than expected",
    "the experience was unpleasant",
    "it was not worth the money",
    "everything felt hard to use",
)

NEUTRAL_OUTCOMES = (
    "I have no strong opinion",
    "it matched my expectation",
    "the experience was neither good nor bad",
    "it was fine for basic use",
    "there is nothing special to mention",
    "it worked as described",
)

TEMPLATES = (
    "The {subject} was {adjective}, and {outcome}.",
    "I found the {subject} {adjective}; {outcome}.",
    "Overall, the {subject} felt {adjective} because {outcome}.",
    "My review of the {subject}: {adjective}. {outcome}.",
    "The {subject} is {adjective}, so {outcome}.",
)


def make_review(label: str, rng: random.Random) -> str:
    subject = rng.choice(SUBJECTS)

    if label == "Positive":
        adjective = rng.choice(POSITIVE_ADJECTIVES)
        outcome = rng.choice(POSITIVE_OUTCOMES)
    elif label == "Negative":
        adjective = rng.choice(NEGATIVE_ADJECTIVES)
        outcome = rng.choice(NEGATIVE_OUTCOMES)
    else:
        adjective = rng.choice(NEUTRAL_ADJECTIVES)
        outcome = rng.choice(NEUTRAL_OUTCOMES)

    return rng.choice(TEMPLATES).format(
        subject=subject,
        adjective=adjective,
        outcome=outcome,
    )


def generate_dataset(output_path: Path, samples: int, seed: int) -> None:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(samples):
        label = LABELS[index % len(LABELS)]
        rows.append((make_review(label, rng), label))

    rng.shuffle(rows)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("text", "sentiment"))
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a balanced demo sentiment dataset."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args.output, args.samples, args.seed)
    print(f"Generated {args.samples:,} rows at {args.output}")
