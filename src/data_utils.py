import random
from pathlib import Path

from sklearn.model_selection import train_test_split

import pandas as pd


SEED = 42

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CLASS_DISTRIBUTION = {
    "technical": 1500,
    "account": 1200,
    "billing": 900,
    "general": 700,
    "feature_request": 450,
    "cancellation": 250,
}


TICKET_TEMPLATES = {
    "technical": [
        "The application crashes when I {action}.",
        "I am getting an error when I try to {action}.",
        "The system stops working whenever I {action}.",
        "There is a bug preventing me from {action}.",
        "The application is not working correctly when I {action}.",
        "I keep receiving an error while trying to {action}.",
        "The software freezes when I {action}.",
        "Something is broken and I cannot {action}.",
    ],
    "account": [
        "I cannot log into my account.",
        "I forgot my password and cannot access my account.",
        "How can I change my password?",
        "I need to update the email address on my profile.",
        "I cannot access my account after changing my password.",
        "How do I change my account permissions?",
        "My account is locked and I cannot sign in.",
        "I need help updating my profile information.",
    ],
    "billing": [
        "I was charged twice for my subscription.",
        "I do not recognize a charge on my account.",
        "Why was I charged for this month?",
        "My payment was declined.",
        "I need a copy of my invoice.",
        "How can I change my payment method?",
        "The subscription price on my invoice is incorrect.",
        "I was charged the wrong amount.",
    ],
    "general": [
        "Can you give me more information about the service?",
        "Where can I find more information about this product?",
        "How does the platform work?",
        "I have a general question about the service.",
        "Can you explain how this works?",
        "Where can I find the documentation?",
        "I would like more information about your plans.",
        "Can you tell me more about the platform?",
    ],
    "feature_request": [
        "I would like the platform to support {feature}.",
        "Could you add {feature} to the application?",
        "It would be useful to have {feature}.",
        "I would like to request a new feature for {feature}.",
        "Please consider adding {feature} in a future update.",
        "Is it possible to add {feature}?",
        "I think the platform would be better with {feature}.",
        "Could you implement {feature}?",
    ],
    "cancellation": [
        "I want to cancel my subscription.",
        "How can I cancel my account?",
        "I would like to cancel my plan.",
        "Please help me cancel my subscription.",
        "I want to stop using the service.",
        "I do not want to renew my subscription.",
        "How do I terminate my subscription?",
        "I want to close my account and cancel the service.",
    ],
}


ACTION_VARIATIONS = [
    "upload a file",
    "open a document",
    "save a project",
    "export a report",
    "generate a document",
    "import data",
    "download a file",
    "open the dashboard",
]


FEATURE_VARIATIONS = [
    "dark mode",
    "bulk file uploads",
    "a mobile application",
    "advanced reporting",
    "custom notifications",
    "calendar integration",
    "additional export formats",
    "an improved search system",
]


def generate_ticket(category: str) -> str:
    """Generate a single synthetic support ticket."""

    template = random.choice(TICKET_TEMPLATES[category])

    if "{action}" in template:
        return template.format(
            action=random.choice(ACTION_VARIATIONS)
        )

    if "{feature}" in template:
        return template.format(
            feature=random.choice(FEATURE_VARIATIONS)
        )

    return template


def generate_confidence(category: str) -> int:
    """
    Generate a synthetic human-classifier confidence score
    between 1 and 5.
    """

    if category in {"technical", "account", "billing"}:
        return random.choices(
            [3, 4, 5],
            weights=[10, 35, 55],
            k=1,
        )[0]

    return random.choices(
        [2, 3, 4, 5],
        weights=[10, 25, 40, 25],
        k=1,
    )[0]


def create_dataset() -> pd.DataFrame:
    """Create the complete synthetic support-ticket dataset."""

    random.seed(SEED)

    rows = []
    ticket_id = 1

    for category, quantity in CLASS_DISTRIBUTION.items():
        for _ in range(quantity):
            rows.append(
                {
                    "id": ticket_id,
                    "text": generate_ticket(category),
                    "category": category,
                    "confidence": generate_confidence(category),
                }
            )

            ticket_id += 1

    dataset = pd.DataFrame(rows)

    dataset = dataset.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    return dataset


def save_dataset(dataset: pd.DataFrame) -> None:
    """Save the generated dataset to CSV."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / "support_tickets.csv"

    dataset.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"Dataset guardado en: {output_path}")
    print(f"Total de tickets: {len(dataset)}")

def split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation and test sets.

    Distribution:
        - 70% train
        - 15% validation
        - 15% test

    Stratification is used to preserve class distribution.
    """

    train, temp = train_test_split(
        dataset,
        test_size=0.30,
        stratify=dataset["category"],
        random_state=SEED,
    )

    validation, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["category"],
        random_state=SEED,
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Save train, validation and test datasets."""

    train.to_csv(
        DATA_DIR / "train.csv",
        index=False,
        encoding="utf-8",
    )

    validation.to_csv(
        DATA_DIR / "val.csv",
        index=False,
        encoding="utf-8",
    )

    test.to_csv(
        DATA_DIR / "test.csv",
        index=False,
        encoding="utf-8",
    )

    print("\nDatasets guardados:")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")

if __name__ == "__main__":
    dataset = create_dataset()

    print("\nDistribución de clases:")
    print(dataset["category"].value_counts())

    print("\nPrimeros tickets:")
    print(dataset.head())

    save_dataset(dataset)

    train, validation, test = split_dataset(dataset)

    save_splits(
        train,
        validation,
        test,
    )

    print("\nDistribución por conjunto:")
    print("\nTRAIN:")
    print(train["category"].value_counts())

    print("\nVALIDATION:")
    print(validation["category"].value_counts())

    print("\nTEST:")
    print(test["category"].value_counts())