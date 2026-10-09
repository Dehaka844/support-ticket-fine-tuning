import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


SEED = 42
MODEL_NAME = "bert-base-uncased"

DATA_DIR = Path("data")
MODEL_DIR = Path("models/bert_full")

LABELS = [
    "technical",
    "account",
    "billing",
    "general",
    "feature_request",
    "cancellation",
]

LABEL2ID = {
    label: index
    for index, label in enumerate(LABELS)
}

ID2LABEL = {
    index: label
    for label, index in LABEL2ID.items()
}


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation and test datasets."""

    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    return train, validation, test


def prepare_dataset(dataframe: pd.DataFrame) -> Dataset:
    """Convert a pandas DataFrame into a Hugging Face Dataset."""

    dataframe = dataframe.copy()

    dataframe["label"] = dataframe["category"].map(LABEL2ID)

    return Dataset.from_pandas(
        dataframe[
            [
                "text",
                "label",
            ]
        ],
        preserve_index=False,
    )


def calculate_class_weights(
    dataframe: pd.DataFrame,
) -> torch.Tensor:
    """
    Calculate balanced class weights using only the training set.
    """

    counts = (
        dataframe["category"]
        .value_counts()
        .reindex(LABELS)
    )

    total = len(dataframe)
    number_of_classes = len(LABELS)

    weights = total / (number_of_classes * counts)

    return torch.tensor(
        weights.values,
        dtype=torch.float32,
    )


def compute_metrics(eval_prediction):
    """Calculate classification metrics."""

    predictions, labels = eval_prediction

    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "f1_macro": f1_score(
            labels,
            predictions,
            average="macro",
        ),
        "precision_macro": precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


class WeightedTrainer(Trainer):
    """Trainer with weighted Cross Entropy Loss."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        weights = self.class_weights.to(logits.device)

        loss_function = torch.nn.CrossEntropyLoss(
            weight=weights
        )

        loss = loss_function(
            logits,
            labels,
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


def main() -> None:

    set_seed()

    print("=" * 60)
    print("BERT - FULL FINE TUNING")
    print("=" * 60)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"\nDispositivo: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    train_df, validation_df, test_df = load_data()

    print("\nTamaños:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:       {len(test_df)}")

    train_dataset = prepare_dataset(train_df)
    validation_dataset = prepare_dataset(validation_df)
    test_dataset = prepare_dataset(test_df)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
    )

    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
    )

    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
    )

    print("\nTokenización completada.")

    class_weights = calculate_class_weights(
        train_df
    )

    print("\nClass weights:")

    for label, weight in zip(
        LABELS,
        class_weights,
    ):
        print(
            f"{label:20s}: {weight:.4f}"
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir="models/bert_full",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=50,

        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        lr_scheduler_type="linear",

        fp16=True,
        report_to="none",
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        compute_metrics=compute_metrics,

        class_weights=class_weights,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    print("\nComenzando entrenamiento...\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory: {peak_memory:.2f} MB")
    

    print("\nEntrenamiento terminado.")

    print("\nEvaluando en TEST...")

    predictions = trainer.predict(
        test_dataset
    )

    predicted_labels = np.argmax(
        predictions.predictions,
        axis=1,
    )

    true_labels = predictions.label_ids

    print("\n=== MÉTRICAS ===")

    print(
        f"Accuracy: "
        f"{accuracy_score(true_labels, predicted_labels):.4f}"
    )

    print(
        f"F1 Macro: "
        f"{f1_score(true_labels, predicted_labels, average='macro'):.4f}"
    )

    print("\n=== REPORTE POR CLASE ===")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print("\n=== MATRIZ DE CONFUSIÓN ===")

    print(
        confusion_matrix(
            true_labels,
            predicted_labels,
        )
    )

    trainer.save_model(
        MODEL_DIR
    )

    tokenizer.save_pretrained(
        MODEL_DIR
    )

    print(
        f"\nModelo guardado en: {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()