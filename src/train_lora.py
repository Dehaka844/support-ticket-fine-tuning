import os
import time
import json
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_NAME = "bert-base-uncased"

SEED = 42
MAX_LENGTH = 128

LABELS = [
    "technical",
    "account",
    "billing",
    "general",
    "feature_request",
    "cancellation",
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

OUTPUT_DIR = "models/bert_lora"


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MÉTRICAS
# ============================================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }


# ============================================================
# TOKENIZACIÓN
# ============================================================

def tokenize_dataset(dataset, tokenizer):

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    return dataset.map(
        tokenize,
        batched=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 60)
    print("BERT - LoRA FINE TUNING")
    print("=" * 60)

    # --------------------------------------------------------
    # DISPOSITIVO
    # --------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDispositivo: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # CARGAR DATASET
    # --------------------------------------------------------

    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    print("\nTamaños:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")

    # --------------------------------------------------------
    # PREPARAR LABELS
    # --------------------------------------------------------

    for df in [train_df, val_df, test_df]:
        df["label"] = df["category"].map(LABEL2ID)

    train_dataset = Dataset.from_pandas(
        train_df[["text", "label"]],
        preserve_index=False,
    )

    val_dataset = Dataset.from_pandas(
        val_df[["text", "label"]],
        preserve_index=False,
    )

    test_dataset = Dataset.from_pandas(
        test_df[["text", "label"]],
        preserve_index=False,
    )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    val_dataset = tokenize_dataset(val_dataset, tokenizer)
    test_dataset = tokenize_dataset(test_dataset, tokenizer)

    print("\nTokenización completada.")

    # --------------------------------------------------------
    # MODELO BASE
    # --------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # --------------------------------------------------------
    # CONFIGURACIÓN LoRA
    # --------------------------------------------------------

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,

        # Mayor capacidad de adaptación
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,

        # Aplicar LoRA a query, key, value y output
        target_modules=[
            "query",
            "key",
            "value",
            "dense",
        ],

        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print("\n" + "=" * 60)
    print("CONFIGURACIÓN LoRA")
    print("=" * 60)

    model.print_trainable_parameters()

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

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

        seed=SEED,
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=val_dataset,

        processing_class=tokenizer,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    # --------------------------------------------------------
    # ENTRENAMIENTO
    # --------------------------------------------------------

    print("\nComenzando entrenamiento LoRA...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    train_result = trainer.train()

    training_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # MEMORIA GPU
    # --------------------------------------------------------

    peak_memory_mb = 0

    if torch.cuda.is_available():
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        )

    print("\nEntrenamiento terminado.")

    print(f"Tiempo entrenamiento: {training_time:.2f} segundos")
    print(f"Memoria GPU máxima: {peak_memory_mb:.2f} MB")

    # --------------------------------------------------------
    # EVALUACIÓN TEST
    # --------------------------------------------------------

    print("\nEvaluando en TEST...")

    predictions = trainer.predict(test_dataset)

    predicted_labels = np.argmax(
        predictions.predictions,
        axis=-1,
    )

    true_labels = predictions.label_ids

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    print("\n=== MÉTRICAS ===")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1:.4f}")

    # --------------------------------------------------------
    # REPORTE
    # --------------------------------------------------------

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )

    print("\n=== REPORTE POR CLASE ===")
    print(report)

    # --------------------------------------------------------
    # MATRIZ DE CONFUSIÓN
    # --------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    print("\n=== MATRIZ DE CONFUSIÓN ===")
    print(cm)

    # --------------------------------------------------------
    # GUARDAR MODELO
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(
        f"\nModelo LoRA guardado en: {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # PARÁMETROS
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    trainable_percentage = (
        trainable_params / total_params
    ) * 100

    print("\n=== PARÁMETROS ===")

    print(
        f"Parámetros totales: "
        f"{total_params:,}"
    )

    print(
        f"Parámetros entrenables: "
        f"{trainable_params:,}"
    )

    print(
        f"Porcentaje entrenable: "
        f"{trainable_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # GUARDAR RESULTADOS
    # --------------------------------------------------------

    results = {
        "model": MODEL_NAME,
        "method": "LoRA",
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "training_time_seconds": float(training_time),
        "peak_gpu_memory_mb": float(peak_memory_mb),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "trainable_percentage": float(
            trainable_percentage
        ),
        "seed": SEED,
    }

    os.makedirs("results", exist_ok=True)

    with open(
        "results/lora_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        "\nResultados guardados en: "
        "results/lora_metrics.json"
    )


if __name__ == "__main__":
    main()