import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_PATH = "models/bert_full"

LABELS = [
    "technical",
    "account",
    "billing",
    "general",
    "feature_request",
    "cancellation",
]


def main():

    print("=" * 60)
    print("ANÁLISIS DE AUTOMATIZACIÓN")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDispositivo: {device}")

    # --------------------------------------------------------
    # CARGAR MODELO
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # CARGAR TEST
    # --------------------------------------------------------

    df = pd.read_csv("data/test.csv")

    print(f"Tickets analizados: {len(df)}")

    texts = df["text"].tolist()

    # --------------------------------------------------------
    # INFERENCIA POR LOTES
    # --------------------------------------------------------

    batch_size = 32

    predictions = []
    confidences = []

    with torch.inference_mode():

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i + batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=-1,
            )

            predictions.extend(
                prediction.cpu().tolist()
            )

            confidences.extend(
                confidence.cpu().tolist()
            )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    df["prediction"] = [
        LABELS[p]
        for p in predictions
    ]

    df["confidence"] = confidences

    df["decision"] = df["confidence"].apply(
        lambda x: (
            "automatic"
            if x > 0.8
            else "human_review"
        )
    )

    automatic = (
        df["decision"] == "automatic"
    ).sum()

    human_review = (
        df["decision"] == "human_review"
    ).sum()

    total = len(df)

    automatic_percentage = (
        automatic / total * 100
    )

    human_percentage = (
        human_review / total * 100
    )

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    print(f"\nTotal de tickets:       {total}")
    print(f"Automáticos:            {automatic}")
    print(f"Revisión humana:        {human_review}")

    print(
        f"\nAutomatización estimada: "
        f"{automatic_percentage:.2f}%"
    )

    print(
        f"Revisión humana:        "
        f"{human_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # DISTRIBUCIÓN DE CONFIANZA
    # --------------------------------------------------------

    print("\n=== DISTRIBUCIÓN DE CONFIANZA ===")

    print(
        f"Confianza media: "
        f"{df['confidence'].mean():.4f}"
    )

    print(
        f"Confianza mínima: "
        f"{df['confidence'].min():.4f}"
    )

    print(
        f"Confianza máxima: "
        f"{df['confidence'].max():.4f}"
    )

    # --------------------------------------------------------
    # GUARDAR RESULTADOS
    # --------------------------------------------------------

    df.to_csv(
        "results/automation_analysis.csv",
        index=False,
    )

    print(
        "\nResultados guardados en:"
        " results/automation_analysis.csv"
    )


if __name__ == "__main__":
    main()