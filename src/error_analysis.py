import numpy as np
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

ID2LABEL = {
    i: label
    for i, label in enumerate(LABELS)
}


# ============================================================
# EJEMPLOS NUEVOS NO VISTOS DURANTE EL ENTRENAMIENTO
# ============================================================

TEST_EXAMPLES = [
    {
        "text": "The application crashes every time I try to upload a document.",
        "expected": "technical",
    },
    {
        "text": "I need to change the email address associated with my account.",
        "expected": "account",
    },
    {
        "text": "I was charged twice for the same subscription this month.",
        "expected": "billing",
    },
    {
        "text": "I would like to know what documents are required to register.",
        "expected": "general",
    },
    {
        "text": "It would be useful to have a dark mode in the application.",
        "expected": "feature_request",
    },
    {
        "text": "Please cancel my subscription immediately.",
        "expected": "cancellation",
    },

    # Casos algo más ambiguos
    {
        "text": "I cannot cancel my subscription because the button does not work.",
        "expected": "cancellation",
    },
    {
        "text": "I was charged for a service that I want to cancel.",
        "expected": "cancellation",
    },
    {
        "text": "The website is not loading and I cannot access my account.",
        "expected": "technical",
    },
    {
        "text": "Can I get a refund for the payment I made yesterday?",
        "expected": "billing",
    },
]


def main():

    print("=" * 60)
    print("ANÁLISIS DE ERRORES")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDispositivo: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    # PREDICCIONES
    # --------------------------------------------------------

    errors = []

    print("\n=== PREDICCIONES ===\n")

    for example in TEST_EXAMPLES:

        text = example["text"]
        expected = example["expected"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=-1,
            )

        predicted_id = prediction.item()
        predicted_label = ID2LABEL[predicted_id]
        confidence = confidence.item()

        correct = predicted_label == expected

        status = "OK" if correct else "ERROR"

        print(f"[{status}]")
        print(f"Texto:     {text}")
        print(f"Esperado:  {expected}")
        print(f"Predicho:  {predicted_label}")
        print(f"Confianza: {confidence:.4f}")
        print()

        if not correct:

            errors.append(
                {
                    "text": text,
                    "expected": expected,
                    "predicted": predicted_label,
                    "confidence": confidence,
                }
            )

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    total = len(TEST_EXAMPLES)
    correct = total - len(errors)

    accuracy = correct / total

    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    print(f"Ejemplos evaluados: {total}")
    print(f"Aciertos:           {correct}")
    print(f"Errores:            {len(errors)}")
    print(f"Accuracy:           {accuracy:.2%}")

    # --------------------------------------------------------
    # ERRORES
    # --------------------------------------------------------

    if errors:

        print("\n=== ERRORES DETECTADOS ===")

        for error in errors:

            print("\nTexto:")
            print(error["text"])

            print(
                f"Esperado: {error['expected']} | "
                f"Predicho: {error['predicted']} | "
                f"Confianza: {error['confidence']:.4f}"
            )

    else:

        print(
            "\nNo se han detectado errores "
            "en los ejemplos nuevos."
        )


if __name__ == "__main__":
    main()