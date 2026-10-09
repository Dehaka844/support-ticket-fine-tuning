import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "models/bert_full"

LABELS = [
    "technical",
    "account",
    "billing",
    "general",
    "feature_request",
    "cancellation",
]


class TicketClassifier:

    def __init__(self, model_path=MODEL_PATH):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=-1,
        )

        predicted_label = LABELS[prediction.item()]
        confidence = confidence.item()

        # Umbral exigido por el ejercicio
        if confidence >= 0.8:
            status = "automatic"
        else:
            status = "human_review"

        return {
            "text": text,
            "category": predicted_label,
            "confidence": round(confidence, 4),
            "status": status,
        }


def main():

    print("=" * 60)
    print("INFERENCIA DEL MODELO")
    print("=" * 60)

    classifier = TicketClassifier()

    examples = [
        "My application crashes when I try to open it.",
        "I want to cancel my subscription.",
        "I was charged twice this month.",
        "I would like to request a new dark mode feature.",
        "I don't know what documents I need to register.",
        "The website is not loading and I cannot access my account.",
    ]

    for text in examples:

        result = classifier.predict(text)

        print("\nTicket:")
        print(result["text"])

        print(f"Category:   {result['category']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Decision:   {result['status']}")


if __name__ == "__main__":
    main()