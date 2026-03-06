from dataclasses import dataclass
from typing import Dict


class Preprocessor:
    def run(self, text: str) -> str:
        return text.strip().lower()


class Model:
    def score(self, text: str) -> float:
        # Dummy model: score = length
        return float(len(text)) / 10.0


class Postprocessor:
    def to_label(self, score: float) -> str:
        if score > 2.9:
            return 'long'
        elif score > 1.0:
            return 'short'
        return "very_short"


@dataclass
class InferenceService:
    pre: Preprocessor
    model: Model
    post: Postprocessor

    def predict(self, text: str) -> Dict[str, object]:
        clean = self.pre.run(text)
        s = self.model.score(clean)
        label = self.post.to_label(s)
        return {"label": label, "score": s, "clean_text": clean, "length": len(clean)}


if __name__ == "__main__":
    svc = InferenceService(Preprocessor(), Model(), Postprocessor())

    samples = [
        "Good morning",
        "This is a longer sentence for testing.",
        "   HELLO   ",
    ]

    for t in samples:
        print(t, "->", svc.predict(t))
