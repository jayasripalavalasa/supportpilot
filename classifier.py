"""Pre-processing, classification, severity and priority for SupportPilot."""
import re
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent / "models" / "ticket_classifier.pkl"

# Work 3 - Text pre-processing. Negations ("not", "cannot") are kept on purpose:
# they matter for IT tickets ("VPN is not connecting").
STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "to", "of", "and",
    "i", "my", "me", "in", "on", "for", "it", "this", "that", "with", "at",
    "be", "been", "has", "have", "had", "so", "very",
}


def clean_text(text):
    """lowercase -> strip symbols -> tokenize -> drop stop-words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)


# Work 5 - Severity engine (rule based, from the mentor's template)
CRITICAL_WORDS = ["server down", "entire company", "production down",
                  "security breach", "data breach", "ransomware"]
HIGH_WORDS = ["urgent", "cannot work", "can't work", "business stopped",
              "client meeting", "client presentation", "vpn not working",
              "asap", "deadline"]
MEDIUM_WORDS = ["slow", "error", "problem", "issue", "not working", "failed"]


def predict_severity(ticket):
    text = ticket.lower()
    for word in CRITICAL_WORDS:
        if word in text:
            return "Critical"
    for word in HIGH_WORDS:
        if word in text:
            return "High"
    for word in MEDIUM_WORDS:
        if word in text:
            return "Medium"
    return "Low"


# Work 6 - Priority
def calculate_priority(severity, business_impact):
    """Severity + business impact -> priority (slide 34)."""
    if severity == "Critical" and business_impact == "High":
        return "P1"
    elif severity == "High" and business_impact == "High":
        return "P1"
    elif severity == "High":
        return "P2"
    elif severity == "Medium":
        return "P3"
    return "P4"


SEVERITY_TO_PRIORITY = {"Critical": "P1", "High": "P2", "Medium": "P3", "Low": "P4"}

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run: python train_model.py")
        _model = joblib.load(MODEL_PATH)  # sklearn Pipeline: TF-IDF + LogReg
    return _model


def classify(ticket):
    """Return (category, confidence)."""
    model = load_model()
    probs = model.predict_proba([ticket])[0]
    idx = probs.argmax()
    return model.classes_[idx], float(probs[idx])


def process_ticket(ticket):
    """Mentor's demo function: returns (category, severity, priority)."""
    category, _ = classify(ticket)
    severity = predict_severity(ticket)
    priority = SEVERITY_TO_PRIORITY[severity]
    return category, severity, priority


def process_ticket_full(ticket, business_impact=None):
    """Same as process_ticket but also returns confidence.

    If business_impact is given ("High"/"Medium"/"Low"), the priority is
    computed with calculate_priority() instead of the severity-only mapping.
    """
    category, confidence = classify(ticket)
    severity = predict_severity(ticket)
    if business_impact:
        priority = calculate_priority(severity, business_impact)
    else:
        priority = SEVERITY_TO_PRIORITY[severity]
    return {
        "category": category,
        "severity": severity,
        "priority": priority,
        "confidence": round(confidence, 2),
    }
