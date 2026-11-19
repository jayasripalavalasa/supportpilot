"""Train the ticket classifier (Work 4) and save it to models/."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from classifier import MODEL_PATH, clean_text

# Base phrases per category. The deck's 10-row sample is too small to reach
# the >=90% target, so this is a larger set. Replace/extend with real tickets.
BASE = {
    "Network": [
        "WiFi is not working", "Internet connection is very slow",
        "No internet access in my office", "Cannot connect to the office wifi",
        "Network keeps dropping every few minutes", "LAN cable connected but no network",
        "Unable to reach shared network drive", "Router is down on our floor",
        "Website pages take forever to load", "Ethernet port not detecting connection",
        "DNS error when opening any website", "Wireless signal is very weak",
    ],
    "VPN": [
        "VPN is not connecting", "Unable to access company VPN",
        "VPN connection failure", "Cannot log in to the VPN client",
        "VPN disconnects after a few minutes", "Need VPN access for remote work",
        "VPN authentication failed", "GlobalProtect VPN shows timeout error",
        "Cannot connect to company VPN from home", "VPN client will not launch",
        "Request VPN access for new joiner", "VPN tunnel drops during video calls",
    ],
    "Password": [
        "I forgot my password", "Please reset my password",
        "My account is locked after wrong password attempts", "Password expired and cannot change it",
        "Reset my email password", "Unable to login, password not accepted",
        "Need to unlock my account", "Forgot my domain login password",
        "Password reset link is not working", "Change my password please",
        "Multi factor authentication code not received for login", "Account locked out",
    ],
    "Software": [
        "Install Microsoft Office", "Application installation required",
        "Need Adobe Acrobat installed", "Please install Python on my laptop",
        "Microsoft Teams keeps crashing", "Excel shows error on opening files",
        "Need a license for Visual Studio", "Software update failed to install",
        "Outlook is not opening", "Unable to install Zoom",
        "Application freezes when I open it", "Please uninstall unwanted software",
    ],
    "Hardware": [
        "Laptop keyboard is not working", "Monitor display is not working",
        "Mouse is not responding", "Laptop battery drains very fast",
        "Printer is not printing", "Screen is flickering",
        "Laptop fan makes loud noise", "USB ports are not working",
        "Webcam is not detected", "Headset microphone is broken",
        "Laptop will not turn on", "Docking station not charging laptop",
    ],
    "System": [
        "Operating system is very slow", "Windows update stuck at 50 percent",
        "Laptop shows blue screen error", "Computer restarts automatically",
        "Disk space is full on my computer", "System crashes on startup",
        "Windows not booting properly", "High CPU usage and system hangs",
        "Computer is stuck on loading screen", "Operating system needs reinstallation",
        "Task manager shows memory almost full", "Laptop hangs when multiple apps are open",
    ],
}

PREFIXES = ["", "Hi, ", "Please help: ", "Urgent: ", "Hello team, ", "Issue - "]
SUFFIXES = ["", " please fix", " since morning", " need help"]


def build_dataframe():
    rows = [(t, c) for c, texts in BASE.items() for t in texts]
    return pd.DataFrame(rows, columns=["ticket", "category"])


def augment(df):
    rows = []
    for t, c in zip(df["ticket"], df["category"]):
        for p in PREFIXES:
            for s in SUFFIXES:
                rows.append((f"{p}{t}{s}", c))
    return pd.DataFrame(rows, columns=["ticket", "category"])


def main():
    df = build_dataframe()
    # split on ORIGINAL phrases first so test tickets are never seen in training
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["category"]
    )
    train_df = augment(train_df)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=10)),
    ])
    model.fit(train_df["ticket"], train_df["category"])

    preds = model.predict(test_df["ticket"])
    print(classification_report(test_df["category"], preds))
    print("Accuracy:", round(model.score(test_df["ticket"], test_df["category"]), 3))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()
