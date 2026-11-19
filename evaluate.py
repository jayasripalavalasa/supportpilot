"""Evaluation (slides 51-53): classification and severity accuracy vs targets.

Replace the lists below with your own ~100 labelled tickets for the final report.
"""
from classifier import classify, predict_severity

# (ticket text, expected category, expected severity)
TEST = [
    ("VPN is not connecting and I have an important client meeting in 30 minutes", "VPN", "High"),
    ("Please reset my password", "Password", "Low"),
    ("Wifi keeps disconnecting in the conference room", "Network", "Low"),
    ("The whole company cannot access email, server down", "System", "Critical"),
    ("Laptop screen is flickering", "Hardware", "Low"),
    ("Install Microsoft Office on my new laptop", "Software", "Low"),
    ("Internet is very slow today", "Network", "Medium"),
    ("Possible security breach, unknown login on my account", "Password", "Critical"),
    ("Urgent: printer is not printing and I have a deadline", "Hardware", "High"),
    ("Outlook shows an error when I open it", "Software", "Medium"),
    ("Windows update stuck, computer is very slow", "System", "Medium"),
    ("Cannot connect to company VPN from home", "VPN", "Low"),
]


def main():
    cls_ok = sum(classify(t)[0] == c for t, c, _ in TEST)
    sev_ok = sum(predict_severity(t) == s for t, _, s in TEST)
    n = len(TEST)
    cls_acc, sev_acc = cls_ok / n * 100, sev_ok / n * 100
    print(f"Classification accuracy: {cls_acc:.1f}%  (target >= 90%)  "
          f"{'PASS' if cls_acc >= 90 else 'FAIL'}")
    print(f"Severity accuracy:       {sev_acc:.1f}%  (target >= 85%)  "
          f"{'PASS' if sev_acc >= 85 else 'FAIL'}")
    for t, c, s in TEST:
        pc, conf = classify(t)
        ps = predict_severity(t)
        flag = "" if (pc == c and ps == s) else "   <-- mismatch"
        print(f"- {t[:55]:55} {pc:9}({conf:.0%}) {ps:8}{flag}")


if __name__ == "__main__":
    main()
