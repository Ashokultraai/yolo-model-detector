"""Global label space shared by all three models.

Each model only knows a subset of these classes, but the backend merges every
model's output into this single unified label space so the app sees one list.

If you change the object set, change it HERE and in each model's class list.
"""

# The unified set of objects the whole system can detect.
GLOBAL_LABELS = ["mobile", "fan", "clock", "watch", "laptop"]

# A stable color (R, G, B) per global label, used by the app / debug drawing.
LABEL_COLORS = {
    "mobile": (255, 99, 71),    # tomato
    "fan":    (30, 144, 255),   # dodger blue
    "clock":  (60, 179, 113),   # medium sea green
    "watch":  (255, 165, 0),    # orange
    "laptop": (147, 112, 219),  # medium purple
}


def global_index(label: str) -> int:
    return GLOBAL_LABELS.index(label)
