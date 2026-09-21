import csv
import math

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
COURSES = [
    "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
    "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
    "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying",
]
COLORS = {"Gryffindor": "#ae0001", "Hufflepuff": "#f0c75e",
          "Ravenclaw": "#222f5b", "Slytherin": "#2a623d"}
FILE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def load_dataset(path, columns=(), labeled=False):
    """read the csv rows, checking the needed columns"""
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except (UnicodeDecodeError, csv.Error) as err:
        raise ValueError(f"{path}: not a valid csv ({err})") from None
    if not rows:
        raise ValueError(f"{path}: no data rows")
    missing = [c for c in columns if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    if labeled:
        rows = [r for r in rows if house(r) in HOUSES]
        if not rows:
            raise ValueError(f"{path}: no labeled rows (Hogwarts House is empty)")
    return rows


def house(row):
    return (row.get("Hogwarts House") or "").strip()


def number(row, column):
    """cell as float, None if empty or not a finite number"""
    try:
        x = float((row.get(column) or "").strip())
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def scores(rows, course, name):
    """scores of one house in one course"""
    values = [number(r, course) for r in rows if house(r) == name]
    return [x for x in values if x is not None]


def pairs(rows, a, b):
    """(a, b, house) of the rows that have both scores"""
    points = [(number(r, a), number(r, b), house(r)) for r in rows]
    return [p for p in points if p[0] is not None and p[1] is not None]


def show(fig, path):
    """open a window, or save a png when there is no display"""
    import matplotlib
    import matplotlib.pyplot as plt
    if matplotlib.get_backend().lower() in FILE_BACKENDS:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"no interactive display: graph saved to {path}")
    else:
        plt.show()
