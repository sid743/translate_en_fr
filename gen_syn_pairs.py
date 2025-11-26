import random
import argparse
from pathlib import Path

random.seed(42)

# Basic vocab lists
NAMES_EN = ["Alice", "Bob", "Claire", "David", "Emma", "Lucas", "Sophie", "Thomas"]
PLACES_EN = ["the park", "the museum", "the supermarket", "the library", "the office", "the station"]
PLACES_FR = ["le parc", "le musée", "le supermarché", "la bibliothèque", "le bureau", "la gare"]

OBJECTS_EN = ["book", "phone", "car", "laptop", "bag", "table", "chair", "ticket"]
OBJECTS_FR = ["livre", "téléphone", "voiture", "ordinateur", "sac", "table", "chaise", "billet"]

FOODS_EN = ["pizza", "bread", "coffee", "tea", "rice", "pasta", "salad", "cake"]
FOODS_FR = ["pizza", "pain", "café", "thé", "riz", "pâtes", "salade", "gâteau"]

DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

TIMES_EN = ["this morning", "this afternoon", "this evening", "tomorrow", "yesterday"]
TIMES_FR = ["ce matin", "cet après-midi", "ce soir", "demain", "hier"]

ADJ_EN = ["big", "small", "beautiful", "interesting", "difficult", "easy", "important", "expensive"]
ADJ_FR = ["grand", "petit", "magnifique", "intéressant", "difficile", "facile", "important", "cher"]

VERBS_SIMPLE = [
    ("to like", "likes", "like", "aime", "aiment"),           
    ("to want", "wants", "want", "veut", "veulent"),
    ("to see", "sees", "see", "voit", "voient"),
    ("to buy", "buys", "buy", "achète", "achètent"),
    ("to read", "reads", "read", "lit", "lisent"),
    ("to eat", "eats", "eat", "mange", "mangent"),
]

def sample_pair():
    """Return a (english_sentence, french_sentence) pair."""
    t = random.randint(1, 8)

    if t == 1:
        # Simple preference: "Alice likes pizza."
        name = random.choice(NAMES_EN)
        food_en = random.choice(FOODS_EN)
        food_fr = FOODS_FR[FOODS_EN.index(food_en)]
        _, v3sg, _, fr_sg, _ = random.choice(VERBS_SIMPLE)
        en = f"{name} {v3sg} {food_en}."
        fr = f"{name} {fr_sg} la {food_fr}."
        return en, fr

    elif t == 2:
        # Location and time: "On Monday, Alice goes to the park."
        name = random.choice(NAMES_EN)
        place_idx = random.randrange(len(PLACES_EN))
        place_en = PLACES_EN[place_idx]
        place_fr = PLACES_FR[place_idx]
        day_idx = random.randrange(len(DAYS_EN))
        day_en = DAYS_EN[day_idx]
        day_fr = DAYS_FR[day_idx]
        time_idx = random.randrange(len(TIMES_EN))
        time_en = TIMES_EN[time_idx]
        time_fr = TIMES_FR[time_idx]
        en = f"On {day_en} {time_en}, {name} goes to {place_en}."
        fr = f"{day_fr} {time_fr}, {name} va à {place_fr}."
        return en, fr

    elif t == 3:
        # Questions: "Where is the book?" / "Où est le livre ?"
        obj_idx = random.randrange(len(OBJECTS_EN))
        obj_en = OBJECTS_EN[obj_idx]
        obj_fr = OBJECTS_FR[obj_idx]
        en = f"Where is the {obj_en}?"
        fr = f"Où est le {obj_fr} ?"
        return en, fr

    elif t == 4:
        # Negation: "Alice does not like coffee."
        name = random.choice(NAMES_EN)
        food_idx = random.randrange(len(FOODS_EN))
        food_en = FOODS_EN[food_idx]
        food_fr = FOODS_FR[food_idx]
        _, _, _, fr_sg, _ = random.choice(VERBS_SIMPLE)
        en = f"{name} does not like {food_en}."
        fr = f"{name} n'{fr_sg} pas le {food_fr}."
        return en, fr

    elif t == 5:
        # Adjective & noun: "This book is very interesting."
        obj_idx = random.randrange(len(OBJECTS_EN))
        obj_en = OBJECTS_EN[obj_idx]
        obj_fr = OBJECTS_FR[obj_idx]
        adj_idx = random.randrange(len(ADJ_EN))
        adj_en = ADJ_EN[adj_idx]
        adj_fr = ADJ_FR[adj_idx]
        en = f"This {obj_en} is very {adj_en}."
        fr = f"Ce {obj_fr} est très {adj_fr}."
        return en, fr

    elif t == 6:
        # Compound sentence: "Alice reads a book and drinks coffee."
        name = random.choice(NAMES_EN)
        food_idx = random.randrange(len(FOODS_EN))
        food_en = FOODS_EN[food_idx]
        food_fr = FOODS_FR[food_idx]
        en = f"{name} reads a book and drinks {food_en}."
        fr = f"{name} lit un livre et boit du {food_fr}."
        return en, fr

    elif t == 7:
        # Future-like: "Tomorrow, they will visit the museum."
        place_idx = random.randrange(len(PLACES_EN))
        place_en = PLACES_EN[place_idx]
        place_fr = PLACES_FR[place_idx]
        time_idx = random.randrange(len(TIMES_EN))
        time_en = TIMES_EN[time_idx]
        time_fr = TIMES_FR[time_idx]
        en = f"{time_en.capitalize()}, they will visit {place_en}."
        fr = f"{time_fr.capitalize()}, ils visiteront {place_fr}."
        return en, fr

    else:
        # Past-like: "Yesterday, we bought a new car."
        obj_idx = random.randrange(len(OBJECTS_EN))
        obj_en = OBJECTS_EN[obj_idx]
        obj_fr = OBJECTS_FR[obj_idx]
        time_idx = random.randrange(len(TIMES_EN))
        time_en = TIMES_EN[time_idx]
        time_fr = TIMES_FR[time_idx]
        en = f"{time_en.capitalize()}, we bought a new {obj_en}."
        fr = f"{time_fr.capitalize()}, nous avons acheté une nouvelle {obj_fr}."
        return en, fr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=80000,
                        help="Number of sentence pairs to generate")
    parser.add_argument("--output", type=str, default="en_fr_pairs.tsv",
                        help="Output TSV file (en \\t fr)")
    args = parser.parse_args()

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        for _ in range(args.num_pairs):
            en, fr = sample_pair()
            f.write(en.replace("\t", " ") + "\t" + fr.replace("\t", " ") + "\n")

    print(f"Written {args.num_pairs} pairs to {out_path}")

if __name__ == "__main__":
    main()
