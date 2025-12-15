import random
import argparse
from pathlib import Path

# ----------------------------
# MINI GRAMMAR ENGINE & VOCAB
# ----------------------------

def starts_with_vowel_sound(fr_word: str) -> bool:
    """
    Very lightweight heuristic for French elision.
    We treat leading vowels + common accented vowels + 'h' as vowel-ish.
    (Not perfect French phonetics, but good enough for synthetic data.)
    """
    w = fr_word.strip().lower()
    if not w:
        return False
    return w[0] in "aeiouyhàâäéèêëîïôöùûüÿœ"

def get_determiner(gender: str, word_starts_vowel: bool, determiner_type: str = "definite") -> str:
    """
    Returns determiners with trailing space where appropriate.
    definite:   le / la / l'
    indefinite: un / une
    demonstrative: ce / cet / cette
    """
    if determiner_type == "definite":
        if word_starts_vowel:
            return "l'"
        return "le " if gender == "m" else "la "
    if determiner_type == "indefinite":
        return "un " if gender == "m" else "une "
    if determiner_type == "demonstrative":
        # "cet" used for masculine before vowel/h muet; we approximate with word_starts_vowel
        if gender == "m" and word_starts_vowel:
            return "cet "
        return "ce " if gender == "m" else "cette "
    return ""

# Nouns: (EN, FR, Gender(m/f))
NOUNS = {
    "tech": [
        ("server", "serveur", "m"),
        ("database", "base de données", "f"),
        ("algorithm", "algorithme", "m"),
        ("application", "application", "f"),
        ("screen", "écran", "m"),
        ("keyboard", "clavier", "m"),
        ("password", "mot de passe", "m"),
        ("update", "mise à jour", "f"),
    ],
    "business": [
        ("meeting", "réunion", "f"),
        ("contract", "contrat", "m"),
        ("client", "client", "m"),
        ("deadline", "date limite", "f"),
        ("budget", "budget", "m"),
        ("presentation", "présentation", "f"),
        ("report", "rapport", "m"),
        ("invoice", "facture", "f"),
    ],
    "travel": [
        ("flight", "vol", "m"),
        ("hotel", "hôtel", "m"),
        ("passport", "passeport", "m"),
        ("reservation", "réservation", "f"),
        ("luggage", "bagage", "m"),
        ("ticket", "billet", "m"),
        ("destination", "destination", "f"),
    ],
}

# Adjectives: (EN, FR_masc, FR_fem, Pre_Noun)
# Pre_Noun=True for BAGS-like adjectives (go before noun often)
ADJECTIVES = [
    ("important", "important", "importante", False),
    ("new", "nouveau", "nouvelle", True),
    ("urgent", "urgent", "urgente", False),
    ("available", "disponible", "disponible", False),
    ("expensive", "cher", "chère", False),
    ("complex", "complexe", "complexe", False),
    ("fast", "rapide", "rapide", False),
    ("secure", "sécurisé", "sécurisée", False),
    ("global", "mondial", "mondiale", False),
    ("small", "petit", "petite", True),
]

# Verbs: (EN_inf, EN_3sg, EN_past, FR_inf, FR_3sg, FR_past_participle)
VERBS = [
    ("check", "checks", "checked", "vérifier", "vérifie", "vérifié"),
    ("update", "updates", "updated", "mettre à jour", "met à jour", "mis à jour"),
    ("send", "sends", "sent", "envoyer", "envoie", "envoyé"),
    ("delete", "deletes", "deleted", "supprimer", "supprime", "supprimé"),
    ("confirm", "confirms", "confirmed", "confirmer", "confirme", "confirmé"),
    ("cancel", "cancels", "cancelled", "annuler", "annule", "annulé"),
    ("review", "reviews", "reviewed", "examiner", "examine", "examiné"),
]

ADVERBS = [
    ("immediately", "immédiatement"),
    ("quickly", "rapidement"),
    ("carefully", "soigneusement"),
    ("automatically", "automatiquement"),
    ("manually", "manuellement"),
]

TEMPORAL = [
    ("tomorrow morning", "demain matin"),
    ("next week", "la semaine prochaine"),
    ("as soon as possible", "dès que possible"),
    ("later today", "plus tard aujourd'hui"),
    ("before the meeting", "avant la réunion"),
]

# ----------------------------
# GENERATION
# ----------------------------

def get_random_noun(category: str | None = None):
    cat = category if category else random.choice(list(NOUNS.keys()))
    n_en, n_fr, gender = random.choice(NOUNS[cat])
    vowel = starts_with_vowel_sound(n_fr)
    return n_en, n_fr, gender, vowel

def agree_adj(gender: str, adj_fr_m: str, adj_fr_f: str) -> str:
    return adj_fr_f if gender == "f" else adj_fr_m

def make_noun_phrase_fr(det_fr: str, noun_fr: str, adj_fr: str, pre_noun: bool) -> str:
    # det_fr already contains trailing space or is "l'"
    if pre_noun:
        # "le nouveau rapport" or "l'algorithme nouveau" (not perfect, but ok)
        if det_fr == "l'":
            # l' + adjective is not generally used; keep "l'" + noun + adj if pre_noun True
            # to avoid "l'nouveau", we force post-noun in this edge case
            return f"{det_fr}{noun_fr} {adj_fr}"
        return f"{det_fr}{adj_fr} {noun_fr}"
    return f"{det_fr}{noun_fr} {adj_fr}"

def generate_sentence_pair():
    t = random.randint(1, 8)  # more templates

    subj_en, subj_fr = random.choice([
        ("I", "je"),
        ("you", "tu"),
        ("we", "nous"),
        ("they", "ils"),
    ])

    int_en, int_fr = random.choice([
        ("very", "très"),
        ("really", "vraiment"),
        ("quite", "assez"),
    ])

    # 1) Polite request
    if t == 1:
        v_en_inf, _, _, v_fr_inf, _, _ = random.choice(VERBS)
        n_en, n_fr, gender, vowel = get_random_noun("business")
        adj_en, adj_fr_m, adj_fr_f, pre_noun = random.choice(ADJECTIVES)
        adv_en, adv_fr = random.choice(ADVERBS)

        det_fr = get_determiner(gender, vowel, "definite")
        adj_fr = agree_adj(gender, adj_fr_m, adj_fr_f)
        noun_phrase_fr = make_noun_phrase_fr(det_fr, n_fr, adj_fr, pre_noun)

        en = f"Please {v_en_inf} the {adj_en} {n_en} {adv_en}."
        fr = f"Veuillez {v_fr_inf} {noun_phrase_fr} {adv_fr}."
        return en, fr

    # 2) Conditional
    elif t == 2:
        n1_en, n1_fr, g1, v1 = get_random_noun("tech")
        n2_en, n2_fr, g2, v2 = get_random_noun("tech")
        adv_en, adv_fr = random.choice(ADVERBS)

        det1 = get_determiner(g1, v1, "definite")
        det2 = get_determiner(g2, v2, "definite")

        en = f"If the {n1_en} is available, the {n2_en} updates {adv_en}."
        fr = f"Si {det1}{n1_fr} est disponible, {det2}{n2_fr} se met à jour {adv_fr}."
        return en, fr

    # 3) Past action + time phrase (adds variety)
    elif t == 3:
        context = random.choice(["business", "travel"])
        subj_n_en, subj_n_fr, g_subj, v_subj = get_random_noun(context)
        obj_en, obj_fr, g_obj, v_obj = get_random_noun(context)
        _, _, v_past_en, _, _, v_past_fr = random.choice(VERBS)
        temp_en, temp_fr = random.choice(TEMPORAL)

        det_subj = get_determiner(g_subj, v_subj, "definite")
        det_obj = get_determiner(g_obj, v_obj, "definite")

        en = f"{temp_en.capitalize()}, the {subj_n_en} {v_past_en} the {obj_en}."
        fr = f"{temp_fr.capitalize()}, {det_subj}{subj_n_fr} a {v_past_fr} {det_obj}{obj_fr}."
        return en, fr

    # 4) Modal with varying subject
    elif t == 4:
        v_en_inf, _, _, v_fr_inf, _, _ = random.choice(VERBS)
        n_en, n_fr, gender, vowel = get_random_noun("business")
        adv_en, adv_fr = random.choice(ADVERBS)

        det_fr = get_determiner(gender, vowel, "definite")
        en = f"{subj_en.capitalize()} must {v_en_inf} the {n_en} {adv_en}."
        fr = f"{subj_fr.capitalize()} dois {v_fr_inf} {det_fr}{n_fr} {adv_fr}."
        return en, fr

    # 5) Descriptive statement with intensifier + "thinks that" (huge variety)
    elif t == 5:
        n_en, n_fr, gender, vowel = get_random_noun("tech")
        adj_en, adj_fr_m, adj_fr_f, _ = random.choice(ADJECTIVES)

        det_fr = get_determiner(gender, vowel, "definite")
        adj_fr = agree_adj(gender, adj_fr_m, adj_fr_f)

        en = f"{subj_en.capitalize()} thinks the {n_en} is {int_en} {adj_en}."
        fr = f"{subj_fr.capitalize()} pense que {det_fr}{n_fr} est {int_fr} {adj_fr}."
        return en, fr

    # 6) Future plan with varying subject + time phrase
    elif t == 6:
        temp_en, temp_fr = random.choice(TEMPORAL)
        v_en_inf, _, _, v_fr_inf, _, _ = random.choice(VERBS)
        n_en, n_fr, gender, vowel = get_random_noun("tech")
        adj_en, adj_fr_m, adj_fr_f, pre_noun = random.choice(ADJECTIVES)

        det_fr = get_determiner(gender, vowel, "definite")
        adj_fr = agree_adj(gender, adj_fr_m, adj_fr_f)
        noun_phrase_fr = make_noun_phrase_fr(det_fr, n_fr, adj_fr, pre_noun)

        en = f"{temp_en.capitalize()}, {subj_en} will {v_en_inf} the {adj_en} {n_en}."
        fr = f"{temp_fr.capitalize()}, {subj_fr} vais {v_fr_inf} {noun_phrase_fr}."
        return en, fr

    # 7) Question form (adds lots of uniques)
    elif t == 7:
        v_en_inf, _, _, v_fr_inf, _, _ = random.choice(VERBS)
        n_en, n_fr, gender, vowel = get_random_noun(random.choice(["tech", "business", "travel"]))
        det_fr = get_determiner(gender, vowel, "definite")

        en = f"Can {subj_en} {v_en_inf} the {n_en}?"
        fr = f"Est-ce que {subj_fr} peux {v_fr_inf} {det_fr}{n_fr} ?"
        return en, fr

    # 8) Negation (adds lots of uniques)
    else:
        n_en, n_fr, gender, vowel = get_random_noun(random.choice(["tech", "business", "travel"]))
        adj_en, adj_fr_m, adj_fr_f, _ = random.choice(ADJECTIVES)
        det_fr = get_determiner(gender, vowel, "definite")
        adj_fr = agree_adj(gender, adj_fr_m, adj_fr_f)

        en = f"The {n_en} is not {adj_en}."
        fr = f"{det_fr.capitalize()}{n_fr} n'est pas {adj_fr}."
        return en, fr


# ----------------------------
# MAIN
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=40000, help="Number of sentence pairs to generate")
    parser.add_argument("--output", type=str, default="synthetic_en_fr.tsv", help="Output TSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_attempts_multiplier", type=int, default=50,
                        help="Safety: max attempts = num_pairs * this multiplier")
    parser.add_argument("--progress_every", type=int, default=5000, help="Progress print frequency (pairs)")
    args = parser.parse_args()

    random.seed(args.seed)

    out_path = Path(args.output)
    print(f"Generating {args.num_pairs} synthetic EN-FR pairs...")
    print(f"Output: {out_path.resolve()}")

    seen = set()
    count = 0
    attempts = 0
    max_attempts = max(10_000, args.num_pairs * args.max_attempts_multiplier)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        while count < args.num_pairs and attempts < max_attempts:
            attempts += 1
            en, fr = generate_sentence_pair()
            key = (en, fr)  # safer than only en
            if key in seen:
                continue
            seen.add(key)
            f.write(en + "\t" + fr + "\n")
            count += 1

            if args.progress_every > 0 and count % args.progress_every == 0:
                print(f"  {count}/{args.num_pairs} pairs generated (attempts={attempts})")

    if count < args.num_pairs:
        raise RuntimeError(
            f"Could only generate {count} unique pairs after {attempts} attempts. "
            f"Try increasing max_attempts_multiplier or expanding templates/vocab."
        )

    print(f"Done. Saved {count} pairs to {out_path}")

if __name__ == "__main__":
    main()
