"""Deterministic factual answers from a curated static-fact table.

Prove-or-defer: a lookup fires only when the question matches a strict
pattern AND the key exists in the table; anything else returns None and the
task escalates unchanged. Only stable, canonical facts are included — no
populations, office-holders, records, or anything time-varying. A table hit
costs zero tokens and zero model milliseconds.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------- tables ----

# All 118 elements: symbol -> (name, atomic number).
_ELEMENTS = {
    "H": ("hydrogen", 1), "He": ("helium", 2), "Li": ("lithium", 3),
    "Be": ("beryllium", 4), "B": ("boron", 5), "C": ("carbon", 6),
    "N": ("nitrogen", 7), "O": ("oxygen", 8), "F": ("fluorine", 9),
    "Ne": ("neon", 10), "Na": ("sodium", 11), "Mg": ("magnesium", 12),
    "Al": ("aluminium", 13), "Si": ("silicon", 14), "P": ("phosphorus", 15),
    "S": ("sulfur", 16), "Cl": ("chlorine", 17), "Ar": ("argon", 18),
    "K": ("potassium", 19), "Ca": ("calcium", 20), "Sc": ("scandium", 21),
    "Ti": ("titanium", 22), "V": ("vanadium", 23), "Cr": ("chromium", 24),
    "Mn": ("manganese", 25), "Fe": ("iron", 26), "Co": ("cobalt", 27),
    "Ni": ("nickel", 28), "Cu": ("copper", 29), "Zn": ("zinc", 30),
    "Ga": ("gallium", 31), "Ge": ("germanium", 32), "As": ("arsenic", 33),
    "Se": ("selenium", 34), "Br": ("bromine", 35), "Kr": ("krypton", 36),
    "Rb": ("rubidium", 37), "Sr": ("strontium", 38), "Y": ("yttrium", 39),
    "Zr": ("zirconium", 40), "Nb": ("niobium", 41), "Mo": ("molybdenum", 42),
    "Tc": ("technetium", 43), "Ru": ("ruthenium", 44), "Rh": ("rhodium", 45),
    "Pd": ("palladium", 46), "Ag": ("silver", 47), "Cd": ("cadmium", 48),
    "In": ("indium", 49), "Sn": ("tin", 50), "Sb": ("antimony", 51),
    "Te": ("tellurium", 52), "I": ("iodine", 53), "Xe": ("xenon", 54),
    "Cs": ("caesium", 55), "Ba": ("barium", 56), "La": ("lanthanum", 57),
    "Ce": ("cerium", 58), "Pr": ("praseodymium", 59), "Nd": ("neodymium", 60),
    "Pm": ("promethium", 61), "Sm": ("samarium", 62), "Eu": ("europium", 63),
    "Gd": ("gadolinium", 64), "Tb": ("terbium", 65), "Dy": ("dysprosium", 66),
    "Ho": ("holmium", 67), "Er": ("erbium", 68), "Tm": ("thulium", 69),
    "Yb": ("ytterbium", 70), "Lu": ("lutetium", 71), "Hf": ("hafnium", 72),
    "Ta": ("tantalum", 73), "W": ("tungsten", 74), "Re": ("rhenium", 75),
    "Os": ("osmium", 76), "Ir": ("iridium", 77), "Pt": ("platinum", 78),
    "Au": ("gold", 79), "Hg": ("mercury", 80), "Tl": ("thallium", 81),
    "Pb": ("lead", 82), "Bi": ("bismuth", 83), "Po": ("polonium", 84),
    "At": ("astatine", 85), "Rn": ("radon", 86), "Fr": ("francium", 87),
    "Ra": ("radium", 88), "Ac": ("actinium", 89), "Th": ("thorium", 90),
    "Pa": ("protactinium", 91), "U": ("uranium", 92), "Np": ("neptunium", 93),
    "Pu": ("plutonium", 94), "Am": ("americium", 95), "Cm": ("curium", 96),
    "Bk": ("berkelium", 97), "Cf": ("californium", 98),
    "Es": ("einsteinium", 99), "Fm": ("fermium", 100),
    "Md": ("mendelevium", 101), "No": ("nobelium", 102),
    "Lr": ("lawrencium", 103), "Rf": ("rutherfordium", 104),
    "Db": ("dubnium", 105), "Sg": ("seaborgium", 106), "Bh": ("bohrium", 107),
    "Hs": ("hassium", 108), "Mt": ("meitnerium", 109),
    "Ds": ("darmstadtium", 110), "Rg": ("roentgenium", 111),
    "Cn": ("copernicium", 112), "Nh": ("nihonium", 113),
    "Fl": ("flerovium", 114), "Mc": ("moscovium", 115),
    "Lv": ("livermorium", 116), "Ts": ("tennessine", 117),
    "Og": ("oganesson", 118),
}
_NAME_TO_SYMBOL = {name: sym for sym, (name, _z) in _ELEMENTS.items()}
_NAME_TO_SYMBOL["aluminum"] = "Al"   # US spelling
_NAME_TO_SYMBOL["cesium"] = "Cs"
_NAME_TO_SYMBOL["sulphur"] = "S"

_CAPITALS = {
    "afghanistan": "Kabul", "albania": "Tirana", "algeria": "Algiers",
    "argentina": "Buenos Aires", "armenia": "Yerevan", "australia": "Canberra",
    "austria": "Vienna", "azerbaijan": "Baku", "bangladesh": "Dhaka",
    "belarus": "Minsk", "belgium": "Brussels", "bolivia": "Sucre",
    "brazil": "Brasília", "bulgaria": "Sofia", "cambodia": "Phnom Penh",
    "cameroon": "Yaoundé", "canada": "Ottawa", "chile": "Santiago",
    "china": "Beijing", "colombia": "Bogotá", "croatia": "Zagreb",
    "cuba": "Havana", "cyprus": "Nicosia", "czech republic": "Prague",
    "czechia": "Prague", "denmark": "Copenhagen", "ecuador": "Quito",
    "egypt": "Cairo", "estonia": "Tallinn", "ethiopia": "Addis Ababa",
    "finland": "Helsinki", "france": "Paris", "georgia": "Tbilisi",
    "germany": "Berlin", "ghana": "Accra", "greece": "Athens",
    "hungary": "Budapest", "iceland": "Reykjavik", "india": "New Delhi",
    "indonesia": "Jakarta", "iran": "Tehran", "iraq": "Baghdad",
    "ireland": "Dublin", "israel": "Jerusalem", "italy": "Rome",
    "jamaica": "Kingston", "japan": "Tokyo", "jordan": "Amman",
    "kazakhstan": "Astana", "kenya": "Nairobi", "kuwait": "Kuwait City",
    "laos": "Vientiane", "latvia": "Riga", "lebanon": "Beirut",
    "libya": "Tripoli", "lithuania": "Vilnius", "luxembourg": "Luxembourg",
    "malaysia": "Kuala Lumpur", "mexico": "Mexico City", "mongolia": "Ulaanbaatar",
    "morocco": "Rabat", "myanmar": "Naypyidaw", "nepal": "Kathmandu",
    "netherlands": "Amsterdam", "new zealand": "Wellington",
    "nigeria": "Abuja", "north korea": "Pyongyang", "norway": "Oslo",
    "oman": "Muscat", "pakistan": "Islamabad", "panama": "Panama City",
    "paraguay": "Asunción", "peru": "Lima", "philippines": "Manila",
    "poland": "Warsaw", "portugal": "Lisbon", "qatar": "Doha",
    "romania": "Bucharest", "russia": "Moscow", "saudi arabia": "Riyadh",
    "senegal": "Dakar", "serbia": "Belgrade", "singapore": "Singapore",
    "slovakia": "Bratislava", "slovenia": "Ljubljana",
    "south africa": "Pretoria", "south korea": "Seoul", "spain": "Madrid",
    "sri lanka": "Sri Jayawardenepura Kotte", "sweden": "Stockholm",
    "switzerland": "Bern", "syria": "Damascus", "taiwan": "Taipei",
    "tanzania": "Dodoma", "thailand": "Bangkok", "tunisia": "Tunis",
    "turkey": "Ankara", "uganda": "Kampala", "ukraine": "Kyiv",
    "united arab emirates": "Abu Dhabi", "united kingdom": "London",
    "uk": "London", "england": "London", "united states": "Washington, D.C.",
    "usa": "Washington, D.C.", "us": "Washington, D.C.",
    "the united states": "Washington, D.C.", "uruguay": "Montevideo",
    "uzbekistan": "Tashkent", "venezuela": "Caracas", "vietnam": "Hanoi",
    "yemen": "Sanaa", "zambia": "Lusaka", "zimbabwe": "Harare",
}

_CURRENCIES = {
    "japan": "yen", "china": "yuan (renminbi)", "india": "rupee",
    "united kingdom": "pound sterling", "uk": "pound sterling",
    "united states": "US dollar", "usa": "US dollar", "us": "US dollar",
    "france": "euro", "germany": "euro", "italy": "euro", "spain": "euro",
    "russia": "ruble", "brazil": "real", "mexico": "peso",
    "south korea": "won", "switzerland": "Swiss franc", "canada": "Canadian dollar",
    "australia": "Australian dollar", "sweden": "krona", "norway": "krone",
    "denmark": "krone", "turkey": "lira", "thailand": "baht",
    "vietnam": "dong", "indonesia": "rupiah", "south africa": "rand",
    "israel": "shekel", "saudi arabia": "riyal", "egypt": "Egyptian pound",
    "nigeria": "naira", "kenya": "Kenyan shilling", "poland": "zloty",
    "argentina": "Argentine peso", "philippines": "Philippine peso",
    "malaysia": "ringgit", "singapore": "Singapore dollar",
    "bangladesh": "taka", "pakistan": "Pakistani rupee",
}

_FORMULAS = {
    "water": "H2O", "table salt": "NaCl", "salt": "NaCl",
    "carbon dioxide": "CO2", "methane": "CH4", "ammonia": "NH3",
    "glucose": "C6H12O6", "ozone": "O3", "hydrogen peroxide": "H2O2",
    "sulfuric acid": "H2SO4", "carbon monoxide": "CO",
}

_AUTHORS = {
    "romeo and juliet": "William Shakespeare", "hamlet": "William Shakespeare",
    "macbeth": "William Shakespeare", "othello": "William Shakespeare",
    "king lear": "William Shakespeare",
    "a midsummer night's dream": "William Shakespeare",
    "pride and prejudice": "Jane Austen", "emma": "Jane Austen",
    "sense and sensibility": "Jane Austen",
    "1984": "George Orwell", "nineteen eighty-four": "George Orwell",
    "animal farm": "George Orwell",
    "war and peace": "Leo Tolstoy", "anna karenina": "Leo Tolstoy",
    "the odyssey": "Homer", "the iliad": "Homer",
    "don quixote": "Miguel de Cervantes",
    "the great gatsby": "F. Scott Fitzgerald",
    "to kill a mockingbird": "Harper Lee",
    "moby-dick": "Herman Melville", "moby dick": "Herman Melville",
    "crime and punishment": "Fyodor Dostoevsky",
    "the brothers karamazov": "Fyodor Dostoevsky",
    "great expectations": "Charles Dickens",
    "a tale of two cities": "Charles Dickens",
    "oliver twist": "Charles Dickens",
    "jane eyre": "Charlotte Brontë", "wuthering heights": "Emily Brontë",
    "frankenstein": "Mary Shelley", "dracula": "Bram Stoker",
    "the adventures of huckleberry finn": "Mark Twain",
    "the adventures of tom sawyer": "Mark Twain",
    "on the origin of species": "Charles Darwin",
    "the origin of species": "Charles Darwin",
    "harry potter": "J.K. Rowling",
    "the lord of the rings": "J.R.R. Tolkien", "the hobbit": "J.R.R. Tolkien",
    "one hundred years of solitude": "Gabriel García Márquez",
    "the old man and the sea": "Ernest Hemingway",
    "a farewell to arms": "Ernest Hemingway",
    "the catcher in the rye": "J.D. Salinger",
    "brave new world": "Aldous Huxley",
    "the divine comedy": "Dante Alighieri",
    "ulysses": "James Joyce", "lolita": "Vladimir Nabokov",
    "the picture of dorian gray": "Oscar Wilde",
}

_PAINTERS = {
    "mona lisa": "Leonardo da Vinci", "the last supper": "Leonardo da Vinci",
    "starry night": "Vincent van Gogh", "the starry night": "Vincent van Gogh",
    "sunflowers": "Vincent van Gogh",
    "the scream": "Edvard Munch", "guernica": "Pablo Picasso",
    "the persistence of memory": "Salvador Dalí",
    "girl with a pearl earring": "Johannes Vermeer",
    "the birth of venus": "Sandro Botticelli",
    "american gothic": "Grant Wood",
    "the night watch": "Rembrandt",
    "water lilies": "Claude Monet", "impression, sunrise": "Claude Monet",
}

# One-off canonical facts, matched by ALL keywords present (lowercase).
# Only stable textbook facts; nothing time-varying or contested.
_SIMPLE = [
    (("boiling", "water", "celsius"), "100"),
    (("boiling", "water", "fahrenheit"), "212"),
    (("freezing", "water", "celsius"), "0"),
    (("freezing", "water", "fahrenheit"), "32"),
    (("how many", "planets"), "eight (8)"),
    (("largest", "planet"), "Jupiter"),
    (("smallest", "planet"), "Mercury"),
    (("closest", "planet", "sun"), "Mercury"),
    (("red planet",), "Mars"),
    (("hottest", "planet"), "Venus"),
    (("how many", "continents"), "seven (7)"),
    (("largest", "ocean"), "the Pacific Ocean"),
    (("smallest", "ocean"), "the Arctic Ocean"),
    (("tallest", "mountain"), "Mount Everest"),
    (("highest", "mountain"), "Mount Everest"),
    (("largest", "country", "area"), "Russia"),
    (("largest", "country", "world"), "Russia"),
    (("smallest", "country"), "Vatican City"),
    (("largest", "mammal"), "the blue whale"),
    (("largest", "animal"), "the blue whale"),
    (("fastest", "land animal"), "the cheetah"),
    (("hardest", "natural"), "diamond"),
    (("how many", "bones", "adult"), "206"),
    (("chambers", "human heart"), "four (4)"),
    (("chambers", "the heart"), "four (4)"),
    (("how many", "elements", "periodic"), "118"),
    (("gas", "plants", "photosynthesis"), "carbon dioxide"),
    (("most abundant", "gas", "atmosphere"), "nitrogen"),
    (("powerhouse", "cell"), "the mitochondria"),
    (("smallest", "prime"), "2 (two)"),
    (("first", "person", "moon"), "Neil Armstrong"),
    (("first", "man", "moon"), "Neil Armstrong"),
    (("first", "person", "space"), "Yuri Gagarin"),
    (("first", "president", "united states"), "George Washington"),
    (("discovered", "penicillin"), "Alexander Fleming"),
    (("theory", "general relativity"), "Albert Einstein"),
    (("theory", "evolution", "natural selection"), "Charles Darwin"),
    (("speed", "light", "vacuum"), "299,792,458 meters per second (about 3.0 x 10^8 m/s)"),
    (("square root", "144"), "12 (twelve)"),
]

# ------------------------------------------------------------- matching ----

_CAPITAL_Q = re.compile(r"(?i)\bcapital(?:\s+city)?\s+of\s+(?:the\s+)?([a-z .']+?)\s*[?.!]?\s*$")
_ELEM_BY_SYMBOL = re.compile(r"(?i)\belement\b[^?.]*?\bsymbol\s+['\"`]?([A-Za-z]{1,2})['\"`]?")
_SYMBOL_OF = re.compile(r"(?i)\b(?:chemical\s+)?symbol\s+(?:for|of)\s+(?:the\s+element\s+)?['\"`]?([a-z]+)['\"`]?\s*[?.!]?\s*$")
_ATOMIC_NUM = re.compile(r"(?i)\batomic\s+number\s+of\s+['\"`]?([a-z]+)['\"`]?\s*[?.!]?\s*$")
_CURRENCY_Q = re.compile(r"(?i)\bcurrency\s+(?:of|used\s+in)\s+(?:the\s+)?([a-z .']+?)\s*[?.!]?\s*$")
_FORMULA_Q = re.compile(r"(?i)\bchemical\s+formula\s+(?:for|of)\s+([a-z ]+?)\s*[?.!]?\s*$")
_WROTE_Q = re.compile(r"(?i)\b(?:who\s+wrote|author\s+of|who\s+is\s+the\s+author\s+of)\b")
_PAINTED_Q = re.compile(r"(?i)\b(?:who\s+painted|artist\s+(?:of|behind)|who\s+created\s+the\s+painting)\b")

# Never answer anything time-varying from a table.
_VOLATILE = re.compile(
    r"(?i)\b(current|currently|today|now|latest|newest|this\s+year|202\d|"
    r"population|how\s+old|net\s+worth|price|record\s+holder|reigning|ceo|"
    r"president\s+of\b(?!.*first)|prime\s+minister)\b")


def lookup(prompt: str) -> str | None:
    """A single canonical fact, or None. Multi-part prompts defer."""
    if _VOLATILE.search(prompt):
        return None
    if prompt.count("?") > 1:          # multi-question prompts -> model
        return None
    low = prompt.lower()

    m = _CAPITAL_Q.search(prompt)
    if m:
        return _CAPITALS.get(m.group(1).strip().lower())

    m = _ELEM_BY_SYMBOL.search(prompt)
    if m:
        sym = m.group(1)
        # Symbols are case-sensitive in the table; try exact then title-case.
        hit = _ELEMENTS.get(sym) or _ELEMENTS.get(sym.capitalize())
        if hit:
            return hit[0]
        return None

    m = _SYMBOL_OF.search(prompt)
    if m:
        return _NAME_TO_SYMBOL.get(m.group(1).strip().lower())

    m = _ATOMIC_NUM.search(prompt)
    if m:
        sym = _NAME_TO_SYMBOL.get(m.group(1).strip().lower())
        return str(_ELEMENTS[sym][1]) if sym else None

    m = _CURRENCY_Q.search(prompt)
    if m:
        return _CURRENCIES.get(m.group(1).strip().lower())

    m = _FORMULA_Q.search(prompt)
    if m:
        return _FORMULAS.get(m.group(1).strip().lower())

    if _WROTE_Q.search(prompt):
        for title, author in _AUTHORS.items():
            if title in low:
                return author
        return None

    if _PAINTED_Q.search(prompt):
        for title, painter in _PAINTERS.items():
            if title in low:
                return painter
        return None

    for keywords, answer in _SIMPLE:
        if all(k in low for k in keywords):
            return answer
    return None
