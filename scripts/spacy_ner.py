#!/usr/bin/env python3
"""Isolated spaCy NER extractor (runs in its own venv, never the app env).

stdin: the raw text to analyse. stdout: JSON list of {"label","text"} with
labels already mapped to the contest vocabulary. Any failure -> empty list.
"""
import json
import sys

MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "DATE": "date",
}


def main():
    text = sys.stdin.read()
    try:
        import spacy

        nlp = spacy.load("en_core_web_md")
        doc = nlp(text)
        out = []
        for ent in doc.ents:
            label = MAP.get(ent.label_)
            if label:
                out.append({"label": label, "text": ent.text.strip()})
        print(json.dumps(out))
    except Exception:
        print("[]")


if __name__ == "__main__":
    main()
