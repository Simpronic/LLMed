import spacy
import json

# ============================================================
# CONFIGURAZIONE
# ============================================================

SRC_JSONL    = "gold_patients_only.jsonl"     # file gold (lo usiamo per i testi)
OUTPUT_JSONL = "predictions_spacy.jsonl"      # output predizioni

# ============================================================
# INIZIALIZZAZIONE SPACY
# ============================================================

# Assicurati di aver fatto:
#   python -m spacy download it_core_news_lg
nlp = spacy.load("it_core_news_lg")

# ============================================================
# ESTRAZIONE ENTITÀ PER CON SPACY
# ============================================================

def extract_per_spans_with_spacy(text: str):
    """
    Estrae tutte le entità di tipo PER con spaCy e
    restituisce una lista di [start, end, "PER"] sul testo originale.
    """
    doc = nlp(text)
    spans = []

    for ent in doc.ents:
        # In spaCy il tipo si legge con ent.label_
        if ent.label_ != "PER":
            continue

        start = ent.start_char
        end = ent.end_char
        spans.append([start, end, "PER"])

    # opzionale: deduplica e ordina
    uniq = {}
    for s, e, label in spans:
        uniq[(s, e, label)] = True

    out = [[s, e, label] for (s, e, label) in sorted(uniq.keys())]
    return out


# ============================================================
# MAIN: GENERAZIONE PREDIZIONI ADERENTI AL GOLD
# ============================================================

def main():
    with open(SRC_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            text = rec["text"]
            rid = rec["id"]

            pred_entities = extract_per_spans_with_spacy(text)

            out_rec = {
                "id": rid,
                "text": text,
                "entities": pred_entities
            }

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"Predizioni spaCy scritte in: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
