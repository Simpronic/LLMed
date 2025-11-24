import json

from presidio_analyzer import AnalyzerEngine
# Se hai una configurazione NLP custom (es. spaCy italiano),
# qui importeresti anche NlpEngineProvider e la useresti per costruire l'engine.


# ============================================================
# CONFIGURAZIONE
# ============================================================

SRC_JSONL    = "gold_patients_only.jsonl"     # file gold (lo usiamo per i testi)
OUTPUT_JSONL = "predictions_presidio.jsonl"   # output predizioni


# ============================================================
# INIZIALIZZAZIONE PRESIDIO
# ============================================================

# Caso semplice: AnalyzerEngine con configurazione di default.
# Se hai configurato un NLP engine custom (es. spaCy it_core_news_lg),
# puoi passarlo come parametro nlp_engine=... e indicare supported_languages.
analyzer = AnalyzerEngine()


# ============================================================
# ESTRAZIONE ENTITÀ PER CON PRESIDIO
# ============================================================

def extract_per_spans_with_presidio(text: str):
    """
    Usa Presidio per estrarre entità di tipo PERSON,
    e restituisce una lista di [start, end, "PER"]
    sul testo originale, compatibile col formato usato prima.
    """

    # language deve corrispondere a quello supportato dall'NLP engine.
    # Con la configurazione di default è tipicamente "en".
    # Se hai un engine italiano, usa "it".
    results = analyzer.analyze(
        text=text,
        entities=["PERSON"],   # chiediamo solo PERSON
        language="en"          # cambialo in "it" se il tuo engine supporta l'italiano
    )

    spans = []

    for r in results:
        # Presidio usa entity_type="PERSON"
        if r.entity_type != "PERSON":
            continue

        start = r.start
        end = r.end

        # Mappiamo PERSON -> "PER" per restare compatibili con il gold
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

            # qui usiamo Presidio invece di Stanza
            pred_entities = extract_per_spans_with_presidio(text)

            out_rec = {
                "id": rid,
                "text": text,
                "entities": pred_entities
            }

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"Predizioni Presidio scritte in: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
