import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ============================================================
# CONFIGURAZIONE
# ============================================================

SRC_JSONL    = "gold_patients_only.jsonl"      # input (solo testo/id)
OUTPUT_JSONL = "predictions_hf.jsonl"          # output predizioni in formato gold

# Percorso del TUO modello NER fine-tuned
MODEL_DIR = r"C:\Users\Marco Di Fiandra\Desktop\NefrocenterProjects\LLMed\bert-kind-ner\checkpoint-14163"

# ============================================================
# INIZIALIZZAZIONE MODELLO NER HF
# ============================================================

device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",   # unisce B-/I- token in un'unica entità
    device=device,
)

# ============================================================
# ESTRAZIONE ENTITÀ PER DAL MODELLO HF
# ============================================================

def extract_per_spans_hf(text: str):
    """
    Estrae le entità PER dal modello HF e restituisce
    una lista di [start, end, "PER"] in formato gold.
    Nessun filtro, nessuna regex, nessun NLI.
    """
    spans = []
    outputs = ner_pipeline(text)

    for ent in outputs:
        # l’entità aggregata viene salvata in "entity_group"
        if "PER" not in ent.get("entity_group", ""):
            continue

        start = ent["start"]
        end = ent["end"]
        spans.append([start, end, "PER"])

    # dedup e sort
    uniq = {(s, e, l): True for s, e, l in spans}
    out = [[s, e, l] for (s, e, l) in sorted(uniq.keys())]

    return out


# ============================================================
# MAIN: GENERA PREDIZIONI IN FORMATO GOLD
# ============================================================

def main():
    with open(SRC_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            rec  = json.loads(line)
            text = rec["text"]
            rid  = rec["id"]

            pred_entities = extract_per_spans_hf(text)

            out = {
                "id": rid,
                "text": text,
                "entities": pred_entities
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[HF ONLY] Predizioni scritte in: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
