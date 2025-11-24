import json
import torch
import regex as re
from functools import lru_cache

import stanza
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline,
)

# ============================================================
# CONFIGURAZIONE
# ============================================================

SRC_JSONL    = "gold_patients_only.jsonl"              # gold
OUTPUT_JSONL = "predictions_system_ensemble.jsonl"     # predizioni ensemble

# modello HF NER (il tuo fine-tunato o uno pubblico)
MODEL_DIR_NER = r"C:\Users\Marco Di Fiandra\Desktop\NefrocenterProjects\LLMed\bert-kind-ner\checkpoint-14163"

# modello NLI per zero-shot (persona vs farmaco/altro)
MODEL_NLI = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

device_ner = 0 if torch.cuda.is_available() else -1
device_nli = 0 if torch.cuda.is_available() else -1

# ============================================================
# REGEX DI DOMINIO
# ============================================================

# blocco anagrafico tipo:
#   Paziente : ABAGNALE ANNA ... nato il / data nascita
ANAGRAFIC_REGEX = re.compile(
    r'(?P<prefix>'
    r'(?:Paziente\s*:?|Paziente|'
    r'Sig\.?\.?r?|Sig\.?\.?ra|Sig\.?\.?na|'
    r'Signor(?:e)?|Signora)'
    r'\s+)'
    r'(?P<names>.+?)'
    r'(?P<suffix>'
    r'(?:\s+nato\b|\s+nata\b|\s+data\s+nascita\b))',
    flags=re.IGNORECASE
)

# nomi in MAIUSCOLO in contesto anagrafico (abbastanza generico)
UPPER_NAME = re.compile(
    r'\b([A-ZÀ-Ý][A-ZÀ-Ý]+(?:\s+[A-ZÀ-Ý][A-ZÀ-Ý]+)+)\b'
)

ANAGRAFIC_CONTEXT_BEFORE = re.compile(
    r'(paziente|sig\.?r?|sig\.?ra|sig\.?na|signor\w*)',
    flags=re.IGNORECASE
)

ANAGRAFIC_CONTEXT_AFTER = re.compile(
    r'(nato|nata|data\s+nascita|sesso)',
    flags=re.IGNORECASE
)

# codice fiscale italiano (per anonimizzazione / pulizia, opzionale)
CF_REGEX = re.compile(
    r'\b'
    r'[A-Z]{6}'        # cognome + nome (6 lettere)
    r'\d{2}'           # anno
    r'[A-EHLMPRST]'    # mese
    r'\d{2}'           # giorno
    r'[A-Z]'           # iniziale comune
    r'\d{3}'           # codice comune
    r'[A-Z]'           # carattere di controllo
    r'\b'
)


def anonimize_CF(text: str) -> str:
    """Sostituisce eventuali codici fiscali con ### (se vuoi pre-pulire)."""
    return CF_REGEX.sub("###", text)


# ============================================================
# MODELLI
# ============================================================

# HF NER
tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_DIR_NER)
model_ner = AutoModelForTokenClassification.from_pretrained(MODEL_DIR_NER)

ner_pipeline = pipeline(
    "token-classification",
    model=model_ner,
    tokenizer=tokenizer_ner,
    aggregation_strategy="simple",   # unisce B-/I- token in un'unica entità
    device=device_ner,
)

# Stanza NER (italiano)
nlp = stanza.Pipeline(lang='it', processors='tokenize,ner', use_gpu=(device_ner == 0))

# Zero-shot NLI
nli_pipeline = pipeline(
    "zero-shot-classification",
    model=MODEL_NLI,
    device=device_nli
)

# ============================================================
# UTILITY GENERALI
# ============================================================

def overlaps(span_a, span_b) -> bool:
    """
    True se gli intervalli [start, end) si sovrappongono su almeno 1 carattere.
    span = [start, end, "PER"]
    """
    sa, ea, _ = span_a
    sb, eb, _ = span_b
    return not (ea <= sb or eb <= sa)


def overlaps_with_any(span, span_list) -> bool:
    return any(overlaps(span, other) for other in span_list)


def dedup_and_sort(spans):
    """
    Deduplica e ordina per start, poi end.
    spans: lista di [start, end, label]
    """
    uniq = {(int(s), int(e), str(l)) for (s, e, l) in spans}
    return [[s, e, l] for (s, e, l) in sorted(uniq)]


# ============================================================
# NLI: PERSONA vs FARMACO/ALTRO (solo per HF-only)
# ============================================================

@lru_cache(maxsize=10_000)
def is_person_not_drug(token_text: str, context_text: str) -> bool:
    """
    Usa il modello zero-shot per decidere se un token è un nome di persona.
    Ritorna True se il modello ritiene "HumanPersonName" la label top.
    """
    token_text = token_text.strip()
    if not token_text:
        return False

    # euristica veloce: se contiene cifre, non è un nome
    if any(ch.isdigit() for ch in token_text):
        return False

    prompt = (
        f"In questo referto clinico, l'espressione '<ENT>{token_text}</ENT>' "
        f"nel seguente testo: '{context_text}' indica un nome di persona, "
        "un farmaco o qualcos'altro?"
    )

    result = nli_pipeline(
        prompt,
        candidate_labels=[
            "HumanPersonName",
            "DrugOrMedication",
            "BodyPartOrOrgan",
            "OtherConcept",
        ]
    )

    best_label = result["labels"][0]
    # best_score = result["scores"][0]  # se vuoi soglie in futuro

    return best_label == "HumanPersonName"


# ============================================================
# ESTRAZIONE SPAN DA HEADER ANAGRAFICO E MAIUSCOLI
# ============================================================

def extract_header_spans(text: str):
    """
    Usa il blocco anagrafico 'Paziente/Sig.r ... nato/data nascita'
    per produrre span [start, end, "PER"] direttamente sul testo.
    """
    spans = []

    for m in ANAGRAFIC_REGEX.finditer(text):
        start = m.start("names")
        end = m.end("names")
        span_text = text[start:end]
        if span_text.strip():
            spans.append([start, end, "PER"])

    # pattern più generico: blocchi MAIUSCOLI in contesto anagrafico
    for m in UPPER_NAME.finditer(text):
        start, end = m.span(1)
        before = text[max(0, start - 60):start]
        after = text[end:end + 60]

        if (ANAGRAFIC_CONTEXT_BEFORE.search(before) or
                ANAGRAFIC_CONTEXT_AFTER.search(after)):
            spans.append([start, end, "PER"])

    return dedup_and_sort(spans)


# ============================================================
# STANZA: PER ad alta precisione
# ============================================================

def prediction_stanza(text: str):
    """
    Estrae le entità PER dal modello Stanza e restituisce
    una lista di [start, end, "PER"].
    """
    spans = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.type != "PER":
            continue
        spans.append([ent.start_char, ent.end_char, "PER"])

    return dedup_and_sort(spans)


# ============================================================
# HF: PER ad alta recall (senza filtro)
# ============================================================

def prediction_hf_raw(text: str):
    """
    Estrae le entità PER dal modello HF (token-classification)
    SENZA filtraggio NLI. Restituisce [start, end, "PER"].
    """
    spans = []
    outputs = ner_pipeline(text)

    for ent in outputs:
        if "PER" not in ent.get("entity_group", ""):
            continue
        start = ent["start"]
        end = ent["end"]
        spans.append([start, end, "PER"])

    return dedup_and_sort(spans)


# ============================================================
# ENSEMBLE: HEADER + STANZA + HF (+ NLI su HF-only)
# ============================================================

def ensemble_predict(text: str):
    """
    Ensemble finale:
    1) Header anagrafico (regex)  -> sempre incluso
    2) Stanza PER                 -> sempre incluse
    3) HF PER che si sovrappongono a header o stanza -> incluse senza NLI
    4) HF-only PER (non sovrapposte) -> incluse solo se NLI le classifica come persona
    """
    # opzionale: prima rimuovi CF se non vuoi che disturbino i modelli
    clean_text = anonimize_CF(text)

    header_spans = extract_header_spans(clean_text)
    stanza_spans = prediction_stanza(clean_text)
    hf_spans = prediction_hf_raw(clean_text)

    final_spans = []

    # 1) header
    final_spans.extend(header_spans)

    # 2) Stanza
    final_spans.extend(stanza_spans)

    # 3) HF che si sovrappongono a header o stanza -> accettate
    for s, e, l in hf_spans:
        span = [s, e, l]
        if overlaps_with_any(span, header_spans) or overlaps_with_any(span, stanza_spans):
            final_spans.append(span)

    # 4) HF-only -> NLI
    for s, e, l in hf_spans:
        span = [s, e, l]
        if overlaps_with_any(span, header_spans) or overlaps_with_any(span, stanza_spans):
            continue  # già gestito

        span_text = clean_text[s:e]
        if is_person_not_drug(span_text, clean_text):
            final_spans.append(span)

    return dedup_and_sort(final_spans)


# ============================================================
# VALUTAZIONE RISPETTO AL GOLD
# ============================================================

def evaluate(gold_path: str, pred_path: str):
    """
    Calcola metriche span-level (PER) e doc-level:
    - TP, FP, FN a livello di span (overlap-based)
    - TP_doc, FP_doc, FN_doc, TN_doc a livello di documento
    """
    # carica gold e predizioni in dict {id: rec}
    gold = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            gold[rec["id"]] = rec

    pred = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pred[rec["id"]] = rec

    # --- span-level ---
    TP = FP = FN = 0

    for rid, grec in gold.items():
        g_spans = [(int(s), int(e)) for (s, e, _) in grec.get("entities", [])]
        p_spans = [(int(s), int(e)) for (s, e, _) in pred.get(rid, {}).get("entities", [])]

        # match based on overlap > 0
        matched_pred = set()

        for (gs, ge) in g_spans:
            found = False
            for i, (ps, pe) in enumerate(p_spans):
                if not (pe <= gs or ge <= ps):  # overlap
                    found = True
                    matched_pred.add(i)
                    break
            if found:
                TP += 1
            else:
                FN += 1

        # FP = pred non matchati con nessun gold
        FP += len(p_spans) - len(matched_pred)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- doc-level ---
    TP_doc = FP_doc = FN_doc = TN_doc = 0

    for rid, grec in gold.items():
        g_has = len(grec.get("entities", [])) > 0
        p_has = len(pred.get(rid, {}).get("entities", [])) > 0

        if g_has and p_has:
            TP_doc += 1
        elif g_has and not p_has:
            FN_doc += 1
        elif not g_has and p_has:
            FP_doc += 1
        else:
            TN_doc += 1

    recall_doc = TP_doc / (TP_doc + FN_doc) if (TP_doc + FN_doc) > 0 else 0.0
    precision_doc = TP_doc / (TP_doc + FP_doc) if (TP_doc + FP_doc) > 0 else 0.0
    f1_doc = 2 * precision_doc * recall_doc / (precision_doc + recall_doc) if (precision_doc + recall_doc) > 0 else 0.0

    print("=== SPAN-LEVEL (PER) ===")
    print(f"TP = {TP}, FP = {FP}, FN = {FN}")
    print(f"Precision = {precision:.3f}")
    print(f"Recall    = {recall:.3f}")
    print(f"F1        = {f1:.3f}")
    print()
    print("=== DOC-LEVEL (ANY PER) ===")
    print(f"TP_doc = {TP_doc}, FP_doc = {FP_doc}, FN_doc = {FN_doc}, TN_doc = {TN_doc}")
    print(f"Precision_doc = {precision_doc:.3f}")
    print(f"Recall_doc    = {recall_doc:.3f}")
    print(f"F1_doc        = {f1_doc:.3f}")

    return {
        "span": {"TP": TP, "FP": FP, "FN": FN,
                 "precision": precision, "recall": recall, "f1": f1},
        "doc":  {"TP": TP_doc, "FP": FP_doc, "FN": FN_doc, "TN": TN_doc,
                 "precision": precision_doc, "recall": recall_doc, "f1": f1_doc},
    }


# ============================================================
# MAIN
# ============================================================

def main():
    # 1) genera predizioni ensemble
    with open(SRC_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            rec  = json.loads(line)
            text = rec["text"]
            rid  = rec["id"]

            pred_entities = ensemble_predict(text)

            out = {
                "id": rid,
                "text": text,
                "entities": pred_entities
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[ENSEMBLE] Predizioni scritte in: {OUTPUT_JSONL}")

    # 2) valutazione rispetto al gold
    evaluate(SRC_JSONL, OUTPUT_JSONL)


if __name__ == "__main__":
    main()
