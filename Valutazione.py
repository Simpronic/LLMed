import json
import pandas as pd

# ============================================================
# CONFIGURAZIONE
# ============================================================

GOLD_JSONL = "gold_patients_only.jsonl"      # gold standard
PRED_JSONL = "predictions_system_ensemble.jsonl"      # cambia qui per ogni modello

ERRORS_XLSX = "righe_con_errori_spans.xlsx"  # opzionale


# ============================================================
# FUNZIONI DI UTILITÀ
# ============================================================

def load_jsonl(path):
    """
    Carica un JSONL di record tipo:
    {"id": "...", "text": "...", "entities": [[start, end, "PER"], ...]}
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def normalize_entities(entities, text):
    """
    Normalizza una lista di entità [start, end] o [start, end, label]
    fondendo gli span adiacenti con la stessa label se tra i due
    c'è solo whitespace nel testo originale.

    Restituisce sempre una lista di [start, end, label].
    """
    if not entities:
        return []

    # normalizza formato e ordina per start
    ents = []
    for ent in entities:
        if not isinstance(ent, (list, tuple)) or len(ent) < 2:
            continue
        if len(ent) == 2:
            s, e = ent
            label = "PER"
        else:
            s, e, label = ent[0], ent[1], ent[2]
        ents.append([int(s), int(e), str(label)])

    ents.sort(key=lambda x: x[0])

    if not ents:
        return []

    merged = [ents[0]]
    for s, e, label in ents[1:]:
        last_s, last_e, last_label = merged[-1]

        # se tra la fine del precedente e l'inizio del corrente
        # ci sono solo spazi/bianchi, e la label è la stessa,
        # fondi gli span
        between = text[last_e:s]
        if label == last_label and between.strip() == "":
            merged[-1][1] = e
        else:
            merged.append([s, e, label])

    return merged


def entities_to_set(entities):
    """
    Converte [[start, end, label], ...]
    in un SET di tuple (start, end, label) per fare confronti esatti di span.
    """
    out = set()
    for ent in entities or []:
        if not isinstance(ent, (list, tuple)) or len(ent) < 2:
            continue
        # supporta sia [s,e] che [s,e,label]
        if len(ent) == 2:
            s, e = ent
            label = "PER"
        else:
            s, e, label = ent[0], ent[1], ent[2]
        out.add((int(s), int(e), str(label)))
    return out


# ============================================================
# METRICHE (ENTITÀ-SPAN LEVEL E DOC-LEVEL)
# ============================================================

def compute_entity_level_metrics(df, gold_col="gold_set", pred_col="pred_set"):
    """
    TP/FP/FN a livello di ENTITÀ-SPAN (micro),
    usando SET di tuple (start, end, label).
    """
    TP = FP = FN = 0

    for _, row in df.iterrows():
        true_set = row[gold_col]
        pred_set = row[pred_col]

        TP += len(true_set & pred_set)
        FP += len(pred_set - true_set)
        FN += len(true_set - pred_set)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc_pos   = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_pos_only": acc_pos,
    }


def compute_doc_level_confusion(df, gold_col="gold_set", pred_col="pred_set"):
    """
    Confusion matrix a livello di REFERTI (document-level),
    usando presenza/assenza di almeno una entità.
    """
    TP_doc = FP_doc = FN_doc = TN_doc = 0

    for _, row in df.iterrows():
        true_set = row[gold_col]
        pred_set = row[pred_col]

        has_true = len(true_set) > 0
        has_pred = len(pred_set) > 0

        if has_true and has_pred:
            TP_doc += 1
        elif has_true and not has_pred:
            FN_doc += 1
        elif not has_true and has_pred:
            FP_doc += 1
        else:
            TN_doc += 1

    total = TP_doc + FP_doc + FN_doc + TN_doc

    accuracy   = (TP_doc + TN_doc) / total if total > 0 else 0.0
    precision  = TP_doc / (TP_doc + FP_doc) if (TP_doc + FP_doc) > 0 else 0.0
    recall     = TP_doc / (TP_doc + FN_doc) if (TP_doc + FN_doc) > 0 else 0.0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = TN_doc / (TN_doc + FP_doc) if (TN_doc + FP_doc) > 0 else 0.0

    return {
        "TP_doc": TP_doc,
        "FP_doc": FP_doc,
        "FN_doc": FN_doc,
        "TN_doc": TN_doc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "n_docs": total,
    }


def compute_per_doc_macro(df, gold_col="gold_set", pred_col="pred_set"):
    """
    Metriche macro (media per referto) SOLO sui casi con almeno una entità gold,
    usando i set di entità-spans.
    """
    per_doc = []

    for _, row in df.iterrows():
        true_set = row[gold_col]
        pred_set = row[pred_col]

        if len(true_set) == 0:
            continue

        TP = len(true_set & pred_set)
        FP = len(pred_set - true_set)
        FN = len(true_set - pred_set)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_doc.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_true": len(true_set),
            "n_pred": len(pred_set),
        })

    if not per_doc:
        return None

    per_doc_df = pd.DataFrame(per_doc)
    return {
        "macro_precision": per_doc_df["precision"].mean(),
        "macro_recall": per_doc_df["recall"].mean(),
        "macro_f1": per_doc_df["f1"].mean(),
        "n_docs_eval": len(per_doc_df),
    }


def compute_row_errors(df, gold_col="gold_set", pred_col="pred_set"):
    """
    DataFrame con una riga per referto, contenente:
    - TP/FP/FN a livello ENTITÀ-SPAN per quel referto
    - flag corretto / tipo_errore
    """
    rows = []

    for idx, row in df.iterrows():
        true_set = row[gold_col]
        pred_set = row[pred_col]

        TP = len(true_set & pred_set)
        FP = len(pred_set - true_set)
        FN = len(true_set - pred_set)

        if len(true_set) == 0 and len(pred_set) == 0:
            corretto = True
            tipo = "corretto_no_entita"
        elif FP == 0 and FN == 0:
            corretto = True
            tipo = "corretto_entita_ok"
        elif len(true_set) == 0 and len(pred_set) > 0:
            corretto = False
            tipo = "FP_puro"
        elif len(true_set) > 0 and len(pred_set) == 0:
            corretto = False
            tipo = "FN_puro"
        else:
            corretto = False
            tipo = "misto_FP_FN"

        rows.append({
            "row_index": idx,
            "TP_row": TP,
            "FP_row": FP,
            "FN_row": FN,
            "n_true_row": len(true_set),
            "n_pred_row": len(pred_set),
            "corretto": corretto,
            "tipo_errore": tipo,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    # 1) Carica gold e pred
    gold_records = load_jsonl(GOLD_JSONL)
    pred_records = load_jsonl(PRED_JSONL)

    # indicizza pred per id
    pred_by_id = {r["id"]: r for r in pred_records}

    rows = []
    missing_pred_ids = []

    for g in gold_records:
        gid = g["id"]
        text = g["text"]
        gold_entities = g.get("entities", [])

        p = pred_by_id.get(gid)
        if p is None:
            pred_entities = []
            missing_pred_ids.append(gid)
        else:
            pred_entities = p.get("entities", [])

        rows.append({
            "id": gid,
            "text": text,
            "gold_entities": gold_entities,
            "pred_entities": pred_entities,
        })

    df = pd.DataFrame(rows)

    if missing_pred_ids:
        print("Attenzione: alcuni id gold non sono presenti nel file di predizione.")
        print(f"Numero di id mancanti: {len(missing_pred_ids)}")
        print(f"Esempi: {missing_pred_ids[:10]}")

    # 2) normalizza le entità (merge di PER adiacenti separati solo da spazi)
    df["gold_entities_norm"] = df.apply(
        lambda r: normalize_entities(r["gold_entities"], r["text"]),
        axis=1
    )
    df["pred_entities_norm"] = df.apply(
        lambda r: normalize_entities(r["pred_entities"], r["text"]),
        axis=1
    )

    # 3) converte in set di entità-spans usando le entità normalizzate
    df["gold_set"] = df["gold_entities_norm"].apply(entities_to_set)
    df["pred_set"] = df["pred_entities_norm"].apply(entities_to_set)

    # 4) metriche a livello di entità-span
    ent_metrics = compute_entity_level_metrics(df)

    print("=== METRICHE A LIVELLO DI ENTITÀ (span-level, micro) ===")
    print(f"TP: {ent_metrics['TP']}")
    print(f"FP: {ent_metrics['FP']}")
    print(f"FN: {ent_metrics['FN']}")
    print(f"Precision: {ent_metrics['precision']:.4f}")
    print(f"Recall:    {ent_metrics['recall']:.4f}")
    print(f"F1-score:  {ent_metrics['f1']:.4f}")
    print(f"'Accuracy' sui positivi (TP / (TP+FP+FN)): {ent_metrics['accuracy_pos_only']:.4f}")

    # 5) confusion matrix doc-level
    doc_conf = compute_doc_level_confusion(df)

    print("\n=== CONFUSION MATRIX A LIVELLO DI REFERTI (doc-level) ===")
    print(f"TP_doc: {doc_conf['TP_doc']}")
    print(f"FN_doc: {doc_conf['FN_doc']}")
    print(f"FP_doc: {doc_conf['FP_doc']}")
    print(f"TN_doc: {doc_conf['TN_doc']}")
    print(f"Numero totale di referti: {doc_conf['n_docs']}")

    print("\n=== METRICHE DOC-LEVEL (classificazione referti) ===")
    print(f"Accuracy:    {doc_conf['accuracy']:.4f}")
    print(f"Precision:   {doc_conf['precision']:.4f}")
    print(f"Recall:      {doc_conf['recall']:.4f}")
    print(f"F1-score:    {doc_conf['f1']:.4f}")
    print(f"Specificità: {doc_conf['specificity']:.4f}")

    # 6) macro-metriche solo referti con entità gold
    per_doc_metrics = compute_per_doc_macro(df)
    if per_doc_metrics is not None:
        print("\n=== METRICHE PER DOCUMENTO (macro, solo referti con entità gold) ===")
        print(f"Macro Precision: {per_doc_metrics['macro_precision']:.4f}")
        print(f"Macro Recall:    {per_doc_metrics['macro_recall']:.4f}")
        print(f"Macro F1-score:  {per_doc_metrics['macro_f1']:.4f}")
        print(f"Numero di referti valutati: {per_doc_metrics['n_docs_eval']}")

    # 7) righe con errori
    row_errors = compute_row_errors(df)
    wrong_rows = row_errors[~row_errors["corretto"]]

    print("\n=== RIGHE CON ERRORI (span-level) ===")
    print(f"Numero di referti con almeno un errore: {len(wrong_rows)}")

    if len(wrong_rows) > 0:
        merged = wrong_rows.merge(
            df[["id", "text",
                "gold_entities", "pred_entities",
                "gold_entities_norm", "pred_entities_norm"]],
            left_on="row_index",
            right_index=True
        )

        N = 20
        print(f"\nPrime {min(N, len(merged))} righe con errore:")
        cols_to_show = [
            "row_index", "id", "tipo_errore",
            "TP_row", "FP_row", "FN_row",
            "gold_entities_norm", "pred_entities_norm"
        ]
        print(merged[cols_to_show].head(N).to_string(index=False))

        merged.to_excel(ERRORS_XLSX, index=False)
        print(f"\nTutte le righe con errore sono state salvate in: {ERRORS_XLSX}")
    else:
        print("Non sono stati trovati errori a livello di referto (span-level).")


if __name__ == "__main__":
    main()
