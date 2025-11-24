from functools import lru_cache
import stanza
from transformers import(
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    )
import regex as re
from decoratori import singleton
import utils as u


def _overlaps(span_a, span_b) -> bool:
    sa, ea, _ = span_a
    sb, eb, _ = span_b
    return not (ea <= sb or eb <= sa)


def _dedup_and_sort(spans):
    uniq = {(int(s), int(e), str(l)) for (s, e, l) in spans}
    return [(s, e, l) for (s, e, l) in sorted(uniq)]


def _filter_overlaps_keep_widest(spans):
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    groups = []
    current = [spans[0]]

    for sp in spans[1:]:
        if any(_overlaps(sp, g) for g in current):
            current.append(sp)
        else:
            groups.append(current)
            current = [sp]

    groups.append(current)

    final = []
    for g in groups:
        best = max(g, key=lambda x: (x[1] - x[0], -x[0]))
        final.append(best)

    return sorted(final, key=lambda x: x[0])



@singleton
class Anonymizer:

    _HF_MODEL_PATH = u.load_config()["hf_model_NER_path_r"]
    _CF_REGEX = re.compile(r"[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]")
    _ANAGRAFIC_REGEX = re.compile(
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
    _ANAGRAFIC_CONTEXT_BEFORE = re.compile(
        r'(paziente|sig\.?r?|sig\.?ra|sig\.?na|signor\w*)',
        flags=re.IGNORECASE
    )
    _ANAGRAFIC_CONTEXT_AFTER = re.compile(
        r'(nato|nata|data\s+nascita|sesso)',
        flags=re.IGNORECASE
    )
    _CF_REGEX = re.compile(
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

    def __init__(self, use_gpu: bool = True, device: int = 0):
        self._nlp = stanza.Pipeline(
            lang='it',
            processors='tokenize,ner',
            use_gpu=use_gpu
        )
        self._clf = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=device,
        )
        self._tokenizer_ner = AutoTokenizer.from_pretrained(self._HF_MODEL_PATH)
        self._model_ner = AutoModelForTokenClassification.from_pretrained(self._HF_MODEL_PATH)
        self._ner_pipeline = pipeline(
            "token-classification",
            model=self._model_ner,
            tokenizer=self._tokenizer_ner,
            aggregation_strategy="simple",   # unisce B-/I- token in un'unica entità
            device=device,
        )
    def _is_person_span(self, span_text: str, context: str) -> bool:
        key = (span_text, context)
        if key in self._person_cache:
            return self._person_cache[key]

        res = self._clf(
            sequences=span_text,
            candidate_labels=["persona", "non-persona"],
            hypothesis_template="Questo testo si riferisce a {}.",
        )
        is_person = res["labels"][0] == "persona"
        self._person_cache[key] = is_person
        return is_person
    
    def _extract_header_spans(self, text: str):
        spans = []

        # blocco anagrafico “Paziente: NOME”
        for m in self._ANAGRAFIC_REGEX.finditer(text):
            start = m.start("names")
            end = m.end("names")
            spans.append((start, end, "PER"))

        # nomi tutti MAIUSCOLI in contesto anagrafico
        upper_name = re.compile(
            r"\b([A-ZÀ-Ý][A-ZÀ-Ý]+(?:\s+[A-ZÀ-Ý][A-ZÀ-Ý]+)+)\b"
        )

        for m in upper_name.finditer(text):
            start, end = m.span(1)
            before = text[max(0, start - 60):start]
            after = text[end:end + 60]
            if (self._ANAGRAFIC_CONTEXT_BEFORE.search(before) or
                    self._ANAGRAFIC_CONTEXT_AFTER.search(after)):
                spans.append((start, end, "PER"))

        return _dedup_and_sort(spans)
    
    def _collect_person_spans_ensemble(self, text: str):
        clean_text = text

        # ------------------------
        # 1) Header anagrafico
        # ------------------------
        header_spans = self._extract_header_spans(clean_text)

        # ------------------------
        # 2) Stanza NER
        # ------------------------
        stanza_spans = []
        doc = self._nlp(clean_text)
        for ent in doc.entities:
            if ent.type == "PER":
                stanza_spans.append((ent.start_char, ent.end_char, "PER"))

        stanza_spans = _dedup_and_sort(stanza_spans)

        # ------------------------
        # 3) HuggingFace NER
        # ------------------------
        hf_spans = []
        hf_out = self._ner_pipeline(clean_text)
        for ent in hf_out:
            if "PER" in ent.get("entity_group", ""):
                hf_spans.append((ent["start"], ent["end"], "PER"))

        hf_spans = _dedup_and_sort(hf_spans)

        # ------------------------
        # COMBINAZIONE
        # ------------------------
        final = []
        final.extend(header_spans)
        final.extend(stanza_spans)

        def ov(span, pool):
            return any(_overlaps(span, p) for p in pool)

        # 3A) HF che si sovrappongono a header/stanza → tengono senza NLI
        for span in hf_spans:
            if ov(span, header_spans) or ov(span, stanza_spans):
                final.append(span)

        # 3B) HF-only → passano NLI
        for span in hf_spans:
            if ov(span, header_spans) or ov(span, stanza_spans):
                continue
            s, e, _ = span
            text_span = clean_text[s:e]
            if self._is_person_span(text_span, clean_text):
                final.append(span)

        # dedup + risoluzione overlap
        final = _dedup_and_sort(final)
        final = _filter_overlaps_keep_widest(final)

        return final
    
    def detect_entities(self, text: str):
        persons = self._collect_person_spans_ensemble(text)
        entities = [{"start": s, "end": e, "label": l} for (s, e, l) in persons]

        # opzionale: CF come entità separata
        for m in self._CF_REGEX.finditer(text):
            entities.append({"start": m.start(), "end": m.end(), "label": "CF"})

        # ultima pulizia overlap tra tutte le entità
        spans_tuple = [(e["start"], e["end"], e["label"]) for e in entities]
        spans_tuple = _filter_overlaps_keep_widest(spans_tuple)

        final = [
            {"start": s, "end": e, "label": l}
            for (s, e, l) in spans_tuple
        ]
        return final
    
    def anonymize_text(self, text: str, placeholder: str = "<REDACTED>"):
        ents = self.detect_entities(text)
        ents = sorted(ents, key=lambda x: x["start"], reverse=True)

        out = text
        for ent in ents:
            s, e, l = ent["start"], ent["end"], ent["label"]
            if l in ("PER", "CF"):
                out = out[:s] + placeholder + out[e:]

        return out
    