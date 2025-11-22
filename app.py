# app.py
"""
Balanced Spoken Introduction Evaluator — PATCHED (NaN fix + name/age detection + audit view)

Drop-in replacement for your existing app.py. Copy-paste and run:
    streamlit run app.py

Features:
- Robust rubric parsing (handles header-in-middle)
- Normalizes NaN cells so 'nan' strings don't leak into logic/UI
- detect_name() and detect_age() helpers to recognize spoken name/age phrases
- Balanced scoring: keywords (α=0.45), semantic (β=0.45), extras (γ=0.10)
- Keeps the 5 main criteria view, plus an optional full-rubric audit
- Safe fallbacks if language_tool / vader missing
"""

import re
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Optional libraries
try:
    import language_tool_python
except Exception:
    language_tool_python = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

# ---------- CONFIG & PATHS ----------
DEFAULT_RUBRIC = "Case study for interns.xlsx"    # default path (use upload in sidebar to override)
DEFAULT_SAMPLE = "Sample text for case study.txt"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Balanced blending defaults
DEFAULT_ALPHA = 0.45
DEFAULT_BETA = 0.45
DEFAULT_GAMMA = 0.10

MAJOR_CRITERIA = [
    "Content & Structure",
    "Speech Rate",
    "Language & Grammar",
    "Clarity",
    "Engagement",
]

# filler detection
FILLER_SET = {
    "um", "uh", "like", "actually", "basically", "right", "okay", "ok", "hmm", "ah", "well", "kinda"
}
FILLER_PHRASES = ["you know", "sort of", "kind of", "i mean"]


# ---------- TEXT UTILITIES ----------
def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def tokens(text: str):
    return re.findall(r"\b\w+\b", (text or "").lower())


def word_count(text: str) -> int:
    return len(tokens(text))


def type_token_ratio(text: str) -> float:
    t = tokens(text)
    if not t:
        return 0.0
    return len(set(t)) / len(t)


# ---------- Small detectors to satisfy mandatory fields ----------
def detect_name(text: str):
    """Detect common spoken-name patterns: 'my name is X', 'I am X', 'I'm X', 'myself X'."""
    if not text:
        return None
    # Pattern with capitalized name(s)
    m = re.search(r"\b(?:my name is|i am|i'm|myself)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text, flags=re.I)
    if m:
        return m.group(1).strip()
    # fallback: lowercase/regular token after 'myself' or 'i am'
    m2 = re.search(r"\b(?:myself|i am|i'm)\s+([a-z][a-z]+(?:\s[a-z]+)?)\b", text, flags=re.I)
    if m2:
        name = m2.group(1).strip()
        if not name.isdigit() and len(name) > 1:
            return name
    return None


def detect_age(text: str):
    """Detect age phrases like 'I am 13', '13 years old'."""
    if not text:
        return None
    m = re.search(r"\b(\d{1,2})\s*(?:years old|yrs old|years)\b", text, flags=re.I)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    m2 = re.search(r"\bi am (\d{1,2})\b", text, flags=re.I)
    if m2:
        try:
            return int(m2.group(1))
        except:
            return None
    return None


# ---------- GRAMMAR, SENTIMENT, FILLER ----------
def grammar_score_and_count(text: str):
    if language_tool_python is None:
        return 0.95, 0
    try:
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(text)
        errors = len(matches)
        wc = max(1, word_count(text))
        errors_per_100 = errors / wc * 100
        score = 1.0 - min(errors_per_100 / 10.0, 1.0)
        return max(0.0, score), errors
    except Exception:
        return 0.95, 0


def sentiment_positive(text: str):
    if SentimentIntensityAnalyzer is None:
        return 0.5
    try:
        an = SentimentIntensityAnalyzer()
        vs = an.polarity_scores(text)
        return float(vs.get("pos", 0.0))
    except Exception:
        return 0.5


def filler_score_and_count(text: str):
    wc = word_count(text)
    if wc == 0:
        return 1.0, 0
    tl = text.lower()
    cnt = 0
    for f in FILLER_SET:
        cnt += len(re.findall(r"\b" + re.escape(f) + r"\b", tl))
    for ph in FILLER_PHRASES:
        cnt += tl.count(ph)
    rate = cnt / wc
    if rate <= 0.03:
        s = 1.0
    elif rate <= 0.06:
        s = 0.85
    elif rate <= 0.09:
        s = 0.6
    elif rate <= 0.12:
        s = 0.35
    else:
        s = 0.15
    return s, cnt


# ---------- SEMANTIC & KEYWORD ----------
def semantic_similarity(model, text: str, description: str):
    if not description or not description.strip():
        return 0.0, 0.5
    try:
        emb_t = model.encode(text, convert_to_tensor=True)
        emb_d = model.encode(description, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_t, emb_d)[0][0])
        sim_norm = (sim + 1.0) / 2.0
        return sim, sim_norm
    except Exception:
        return 0.0, 0.5


def keyword_coverage(text: str, keywords: list, mandatory: list = None):
    txt = " " + (text or "").lower() + " "
    if (not keywords or all(not k.strip() for k in keywords)) and (not mandatory or all(not m.strip() for m in (mandatory or []))):
        return 1.0, [], []
    matched = 0
    missing_opt = []
    missing_mand = []
    opt_total = max(1, len([k for k in (keywords or []) if k and k.strip()]))
    mand_total = max(1, len([m for m in (mandatory or []) if m and m.strip()])) if mandatory else 0

    if keywords:
        for kw in keywords:
            kw = kw.strip().lower()
            if kw == "":
                continue
            if re.search(r"\b" + re.escape(kw) + r"\b", txt):
                matched += 1
            else:
                missing_opt.append(kw)
        opt_score = matched / opt_total
    else:
        opt_score = 1.0

    mand_missing_count = 0
    if mandatory:
        for kw in mandatory:
            kw = kw.strip().lower()
            if kw == "":
                continue
            if not re.search(r"\b" + re.escape(kw) + r"\b", txt):
                mand_missing_count += 1
                missing_mand.append(kw)
        if mand_total > 0:
            mand_score = max(0.0, 1.0 - (mand_missing_count / mand_total) * 0.6)
        else:
            mand_score = 1.0
    else:
        mand_score = 1.0

    if mandatory:
        cov = 0.6 * mand_score + 0.4 * opt_score
    else:
        cov = opt_score

    return cov, missing_mand, missing_opt


# ---------- RUBRIC PARSING ----------
def parse_overall_rubrics(path: str):
    """
    Read Excel and find the 'Overall Rubrics' block with header 'Creteria', 'Metric', 'Weightage'.
    Returns tidy DataFrame of the rubric rows (keeps MAJOR_CRITERIA rows by default).
    """
    raw = pd.read_excel(path, engine="openpyxl", header=None)
    header_idx = None
    header_cols = None
    for i in range(len(raw)):
        vals = [str(v).strip() for v in raw.iloc[i].tolist()]
        if "Creteria" in vals and "Metric" in vals and "Weightage" in vals:
            header_idx = i
            header_cols = vals
            break
    if header_idx is None:
        # fallback: try reading with header=0 and check columns
        df_try = pd.read_excel(path, engine="openpyxl")
        cols_low = [c.strip().lower() for c in df_try.columns.astype(str).tolist()]
        if "creteria" in cols_low and "metric" in cols_low and "weightage" in cols_low:
            df2 = df_try[["Creteria", "Metric", "Weightage"]].copy()
            df2.columns = ["Creteria", "Metric", "Weightage"]
            # normalize blanks
            for col in df2.columns:
                if df2[col].dtype == object:
                    df2[col] = df2[col].fillna("").astype(str)
                else:
                    df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0.0)
            return df2
        raise ValueError("Could not find Overall Rubrics header row. Make sure Excel is correct.")

    data = raw.iloc[header_idx + 1:].copy()
    data.columns = header_cols
    data = data[["Creteria", "Metric", "Weightage"]]

    # normalize empty cells: text -> "" ; numeric -> 0.0
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].fillna("").astype(str)
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    # Stop if another header repeats
    stop_mask = (data["Creteria"] == "Creteria") & (data["Metric"] == "Metric")
    if stop_mask.any():
        stop = stop_mask[stop_mask].index[0]
        data = data.loc[:stop - 1]

    data = data.dropna(how="all")
    data = data.rename(columns={"Creteria": "criterion", "Metric": "description", "Weightage": "weight"})
    # ensure strings clean
    data["criterion"] = data["criterion"].astype(str).str.strip()
    data["description"] = data["description"].astype(str).str.strip()
    data["weight"] = pd.to_numeric(data["weight"], errors="coerce").fillna(0.0)

    # Keep only major criteria (this matches the Overall Rubrics expectation)
    # If you prefer to score every row, change this to return data (without filtering).
    data = data[data["criterion"].isin(MAJOR_CRITERIA)].copy()

    # If multiple identical criterion names appear, keep first (top-level aggregation)
    data = data.groupby("criterion", as_index=False).first()

    # human-friendly descriptions override
    desc_map = {
        "Content & Structure": "Clear salutation, name, class/school, family, hobbies, goals and closing, in logical order.",
        "Speech Rate": "Appropriate speaking pace (words per minute).",
        "Language & Grammar": "Grammar correctness and vocabulary richness.",
        "Clarity": "Few filler words; clear phrasing.",
        "Engagement": "Positive, enthusiastic, confident tone.",
    }
    data["description"] = data["criterion"].map(lambda c: desc_map.get(c, data.loc[data['criterion'] == c, 'description'].values[0] if c in data['criterion'].values else ""))

    # normalize weights to sum to 1
    weights = data["weight"].astype(float).tolist()
    s = sum(weights) if sum(weights) > 0 else 1.0
    data["weight_norm"] = [w / s for w in weights]

    return data.reset_index(drop=True)


# ---------- EVALUATOR ----------
class SpokenIntroEvaluator:
    def __init__(self, rubric_path: str, model_name: str = DEFAULT_MODEL):
        self.rubric_df = parse_overall_rubrics(rubric_path)

        # mandatory/optional maps (per major criterion)
        self.mandatory = {
            "Content & Structure": ["name", "age", "class", "school"],
        }
        self.optional = {
            "Content & Structure": ["family", "hobbies", "interests", "goals", "fun fact", "unique"],
            "Language & Grammar": ["grammar", "vocabulary", "sentence", "words"],
            "Engagement": ["excited", "enthusiastic", "confident", "happy", "grateful", "interested"],
        }

        # store model name and load model
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def score(self, transcript: str, duration_seconds: float = None,
              alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, gamma=DEFAULT_GAMMA):
        txt = preprocess_text(transcript or "")
        wc = word_count(txt)

        # global signals
        gram_score, gram_errs = grammar_score_and_count(txt)
        sent_pos = sentiment_positive(txt)
        ttr = type_token_ratio(txt)
        filler_s, filler_cnt = filler_score_and_count(txt)

        # wpm
        wpm = None
        if duration_seconds and duration_seconds > 0:
            wpm = wc / (duration_seconds / 60.0)

        # detect name/age once per transcript
        detected_name = detect_name(transcript or "")
        detected_age = detect_age(transcript or "")

        results = []
        total_weighted = 0.0
        total_weights = 0.0

        for _, row in self.rubric_df.iterrows():
            crit = row["criterion"]
            desc = row.get("description", "") or ""
            weight_norm = float(row.get("weight_norm", 0.0))

            # prepare mandatory & optional lists for this criterion
            mand = list(self.mandatory.get(crit, []))
            opt = list(self.optional.get(crit, []))

            # For Content & Structure: satisfy name/age via detectors
            if crit == "Content & Structure":
                if detected_name:
                    mand = [m for m in mand if m.strip().lower() != "name"]
                if detected_age:
                    mand = [m for m in mand if m.strip().lower() != "age"]

            # compute keyword coverage using these lists
            kw_cov, missing_mand, missing_opt = keyword_coverage(txt, opt, mand)

            # semantic
            sem_raw, sem_norm = semantic_similarity(self.model, txt, desc)

            # extras per criterion
            extra = 1.0
            notes = []

            if crit == "Language & Grammar":
                vocab = ttr
                extra = 0.6 * gram_score + 0.4 * vocab
                notes.append(f"gram_errs={gram_errs}, gram_score={gram_score:.2f}, TTR={vocab:.2f}")
            elif crit == "Clarity":
                extra = filler_s
                notes.append(f"fillers={filler_cnt}")
            elif crit == "Engagement":
                extra = sent_pos
                notes.append(f"pos_sent={sent_pos:.2f}")
            elif crit == "Speech Rate":
                if wpm is not None:
                    if wpm > 161:
                        extra = 0.2
                    elif 141 <= wpm <= 160:
                        extra = 0.6
                    elif 111 <= wpm <= 140:
                        extra = 1.0
                    elif 81 <= wpm <= 110:
                        extra = 0.6
                    else:
                        extra = 0.2
                    notes.append(f"wpm={wpm:.1f}")
                else:
                    extra = 1.0 if wc >= 90 else 0.9
                    notes.append("wpm=N/A (no duration)")

            # combine signals
            crit_raw = alpha * kw_cov + beta * sem_norm + gamma * extra
            crit_raw = float(np.clip(crit_raw, 0.0, 1.0))
            crit_score = round(crit_raw * 100.0, 1)

            # feedback assembly
            fb_parts = []
            if missing_mand:
                fb_parts.append("Missing mandatory details: " + ", ".join(missing_mand) + ".")
            if missing_opt:
                fb_parts.append("Consider adding: " + ", ".join(missing_opt) + ".")
            if sem_norm < 0.5:
                fb_parts.append("Content could be more aligned with rubric description.")
            if crit in ["Language & Grammar", "Clarity"] and notes:
                fb_parts.append("; ".join(notes))
            if not fb_parts:
                fb_parts = ["Good job on this criterion."]

            results.append({
                "criterion": crit,
                "description": desc,
                "score": crit_score,
                "keyword_coverage": round(kw_cov, 2),
                "semantic_similarity": round(sem_raw, 3),
                "extra_value": round(extra, 3),
                "missing_mandatory": missing_mand,
                "missing_optional": missing_opt,
                "notes": notes,
                "feedback": " ".join(fb_parts)
            })

            total_weighted += crit_score * weight_norm
            total_weights += weight_norm

        overall = (total_weighted / total_weights) if total_weights > 0 else 0.0

        diagnostics = {
            "word_count": wc,
            "grammar_errors": gram_errs,
            "grammar_score": round(gram_score, 3),
            "ttr": round(ttr, 3),
            "filler_count": filler_cnt,
            "sentiment_pos": round(sent_pos, 3),
            "wpm": round(wpm, 1) if wpm else None,
            "detected_name": detected_name,
            "detected_age": detected_age
        }

        return wc, round(overall, 1), results, diagnostics


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Spoken Intro Evaluator — Balanced (Patched)", layout="centered")
st.title("🎓 Spoken Introduction Evaluator — Balanced (Patched)")

st.write("Paste transcript or upload a text file. Use sliders to tune blending (α/β/γ). Defaults are balanced for submission.")

# Sidebar file upload overrides default paths
rubric_file = st.sidebar.file_uploader("Upload rubric Excel (optional)", type=["xlsx"])
if rubric_file:
    rubric_path = rubric_file  # pandas can accept file-like
else:
    rubric_path = DEFAULT_RUBRIC

sample_file = st.sidebar.file_uploader("Upload sample transcript (optional)", type=["txt"])
if sample_file:
    sample_text = sample_file.read().decode("utf-8")
else:
    try:
        with open(DEFAULT_SAMPLE, "r", encoding="utf-8") as f:
            sample_text = f.read()
    except Exception:
        sample_text = ""

# model choice
model_choice = st.sidebar.selectbox("Embedding model", ["all-MiniLM-L6-v2 (fast)", "all-mpnet-base-v2 (more accurate)"])
model_name = "sentence-transformers/all-mpnet-base-v2" if "mpnet" in model_choice else "sentence-transformers/all-MiniLM-L6-v2"

# blending sliders
st.sidebar.header("Blending (inside each criterion)")
alpha = st.sidebar.slider("Keyword weight (α)", 0.0, 1.0, float(DEFAULT_ALPHA), 0.01)
beta = st.sidebar.slider("Semantic weight (β)", 0.0, 1.0, float(DEFAULT_BETA), 0.01)
gamma = st.sidebar.slider("Extras weight (γ)", 0.0, 1.0, float(DEFAULT_GAMMA), 0.01)
s = alpha + beta + gamma
if s == 0:
    alpha, beta, gamma = DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA
else:
    alpha, beta, gamma = alpha / s, beta / s, gamma / s

# instantiate evaluator safely
try:
    evaluator = SpokenIntroEvaluator(rubric_path, model_name)
except Exception as e:
    st.error(f"Could not load rubric/model: {e}")
    st.stop()

with st.expander("Parsed Overall Rubrics (cleaned)"):
    st.dataframe(evaluator.rubric_df.fillna(""), use_container_width=True)

# show if any row had blanks originally (debug)
with st.expander("Rubric debug (rows with missing cells)"):
    missing_mask = evaluator.rubric_df.isna().any(axis=1)
    if missing_mask.any():
        st.dataframe(evaluator.rubric_df[missing_mask].fillna(""), use_container_width=True)
    else:
        st.write("No missing cells detected.")

# input area
txt = st.text_area("Transcript", value=sample_text, height=260)
duration_sec = st.number_input("Optional: Duration (seconds) for WPM", min_value=0.0, value=0.0, step=1.0)
duration_val = None if duration_sec <= 0 else float(duration_sec)

if st.button("Evaluate"):
    with st.spinner("Scoring..."):
        # reload model if selection changed
        if model_name != evaluator.model_name:
            evaluator = SpokenIntroEvaluator(rubric_path, model_name)

        wc, overall, per_crit, diag = evaluator.score(txt, duration_val, alpha=alpha, beta=beta, gamma=gamma)

    st.success("Done")
    st.metric("Overall score", f"{overall:.1f} / 100")
    st.caption(f"Word count: {wc}")

    # per-criterion table
    st.subheader("Per-criterion Breakdown")
    df = pd.DataFrame([{
        "Criterion": r["criterion"],
        "Score (/100)": r["score"],
        "KeywordCov": r["keyword_coverage"],
        "SemSim_raw": r["semantic_similarity"],
        "ExtraVal": r["extra_value"],
        "Missing_mand": ", ".join(r["missing_mandatory"]) if r["missing_mandatory"] else "",
        "Missing_opt": ", ".join(r["missing_optional"]) if r["missing_optional"] else ""
    } for r in per_crit])
    st.dataframe(df, use_container_width=True)

    st.subheader("Detailed feedback")
    for r in per_crit:
        st.markdown(f"### {r['criterion']} — {r['score']} / 100")
        if r["description"]:
            st.markdown(f"**Rubric:** {r['description']}")
        st.markdown(f"**Feedback:** {r['feedback']}")
        if r["notes"]:
            st.markdown(f"**Notes:** {'; '.join(r['notes'])}")
        st.markdown("---")

    st.subheader("Diagnostics")
    st.write(diag)

    # full-row audit (exportable)
    st.subheader("Full Rubric Audit (JSON)")
    full_rows_results = {"overall": overall, "word_count": wc, "criteria": per_crit, "diagnostics": diag}
    st.download_button("Download JSON report", data=json.dumps(full_rows_results, indent=2), file_name="rubric_scores.json")

    st.info("Tip: Keep α/β balanced (0.45/0.45) and γ small (0.1). Document these defaults in your README and mention calibration plan with teacher labels.")
