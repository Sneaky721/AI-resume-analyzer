import os
import re
from typing import List, Set, Optional, Tuple

from flask import Flask, flash, redirect, render_template, request, url_for
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None


app = Flask(__name__)
app.secret_key = "resume-analyzer-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB file limit

ALLOWED_EXTENSIONS = {"pdf", "txt"}

# Embedding model used for semantic skill detection/similarity.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
# Optional local LLM generation for suggestions (set ENABLE_LLM=1 to enable).
ENABLE_LLM = os.getenv("ENABLE_LLM", "0") == "1"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/flan-t5-small")

JD_SKILL_CANDIDATES_TOP_N = int(os.getenv("JD_SKILL_CANDIDATES_TOP_N", "18"))
RESUME_CANDIDATES_TOP_N = int(os.getenv("RESUME_CANDIDATES_TOP_N", "25"))
SKILL_EMBEDMENT_MIN_TOKEN_CHARS = int(os.getenv("SKILL_EMBEDMENT_MIN_TOKEN_CHARS", "3"))

# Token pattern that keeps common tech tokens like `c++`, `c#`, `node.js`, etc.
TOKEN_PATTERN = r"(?u)\b[\w\+\#\.\-]+\b"

# Large, beginner-friendly predefined skill list.
PREDEFINED_SKILLS = {
    # Programming
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "rust",
    "kotlin", "swift", "php", "ruby", "r", "matlab", "scala", "perl", "bash",
    # Web
    "html", "css", "react", "angular", "vue", "next.js", "node.js", "express",
    "flask", "django", "fastapi", "spring", "bootstrap", "tailwind", "jquery",
    # Data / AI
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
    "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "xgboost", "lightgbm", "tableau", "power bi", "data analysis", "data science",
    "llm", "prompt engineering",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis",
    "elasticsearch", "firebase", "dynamodb",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "github actions", "ci/cd", "linux", "unix", "nginx", "apache",
    # Tools / Testing / Methods
    "git", "github", "gitlab", "bitbucket", "jira", "postman", "rest api",
    "graphql", "microservices", "oop", "design patterns", "unit testing",
    "pytest", "selenium", "agile", "scrum", "problem solving", "communication",
}

NLP_MODEL = None
EMBED_MODEL = None
LLM_PIPELINE = None


def load_nlp_model():
    """
    Load spaCy model with fallback to a blank English pipeline if model is missing.
    """
    global NLP_MODEL
    if NLP_MODEL is not None:
        return NLP_MODEL

    if spacy is None:
        NLP_MODEL = None
        return NLP_MODEL

    try:
        NLP_MODEL = spacy.load("en_core_web_sm")
    except Exception:
        # Fallback keeps app functional even when model is not installed.
        NLP_MODEL = spacy.blank("en")
    return NLP_MODEL


def load_embedding_model():
    """
    Load sentence-transformers model for semantic similarity.
    Falls back to None if sentence-transformers isn't installed.
    """
    global EMBED_MODEL
    if EMBED_MODEL is not None:
        return EMBED_MODEL
    if SentenceTransformer is None:
        return None
    EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return EMBED_MODEL


def load_llm_pipeline():
    """
    Load a small local LLM for suggestions (free/open-source).
    Enabled only when ENABLE_LLM=1.
    """
    global LLM_PIPELINE
    if LLM_PIPELINE is not None:
        return LLM_PIPELINE
    if not ENABLE_LLM:
        return None
    if pipeline is None:
        return None
    try:
        LLM_PIPELINE = pipeline(task="text2text-generation", model=LLM_MODEL_NAME)
        return LLM_PIPELINE
    except Exception:  # pragma: no cover
        LLM_PIPELINE = None
        return None


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_stream) -> str:
    """
    Extract text from a PDF file stream using PyPDF2.
    """
    try:
        reader = PdfReader(file_stream)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as error:
        raise ValueError(f"Could not read PDF file: {error}") from error


def extract_text_from_txt(file_stream) -> str:
    """
    Extract text from a TXT file stream.
    """
    try:
        raw_bytes = file_stream.read()
        return raw_bytes.decode("utf-8", errors="ignore").strip()
    except Exception as error:
        raise ValueError(f"Could not read text file: {error}") from error


def preprocess_text(text: str) -> str:
    """
    Lowercase, tokenize, remove punctuation/stopwords, and normalize text.
    """
    cleaned = re.sub(r"\s+", " ", text).strip().lower()
    nlp = load_nlp_model()

    if nlp is None:
        # Regex fallback if spaCy import is unavailable.
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\+\#\.\-]*", cleaned)
        return " ".join(tokens)

    doc = nlp(cleaned)
    tokens = []
    for token in doc:
        if token.is_space or token.is_punct or token.is_stop:
            continue
        lemma = token.lemma_.strip().lower() if token.lemma_ else token.text.lower()
        if not lemma:
            continue
        if lemma == "-pron-":
            lemma = token.text.lower()
        tokens.append(lemma)
    return " ".join(tokens)


def extract_skills(raw_text: str, processed_text: str) -> List[str]:
    """
    Match predefined skills against both raw and processed text.
    """
    raw = raw_text.lower()
    processed = f" {processed_text.lower()} "
    found = set()

    for skill in PREDEFINED_SKILLS:
        skill_l = skill.lower()
        pattern = rf"\b{re.escape(skill_l)}\b"
        if re.search(pattern, raw) or f" {skill_l} " in processed:
            found.add(skill)

    return sorted(found)


def calculate_match_score(resume_processed: str, jd_processed: str) -> float:
    """
    Calculate TF-IDF cosine similarity score between resume and job description.
    """
    if not resume_processed or not jd_processed:
        return 0.0

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=TOKEN_PATTERN)
    tfidf_matrix = vectorizer.fit_transform([resume_processed, jd_processed])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(similarity) * 100, 2)


def extract_top_keywords(text: str, top_n: int = 12) -> List[str]:
    """
    Extract top keywords from text based on term frequency.
    """
    tokens = text.split()
    counts = {}
    for token in tokens:
        if len(token) < 3:
            continue
        counts[token] = counts.get(token, 0) + 1
    sorted_tokens = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [token for token, _ in sorted_tokens[:top_n]]


def extract_top_tfidf_ngrams(text: str, top_n: int = 18, ngram_range: Tuple[int, int] = (1, 2)) -> List[str]:
    """
    Extract candidate skill phrases from TF-IDF n-grams (no fixed dictionary required).
    """
    if not text:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        token_pattern=TOKEN_PATTERN,
        min_df=1,
        max_df=1.0,
    )
    tfidf = vectorizer.fit_transform([text])
    scores = tfidf.toarray().flatten()
    features = vectorizer.get_feature_names_out()

    ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
    candidates: List[str] = []
    for phrase, _score in ranked:
        phrase = phrase.strip()
        if not phrase:
            continue
        parts = phrase.split()
        if any(len(p) < 2 for p in parts):
            continue
        if len(parts) == 1 and len(parts[0]) < SKILL_EMBEDMENT_MIN_TOKEN_CHARS:
            continue
        if len(parts) > 3:
            continue
        candidates.append(phrase.replace("_", " "))
        if len(candidates) >= top_n:
            break
    return candidates


def split_into_chunks(text: str, max_words: int = 180, overlap_words: int = 40) -> List[str]:
    """
    Split long text into overlapping chunks for embedding-based similarity.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap_words
        if start < 0:
            start = 0
    return chunks


def select_skills_by_embeddings(
    candidates: List[str],
    text_for_matching: str,
    embed_model,
) -> List[str]:
    """
    For each candidate phrase, compute max cosine similarity against chunks of the text.
    """
    if not candidates or not text_for_matching.strip():
        return []

    chunks = split_into_chunks(text_for_matching)
    if not chunks:
        return []

    # Normalize embeddings so dot-product equals cosine similarity.
    chunk_vecs = embed_model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    cand_vecs = embed_model.encode(candidates, normalize_embeddings=True, show_progress_bar=False)

    sims = cand_vecs @ chunk_vecs.T  # [num_candidates, num_chunks]
    max_sims = sims.max(axis=1)

    best = float(max_sims.max()) if len(max_sims) else 0.0
    cutoff = max(0.25, best * 0.55)
    present = [c for c, s in zip(candidates, max_sims) if float(s) >= cutoff]

    # Relax cutoff once if we found nothing.
    if not present and best > 0:
        cutoff = max(0.18, best * 0.35)
        present = [c for c, s in zip(candidates, max_sims) if float(s) >= cutoff]
    return present


def extract_skills_embeddings(resume_processed: str, jd_processed: str):
    """
    Embedding-driven skills extraction.
    Returns: (resume_skills, jd_skills, common_skills, missing_skills, extra_resume_skills)
    """
    embed_model = load_embedding_model()
    if embed_model is None:
        raise RuntimeError("Embeddings not available")

    jd_candidates = extract_top_tfidf_ngrams(jd_processed, top_n=JD_SKILL_CANDIDATES_TOP_N, ngram_range=(1, 2))
    resume_candidates = extract_top_tfidf_ngrams(resume_processed, top_n=RESUME_CANDIDATES_TOP_N, ngram_range=(1, 2))

    common_skills = select_skills_by_embeddings(jd_candidates, resume_processed, embed_model)
    jd_set = set(jd_candidates)
    missing_skills = [c for c in jd_candidates if c not in set(common_skills)]

    # Extra resume skills: resume candidates not in JD that are not similar to any JD candidate.
    extra_candidates = [c for c in resume_candidates if c not in jd_set]
    if not extra_candidates or not jd_candidates:
        extra_resume_skills = []
    else:
        jd_vecs = embed_model.encode(jd_candidates, normalize_embeddings=True, show_progress_bar=False)
        extra_vecs = embed_model.encode(extra_candidates, normalize_embeddings=True, show_progress_bar=False)
        sims = extra_vecs @ jd_vecs.T  # [extra, jd]
        max_sims = sims.max(axis=1)
        extra_cutoff = 0.35
        extra_resume_skills = [c for c, s in zip(extra_candidates, max_sims) if float(s) < extra_cutoff]

    resume_skills = sorted(set(common_skills).union(set(extra_resume_skills)))
    jd_skills = sorted(jd_set)
    return set(resume_skills), set(jd_skills), common_skills, missing_skills, extra_resume_skills


def generate_suggestions_with_llm(
    match_score: float,
    missing_skills: Set[str],
    jd_keywords: List[str],
    resume_processed: str,
    common_skills: Optional[List[str]] = None,
    extra_resume_skills: Optional[List[str]] = None,
) -> List[str]:
    """
    Optionally generate tips with a local open-source LLM.
    Falls back to deterministic template suggestions if LLM isn't enabled/available.
    """
    if not ENABLE_LLM:
        return generate_suggestions(match_score, missing_skills, jd_keywords, resume_processed)

    llm = load_llm_pipeline()
    if llm is None:
        return generate_suggestions(match_score, missing_skills, jd_keywords, resume_processed)

    try:
        prompt = (
            "You are an expert career coach and resume writer.\n"
            "Generate EXACTLY 5 practical resume improvement tips for the candidate to match the job description.\n"
            "Return a numbered list with 1 to 5.\n"
            "Each tip must be actionable (what to add/change), avoid repetition, and focus on missing skills and keyword optimization.\n\n"
            f"Match score: {match_score}%\n"
            f"Common skills: {', '.join(common_skills[:8]) if common_skills else 'None'}\n"
            f"Missing skills: {', '.join(sorted(list(missing_skills))[:10]) if missing_skills else 'None'}\n"
            f"Extra resume skills: {', '.join(extra_resume_skills[:8]) if extra_resume_skills else 'None'}\n"
            f"Job keyword focus: {', '.join(jd_keywords[:12]) if jd_keywords else ''}\n"
        )
        out = llm(prompt, max_new_tokens=220, do_sample=False)
        text = out[0].get("generated_text", "") if out else ""
    except Exception:
        return generate_suggestions(match_score, missing_skills, jd_keywords, resume_processed)

    tips: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(\d+)[\.\)]\s+", line):
            tips.append(re.sub(r"^(\d+)[\.\)]\s+", "", line).strip())

    if len(tips) >= 3:
        return tips[:5]
    return generate_suggestions(match_score, missing_skills, jd_keywords, resume_processed)


def generate_suggestions(
    match_score: float,
    missing_skills: Set[str],
    jd_keywords: List[str],
    resume_processed: str,
) -> List[str]:
    """
    Build practical, interview-ready suggestions based on analysis outputs.
    """
    suggestions = []

    if match_score < 50:
        suggestions.append("Your resume is not closely aligned with this role. Tailor it section-by-section to match the job description.")
    elif match_score < 75:
        suggestions.append("Your resume has moderate alignment. Improve matching by adding role-specific keywords and measurable achievements.")
    else:
        suggestions.append("Your resume is already strongly aligned. Fine-tune wording and project impact to make it stand out even more.")

    if missing_skills:
        top_missing = ", ".join(sorted(list(missing_skills))[:5])
        suggestions.append(f"Add evidence of these missing skills if you have used them: {top_missing}.")

    keyword_gaps = [kw for kw in jd_keywords if kw not in resume_processed]
    if keyword_gaps:
        suggestions.append(
            "Include important job keywords naturally in your summary, projects, and experience: "
            + ", ".join(keyword_gaps[:5])
            + "."
        )

    suggestions.append("Quantify impact in bullet points (for example: reduced processing time by 30% or improved test coverage to 85%).")
    suggestions.append("Keep your resume ATS-friendly: use standard headings, clean formatting, and avoid keyword stuffing.")

    return suggestions[:5]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume_file" not in request.files:
        flash("Please upload a resume file.")
        return redirect(url_for("index"))

    file = request.files["resume_file"]
    job_description = request.form.get("job_description", "").strip()

    if not file or file.filename == "":
        flash("Please choose a resume file.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload PDF or TXT.")
        return redirect(url_for("index"))

    if not job_description:
        flash("Please paste a job description before analyzing.")
        return redirect(url_for("index"))

    extension = file.filename.rsplit(".", 1)[1].lower()

    try:
        if extension == "pdf":
            resume_text = extract_text_from_pdf(file.stream)
        else:
            resume_text = extract_text_from_txt(file.stream)
    except ValueError as error:
        flash(str(error))
        return redirect(url_for("index"))

    if not resume_text.strip():
        flash("Uploaded resume appears to be empty. Please upload a valid file.")
        return redirect(url_for("index"))

    resume_processed = preprocess_text(resume_text)
    jd_processed = preprocess_text(job_description)

    match_score = calculate_match_score(resume_processed, jd_processed)

    # Skills extraction:
    # - Prefer embedding-based detection (less hard-coded).
    # - Fall back to dictionary-based extraction if embeddings aren't available.
    try:
        resume_skills, jd_skills, common_skills, missing_skills, extra_resume_skills = extract_skills_embeddings(
            resume_processed=resume_processed,
            jd_processed=jd_processed,
        )
        missing_skills = sorted(missing_skills)
        extra_resume_skills = sorted(extra_resume_skills)
    except Exception:
        resume_skills = set(extract_skills(resume_text, resume_processed))
        jd_skills = set(extract_skills(job_description, jd_processed))

        common_skills = sorted(resume_skills.intersection(jd_skills))
        missing_skills = sorted(jd_skills - resume_skills)
        extra_resume_skills = sorted(resume_skills - jd_skills)

    jd_keywords = extract_top_keywords(jd_processed)
    suggestions = generate_suggestions_with_llm(
        match_score=match_score,
        missing_skills=set(missing_skills),
        jd_keywords=jd_keywords,
        resume_processed=resume_processed,
        common_skills=common_skills,
        extra_resume_skills=extra_resume_skills,
    )

    return render_template(
        "result.html",
        match_score=match_score,
        resume_skills=sorted(resume_skills),
        jd_skills=sorted(jd_skills),
        common_skills=common_skills,
        missing_skills=missing_skills,
        extra_resume_skills=extra_resume_skills,
        suggestions=suggestions,
    )


if __name__ == "__main__":
    # Turn debug on for local development only.
    app.run(debug=True)
