"""
RAG Engine Module - Embedding-based Resume vs Job Description Matching

Chunks both the resume and job description into small pieces (reusing the
same line-based chunking used by the Guardrail module), embeds every chunk
using a sentence-transformer model, then compares them using cosine
similarity.

Aggregation strategy: for each individual JD requirement/chunk, we find its
single best-matching resume chunk. We then average those best-match scores
across all JD chunks to produce one overall match percentage. This answers
the real ATS question - "for each thing this job needs, did the candidate
demonstrate it somewhere in their resume?" - rather than blending everything
into a vague overall average.
"""

import json
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np

from utils import chunk_text_for_analysis

# Load variables from .env (GROQ_API_KEY, LLM_BACKEND) into the environment
load_dotenv()

# Module-level cache: holds the loaded embedding model so we only load it
# once per program run, not on every single function call.
_embedding_model = None


def load_embedding_model():
    """
    Loads the all-MiniLM-L6-v2 sentence embedding model, caching it in
    memory after the first call so subsequent calls are fast.

    The first call will trigger a one-time download of the model weights
    (~80MB) from Hugging Face's servers if not already cached locally.
    Every call after that reuses the local cache.

    Returns:
        A SentenceTransformer model instance ready to embed text.
    """
    global _embedding_model

    if _embedding_model is None:
        print("Loading sentence embedding model (first time may take a moment)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded successfully.")

    return _embedding_model


def embed_chunks(chunks):
    """
    Converts a list of text chunks into a matrix of embedding vectors.

    Args:
        chunks (list[str]): text pieces to embed.

    Returns:
        numpy.ndarray: shape (num_chunks, 384) - one 384-dimensional vector
            per input chunk. Returns an empty array if chunks is empty.
    """
    if not chunks:
        return np.array([])

    model = load_embedding_model()
    embeddings = model.encode(chunks)
    return embeddings


def compute_match_score(resume_text, jd_text):
    """
    Computes an ATS-style match score between a resume and a job description
    using chunked embeddings and cosine similarity.

    For each JD chunk (representing one requirement/line), we find its
    single best-matching resume chunk by cosine similarity. The final score
    is the average of these best-match scores across all JD chunks,
    converted to a 0-100 percentage.

    Args:
        resume_text (str): full extracted resume text.
        jd_text (str): full job description text.

    Returns:
        dict: {
            "overall_score_percent": float,     # 0-100 final match score
            "jd_chunk_matches": list[dict],      # per-JD-chunk best match details
            "resume_chunks": list[str],          # resume chunks used
            "jd_chunks": list[str],              # JD chunks used
        }
    """
    # --- Chunk both documents ---
    resume_chunks = chunk_text_for_analysis(resume_text)
    jd_chunks = chunk_text_for_analysis(jd_text)

    # Guard against either document being empty after chunking
    if not resume_chunks or not jd_chunks:
        return {
            "overall_score_percent": 0.0,
            "jd_chunk_matches": [],
            "resume_chunks": resume_chunks,
            "jd_chunks": jd_chunks,
        }

    # --- Embed both sets of chunks ---
    resume_embeddings = embed_chunks(resume_chunks)
    jd_embeddings = embed_chunks(jd_chunks)

    # --- Compute the full similarity matrix ---
    # Shape: (num_jd_chunks, num_resume_chunks) - each row is one JD chunk's
    # similarity score against every resume chunk.
    similarity_matrix = cosine_similarity(jd_embeddings, resume_embeddings)

    # --- For each JD chunk, find its single best-matching resume chunk ---
    jd_chunk_matches = []
    best_match_scores = []

    for jd_index, jd_chunk in enumerate(jd_chunks):
        similarities_for_this_jd_chunk = similarity_matrix[jd_index]

        # Find the index of the highest-scoring resume chunk for this JD chunk
        best_resume_index = np.argmax(similarities_for_this_jd_chunk)
        best_score = float(similarities_for_this_jd_chunk[best_resume_index])
        best_resume_chunk = resume_chunks[best_resume_index]

        jd_chunk_matches.append({
            "jd_chunk": jd_chunk,
            "best_matching_resume_chunk": best_resume_chunk,
            "similarity_score": round(best_score, 4),
        })

        best_match_scores.append(best_score)

    # --- Aggregate: average the best-match scores across all JD chunks ---
    average_similarity = sum(best_match_scores) / len(best_match_scores)

    # Cosine similarity for real sentence embeddings typically falls in a
    # 0-1 range for related text (rarely negative in practice for this kind
    # of text). We convert directly to a 0-100 percentage.
    overall_score_percent = round(average_similarity * 100, 2)

    return {
        "overall_score_percent": overall_score_percent,
        "jd_chunk_matches": jd_chunk_matches,
        "resume_chunks": resume_chunks,
        "jd_chunks": jd_chunks,
    }




# General English stopwords - common words with no real topical meaning.
# This is a compact hand-picked list (not an exhaustive NLP corpus), which
# is sufficient here since our documents (resumes/JDs) use fairly
# predictable, formal language rather than open-domain text.
GENERAL_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "and", "or", "but",
    "this", "that", "these", "those", "as", "by", "from", "we", "our",
    "you", "your", "will", "have", "has", "had", "can", "should", "would",
}

# Domain-specific "boilerplate" words common in job descriptions and resumes.
# These carry grammatical/filler meaning in this specific context but aren't
# themselves skills or requirements - matching on them alone shouldn't drive
# a strong BM25 score. This list is a practical, hand-tuned starting point,
# not an exhaustive or scientifically derived set - worth revisiting if
# testing reveals other overly-generic words inflating scores.
JD_RESUME_BOILERPLATE_STOPWORDS = {
    "experience", "experienced", "years", "year", "strong", "familiarity",
    "understanding", "skills", "skill", "required", "requirements",
    "preferred", "looking", "join", "team", "role", "candidate",
    "responsibilities", "ability", "knowledge", "working", "work",
}

BM25_STOPWORDS = GENERAL_STOPWORDS | JD_RESUME_BOILERPLATE_STOPWORDS


def tokenize_for_bm25(text):
    """
    Tokenizer for BM25: lowercases, strips basic punctuation, splits on
    whitespace, and removes stopwords (both general English filler words
    and JD/resume-specific boilerplate terms like "experience" or "years").

    Stopword removal matters more here than in typical BM25 use cases: our
    IDF statistics are computed over a tiny per-document corpus (a single
    resume's ~10-13 chunks), which isn't enough data for IDF alone to
    reliably down-weight generic words the way it would in a large-corpus
    search engine. Explicit stopword removal compensates for that.

    Args:
        text (str): text to tokenize.

    Returns:
        list[str]: lowercase word tokens, with stopwords removed.
    """
    text = text.lower()
    for char in ".,;:!?()[]{}\"'":
        text = text.replace(char, "")

    tokens = text.split()
    return [token for token in tokens if token not in BM25_STOPWORDS]


def compute_bm25_score(resume_chunks, jd_chunks):
    """
    Computes a keyword-based relevance score between resume chunks and JD
    chunks using BM25 (the algorithm behind most production search engines,
    including Elasticsearch's default ranking).

    Same aggregation philosophy as compute_match_score(): for each JD chunk,
    find its best-matching resume chunk, then average those best scores.

    Raw BM25 scores are unbounded (not naturally 0-1 like cosine similarity),
    so we normalize by dividing each JD chunk's best score by the highest
    score seen across all JD chunks in this comparison - this keeps the
    output in a comparable 0-1 range, at the cost of the normalization being
    relative to this specific resume/JD pair rather than a fixed universal
    scale. This is a common, pragmatic approach for BM25 since there's no
    natural upper bound to normalize against otherwise.

    Args:
        resume_chunks (list[str]): resume text chunks.
        jd_chunks (list[str]): JD text chunks.

    Returns:
        dict: {
            "normalized_score_0_to_1": float,
            "jd_chunk_scores": list[dict],  # per-JD-chunk best match + raw score
        }
    """
    if not resume_chunks or not jd_chunks:
        return {"normalized_score_0_to_1": 0.0, "jd_chunk_scores": []}

    tokenized_resume_chunks = [tokenize_for_bm25(chunk) for chunk in resume_chunks]
    bm25_index = BM25Okapi(tokenized_resume_chunks)

    jd_chunk_scores = []
    raw_best_scores = []

    for jd_chunk in jd_chunks:
        tokenized_query = tokenize_for_bm25(jd_chunk)
        # get_scores returns one score per resume chunk, for this JD chunk's query
        scores_against_all_resume_chunks = bm25_index.get_scores(tokenized_query)

        best_index = int(np.argmax(scores_against_all_resume_chunks))
        best_raw_score = float(scores_against_all_resume_chunks[best_index])

        jd_chunk_scores.append({
            "jd_chunk": jd_chunk,
            "best_matching_resume_chunk": resume_chunks[best_index],
            "raw_bm25_score": round(best_raw_score, 4),
        })
        raw_best_scores.append(best_raw_score)

    # Normalize using a FIXED ceiling, not the max score within this specific
    # comparison. Relative (in-comparison) normalization has a real flaw: if
    # every JD chunk has genuinely weak overlap with the resume (a true
    # mismatch), one chunk is still "relatively best" and gets scaled up to
    # look like a strong match - manufacturing a falsely high score out of
    # uniformly weak signals. A fixed ceiling means uniformly weak scores
    # stay uniformly low, as they should.
    #
    # BM25_SCORE_CEILING is an empirical estimate of what a genuinely strong
    # single-chunk match tends to score with our short (line-length) chunks
    # and default BM25Okapi parameters. This is a tunable constant, not a
    # universal law - if real-world testing shows scores consistently
    # saturating at 1.0 (ceiling too low) or staying suspiciously low even
    # for good matches (ceiling too high), this value should be adjusted.
    BM25_SCORE_CEILING = 8.0

    normalized_scores = [
        min(s / BM25_SCORE_CEILING, 1.0) for s in raw_best_scores
    ]

    normalized_score_0_to_1 = sum(normalized_scores) / len(normalized_scores)

    return {
        "normalized_score_0_to_1": normalized_score_0_to_1,
        "jd_chunk_scores": jd_chunk_scores,
    }


def compute_hybrid_match_score(resume_text, jd_text, bm25_weight=0.45, embedding_weight=0.55):
    """
    Computes a hybrid ATS match score combining BM25 (lexical/keyword
    matching) and sentence embeddings (semantic matching), similar in
    spirit to how production ATS/search systems blend multiple relevance
    signals rather than relying on a single technique.

    - BM25 catches exact/near-exact keyword matches confidently (e.g., a
      flat skill list like "Python, SQL, Docker, Git" scores well here,
      even though its grammatical structure differs from a full JD
      sentence, which sometimes penalized embedding-only scoring).
    - Embeddings catch paraphrased/synonym matches BM25 would completely
      miss (e.g., "cloud infrastructure" matching "AWS").

    Args:
        resume_text (str): full resume text.
        jd_text (str): full job description text.
        bm25_weight (float): weight given to the BM25 score (0-1).
        embedding_weight (float): weight given to the embedding score (0-1).
            bm25_weight + embedding_weight should sum to 1.0.

    Returns:
        dict: {
            "overall_score_percent": float,       # final hybrid 0-100 score
            "bm25_score_percent": float,           # BM25 component, 0-100
            "embedding_score_percent": float,      # embedding component, 0-100
            "bm25_details": dict,                  # from compute_bm25_score
            "embedding_details": dict,              # from compute_match_score
        }
    """
    resume_chunks = chunk_text_for_analysis(resume_text)
    jd_chunks = chunk_text_for_analysis(jd_text)

    embedding_result = compute_match_score(resume_text, jd_text)
    bm25_result = compute_bm25_score(resume_chunks, jd_chunks)

    embedding_score_percent = embedding_result["overall_score_percent"]
    bm25_score_percent = round(bm25_result["normalized_score_0_to_1"] * 100, 2)

    overall_score_percent = round(
        (bm25_weight * bm25_score_percent) + (embedding_weight * embedding_score_percent),
        2,
    )

    return {
        "overall_score_percent": overall_score_percent,
        "bm25_score_percent": bm25_score_percent,
        "embedding_score_percent": embedding_score_percent,
        "bm25_details": bm25_result,
        "embedding_details": embedding_result,
    }


OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "llama3.2:3b"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# Which backend to use for skill extraction: "ollama" (local, free, requires
# Ollama running) or "groq" (cloud, free tier, requires GROQ_API_KEY in .env).
# This is the ONLY switch needed to move between local development and
# cloud deployment - the rest of the pipeline is unaffected either way.
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()


def build_skill_extraction_prompt(resume_text, jd_text):
    """
    Builds a structured prompt instructing the LLM to compare a resume
    against a job description and return matched/missing skills plus a
    brief summary, formatted strictly as JSON.

    Args:
        resume_text (str): full resume text.
        jd_text (str): full job description text.

    Returns:
        str: the complete prompt to send to the LLM.
    """
    prompt = f"""You are an expert technical recruiter assistant. Compare the RESUME against the JOB DESCRIPTION below.

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

Respond with ONLY a valid JSON object, no other text before or after it, in exactly this format:
{{
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "summary": "A two-sentence plain-English summary of how well this candidate fits the role."
}}

Rules:
- "matched_skills" must list specific skills/technologies mentioned in the JOB DESCRIPTION that are also clearly present in the RESUME.
- "missing_skills" must list specific skills/technologies mentioned in the JOB DESCRIPTION that are NOT found anywhere in the RESUME.
- Do not invent skills that are not explicitly mentioned in the JOB DESCRIPTION.
- Output ONLY the JSON object. Do not include any explanation, preamble, or markdown formatting like ```json.
"""
    return prompt


def _call_ollama(prompt, timeout_seconds):
    """
    Sends a prompt to the local Ollama server and returns the raw text
    response. Isolated so generate_skill_analysis() can dispatch to this
    or _call_groq() based on LLM_BACKEND, without duplicating error
    handling logic in the main function.

    Args:
        prompt (str): the full prompt to send.
        timeout_seconds (int): how long to wait before giving up.

    Returns:
        tuple[str, str]: (raw_model_output, error_message). error_message
            is an empty string on success.
    """
    request_payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=request_payload, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return "", "Could not connect to Ollama. Is it running? Try 'ollama serve' or check the system tray."
    except requests.exceptions.Timeout:
        return "", f"Ollama did not respond within {timeout_seconds} seconds."
    except requests.exceptions.RequestException as e:
        return "", f"Ollama request failed: {e}"

    response_data = response.json()
    return response_data.get("response", ""), ""


def _call_groq(prompt, timeout_seconds):
    """
    Sends a prompt to Groq's cloud API (OpenAI-compatible chat completions
    format) and returns the raw text response. Requires GROQ_API_KEY to be
    set in the environment (loaded from .env via load_dotenv() at module
    import time).

    Args:
        prompt (str): the full prompt to send.
        timeout_seconds (int): how long to wait before giving up.

    Returns:
        tuple[str, str]: (raw_model_output, error_message). error_message
            is an empty string on success.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "", "GROQ_API_KEY is not set. Add it to your .env file."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_payload = {
        "model": GROQ_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},  # ask Groq to constrain output to valid JSON
        "temperature": 0.2,  # low temperature for more consistent, less creative structured output
    }

    try:
        response = requests.post(
            GROQ_API_URL, headers=headers, json=request_payload, timeout=timeout_seconds
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return "", f"Groq did not respond within {timeout_seconds} seconds."
    except requests.exceptions.RequestException as e:
        return "", f"Groq request failed: {e}"

    response_data = response.json()
    try:
        return response_data["choices"][0]["message"]["content"], ""
    except (KeyError, IndexError):
        return "", "Unexpected response format from Groq."


def generate_skill_analysis(resume_text, jd_text, timeout_seconds=60):
    """
    Calls an LLM (local Ollama or cloud Groq, based on LLM_BACKEND) to
    extract matched/missing skills and a summary, comparing a resume
    against a job description.

    This function is intentionally the ONLY place that dispatches to the
    LLM backend. Switching between local development (Ollama) and cloud
    deployment (Groq) requires only changing LLM_BACKEND in .env - the
    rest of the pipeline (chunking, embeddings, scoring) is unaffected.

    Args:
        resume_text (str): full resume text.
        jd_text (str): full job description text.
        timeout_seconds (int): how long to wait for the LLM before giving up.

    Returns:
        dict: {
            "matched_skills": list[str],
            "missing_skills": list[str],
            "summary": str,
            "raw_model_output": str,   # unparsed response, for debugging
            "parse_success": bool,      # whether JSON parsing succeeded
        }
    """
    prompt = build_skill_extraction_prompt(resume_text, jd_text)

    if LLM_BACKEND == "groq":
        raw_model_output, error = _call_groq(prompt, timeout_seconds)
    else:
        raw_model_output, error = _call_ollama(prompt, timeout_seconds)

    if error:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "summary": "",
            "raw_model_output": "",
            "parse_success": False,
            "error": error,
        }

    # Smaller local models occasionally wrap the JSON with extra
    # conversational text (e.g., "Here's the analysis: {...}") despite
    # being instructed to return only JSON. Before parsing, we extract
    # just the substring between the first '{' and the last '}', which
    # strips away this kind of wrapper text in the common case.
    first_brace = raw_model_output.find("{")
    last_brace = raw_model_output.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned_output = raw_model_output[first_brace: last_brace + 1]
    else:
        # No braces found at all - nothing to extract, pass through as-is
        # so the parse attempt below fails cleanly and reports the issue.
        cleaned_output = raw_model_output

    # Attempt to parse the (cleaned) model output as JSON. Even with
    # format="json" requested, we defensively handle parse failures
    # rather than assuming success, since smaller local models can
    # still occasionally produce malformed JSON even after cleanup.
    try:
        parsed = json.loads(cleaned_output)
        matched_skills = parsed.get("matched_skills", [])
        missing_skills = parsed.get("missing_skills", [])
        summary = parsed.get("summary", "")
        parse_success = True
    except json.JSONDecodeError:
        matched_skills = []
        missing_skills = []
        summary = ""
        parse_success = False

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "summary": summary,
        "raw_model_output": raw_model_output,
        "parse_success": parse_success,
    }


if __name__ == "__main__":
    # Quick manual test with two short examples - one clearly related pair,
    # one clearly unrelated pair - to sanity-check the scoring makes sense
    # before testing against real resume/JD files.
    test_jd = "Requires 3+ years experience with Python and REST APIs."
    test_resume_good = "Built REST APIs using Python and FastAPI for 4 years."
    test_resume_bad = "Experienced chef specializing in French pastry techniques."

    print("\n--- Testing a GOOD match (should score high) ---")
    result_good = compute_match_score(test_resume_good, test_jd)
    print(f"Score: {result_good['overall_score_percent']}%")

    print("\n--- Testing a BAD match (should score low) ---")
    result_bad = compute_match_score(test_resume_bad, test_jd)
    print(f"Score: {result_bad['overall_score_percent']}%")