"""
Runs the full GuardrailATS pipeline against the entire test suite:
  - Every malicious resume variant, through the two-tier guardrail
  - Every clean resume against its matching JD, through the hybrid scorer
  - A cross-mismatch check (junior backend resume vs senior backend JD)

Prints a consolidated report so results can be reviewed all at once,
rather than running individual scripts repeatedly.
"""

import os

from pdf_inspector import extract_text_spans
from guardrail import run_full_guardrail_check
from rag_engine import compute_hybrid_match_score


TEST_SUITE_DIR = os.path.join("..", "data", "test_suite")
CLEAN_DIR = os.path.join(TEST_SUITE_DIR, "clean_resumes")
JD_DIR = os.path.join(TEST_SUITE_DIR, "job_descriptions")
MALICIOUS_DIR = os.path.join(TEST_SUITE_DIR, "malicious_resumes")


def load_resume_text_from_pdf(pdf_path):
    """Extracts all visible text from a resume PDF as one newline-joined string."""
    spans = extract_text_spans(pdf_path)
    return "\n".join(span["text"] for span in spans)


def load_jd_text(jd_path):
    """Reads a job description from a plain text file."""
    with open(jd_path, "r", encoding="utf-8") as f:
        return f.read()


def run_guardrail_tests():
    """
    Runs the two-tier guardrail check against every malicious resume
    variant, printing whether each was caught and by which tier(s).
    """
    print(f"\n{'#' * 70}")
    print("PART 1: GUARDRAIL TESTS (malicious resume variants)")
    print(f"{'#' * 70}")

    malicious_files = sorted(os.listdir(MALICIOUS_DIR))

    for filename in malicious_files:
        pdf_path = os.path.join(MALICIOUS_DIR, filename)

        result = run_full_guardrail_check(pdf_path)

        tier1_caught = len(result["tier1_flagged_spans"]) > 0
        tier2_caught = result["tier2_max_result"]["is_injection"]

        print(f"\n--- {filename} ---")
        print(f"  Overall status: {result['overall_status']}")
        print(f"  Tier 1 caught it: {tier1_caught}")
        print(f"  Tier 2 caught it: {tier2_caught} (confidence: {result['tier2_max_result']['confidence']})")

        if not tier1_caught and not tier2_caught:
            print("  *** NEITHER TIER CAUGHT THIS - full miss, needs investigation ***")


def run_matched_scoring_tests():
    """
    Runs the hybrid scorer for each clean resume against its intended
    matching JD, to confirm genuinely good matches score reasonably high
    across different job domains (not just backend).
    """
    print(f"\n{'#' * 70}")
    print("PART 2: MATCHED SCORING TESTS (clean resume vs its intended JD)")
    print(f"{'#' * 70}")

    pairs = [
        ("frontend_sarah.pdf", "frontend_jd.txt"),
        ("data_scientist_alex.pdf", "data_scientist_jd.txt"),
        ("marketing_emily.pdf", "marketing_jd.txt"),
    ]

    for resume_file, jd_file in pairs:
        resume_text = load_resume_text_from_pdf(os.path.join(CLEAN_DIR, resume_file))
        jd_text = load_jd_text(os.path.join(JD_DIR, jd_file))

        result = compute_hybrid_match_score(resume_text, jd_text)

        print(f"\n--- {resume_file} vs {jd_file} ---")
        print(f"  BM25 score: {result['bm25_score_percent']}%")
        print(f"  Embedding score: {result['embedding_score_percent']}%")
        print(f"  FINAL HYBRID SCORE: {result['overall_score_percent']}%")


def run_mismatch_scoring_test():
    """
    Runs the hybrid scorer for a deliberately weak match: a junior
    backend resume (minimal skills) against a senior backend JD (high
    experience/skill requirements). Both are nominally "backend," so
    this is a subtler mismatch than comparing across unrelated domains -
    tests whether the scorer can distinguish skill-level depth, not
    just topical relevance.
    """
    print(f"\n{'#' * 70}")
    print("PART 3: SUBTLE MISMATCH TEST (junior resume vs senior JD, same domain)")
    print(f"{'#' * 70}")

    resume_text = load_resume_text_from_pdf(os.path.join(CLEAN_DIR, "junior_backend_tom.pdf"))
    jd_text = load_jd_text(os.path.join(JD_DIR, "senior_backend_jd.txt"))

    result = compute_hybrid_match_score(resume_text, jd_text)

    print(f"\n--- junior_backend_tom.pdf vs senior_backend_jd.txt ---")
    print(f"  BM25 score: {result['bm25_score_percent']}%")
    print(f"  Embedding score: {result['embedding_score_percent']}%")
    print(f"  FINAL HYBRID SCORE: {result['overall_score_percent']}%")


if __name__ == "__main__":
    run_guardrail_tests()
    run_matched_scoring_tests()
    run_mismatch_scoring_test()

    print(f"\n{'#' * 70}")
    print("TEST SUITE COMPLETE")
    print(f"{'#' * 70}")