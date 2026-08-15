"""
End-to-end test script for the resume/JD matching engine. Compares the
original embedding-only score against the new hybrid (BM25 + embeddings)
score, side by side, using our real test resume and sample JD.
"""

from pdf_inspector import extract_text_spans
from rag_engine import compute_match_score, compute_hybrid_match_score


def load_resume_text(pdf_path):
    """Extracts all visible text from a resume PDF as one string."""
    spans = extract_text_spans(pdf_path)
    return "\n".join(span["text"] for span in spans)


def load_jd_text(jd_path):
    """Reads a job description from a plain text file."""
    with open(jd_path, "r", encoding="utf-8") as f:
        return f.read()


def print_comparison_report(resume_path, jd_path):
    """
    Runs both the embedding-only and hybrid scoring pipelines and prints
    a side-by-side comparison, so we can see how BM25 shifts the score.
    """
    resume_text = load_resume_text(resume_path)
    jd_text = load_jd_text(jd_path)

    embedding_only_result = compute_match_score(resume_text, jd_text)
    hybrid_result = compute_hybrid_match_score(resume_text, jd_text)

    print(f"\n{'=' * 65}")
    print(f"SCORE COMPARISON: {resume_path} vs {jd_path}")
    print(f"{'=' * 65}")
    print(f"Embedding-only score: {embedding_only_result['overall_score_percent']}%")
    print(f"\nHybrid score breakdown:")
    print(f"  BM25 component:       {hybrid_result['bm25_score_percent']}%")
    print(f"  Embedding component:  {hybrid_result['embedding_score_percent']}%")
    print(f"  FINAL HYBRID SCORE:   {hybrid_result['overall_score_percent']}%")

    print(f"\nPer-requirement BM25 matches:")
    for match in hybrid_result["bm25_details"]["jd_chunk_scores"]:
        print(f"\n  JD: \"{match['jd_chunk']}\"")
        print(f"  Best BM25 match: \"{match['best_matching_resume_chunk']}\"")
        print(f"  Raw BM25 score: {match['raw_bm25_score']}")


def print_mismatch_sanity_check(resume_path):
    """
    Sanity check: runs the hybrid scorer against a deliberately unrelated
    job description (Marketing Manager) to confirm the hybrid approach
    still correctly scores a genuine mismatch LOW, not just inflating
    every score regardless of actual fit.
    """
    resume_text = load_resume_text(resume_path)

    unrelated_jd_text = """Marketing Manager

We are seeking an experienced Marketing Manager to lead our brand strategy.

Requirements:
- 5+ years of experience in brand marketing and campaign management
- Strong copywriting and content creation skills
- Experience with social media strategy and influencer partnerships
- Familiarity with Adobe Creative Suite and Canva
- Excellent verbal and written communication skills

Nice to have:
- Experience with email marketing platforms like Mailchimp
- Bachelor's degree in Marketing or Communications
"""

    hybrid_result = compute_hybrid_match_score(resume_text, unrelated_jd_text)
    embedding_only_result = compute_match_score(resume_text, unrelated_jd_text)

    print(f"\n{'=' * 65}")
    print(f"MISMATCH SANITY CHECK: {resume_path} vs unrelated Marketing JD")
    print(f"{'=' * 65}")
    print(f"Embedding-only score: {embedding_only_result['overall_score_percent']}%")
    print(f"\nHybrid score breakdown:")
    print(f"  BM25 component:       {hybrid_result['bm25_score_percent']}%")
    print(f"  Embedding component:  {hybrid_result['embedding_score_percent']}%")
    print(f"  FINAL HYBRID SCORE:   {hybrid_result['overall_score_percent']}%")

if __name__ == "__main__":
    print_comparison_report("../data/clean_resume.pdf", "../data/sample_jd.txt")
    print_mismatch_sanity_check("../data/clean_resume.pdf")