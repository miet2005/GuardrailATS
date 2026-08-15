"""
End-to-end test script for the combined Tier 1 + Tier 2 Guardrail.
Runs the full orchestrator against both test resumes and prints a
readable summary, confirming the combined verdict logic works correctly,
including the chunked/max-pooled Tier 2 approach.
"""

from guardrail import run_full_guardrail_check


def print_report(pdf_path, label):
    """
    Runs the full guardrail check on a single PDF and prints a readable
    summary of the results.

    Args:
        pdf_path (str): path to the PDF to check.
        label (str): human-readable name for this PDF, used in the report header.
    """
    print(f"\n{'=' * 65}")
    print(f"GUARDRAIL CHECK: {label} ({pdf_path})")
    print(f"{'=' * 65}")

    result = run_full_guardrail_check(pdf_path)

    print(f"OVERALL STATUS: {result['overall_status']}")

    print(f"\nTier 1 (Structural) - Flagged spans: {len(result['tier1_flagged_spans'])}")
    for i, span in enumerate(result["tier1_flagged_spans"], start=1):
        print(f"  #{i}: \"{span['text'].strip()}\"")
        for reason in span["reasons"]:
            print(f"      - {reason}")

    print(f"\nTier 2 (Semantic) - Priority checks (on Tier 1 flagged text): "
          f"{len(result['tier2_priority_results'])}")
    for i, r in enumerate(result["tier2_priority_results"], start=1):
        flag = "INJECTION" if r["is_injection"] else "safe"
        print(f"  #{i} [{flag}] confidence={r['confidence']}: \"{r['text_checked']}\"")

    print(f"\nTier 2 (Semantic) - Chunked checks (on all visible text): "
          f"{len(result['tier2_chunk_results'])} chunks analyzed")
    flagged_chunks = [r for r in result["tier2_chunk_results"] if r["is_injection"]]
    print(f"  Chunks flagged as injection: {len(flagged_chunks)}")
    for i, r in enumerate(flagged_chunks, start=1):
        print(f"  #{i} [INJECTION] confidence={r['confidence']}: \"{r['text_checked']}\"")

    print(f"\nTier 2 - MAX-POOLED FINAL RESULT:")
    max_result = result["tier2_max_result"]
    print(f"  Label: {max_result['label']}")
    print(f"  Confidence: {max_result['confidence']}")
    print(f"  Is Injection: {max_result['is_injection']}")
    if max_result["text_checked"]:
        print(f"  Triggering text: \"{max_result['text_checked']}\"")


if __name__ == "__main__":
    print_report("../data/clean_resume.pdf", "Clean Resume")
    print_report("../data/malicious_resume.pdf", "Malicious Resume")
    print_report("../data/Miet-Pamecha-Resume.pdf", "Miet Resume")