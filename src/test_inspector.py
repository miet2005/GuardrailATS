"""
End-to-end test script for the PDF Inspector.
Runs the anomaly detector against both the clean and malicious test resumes
and prints a readable report, so we can visually confirm the detector works
correctly before wiring it into the full application.
"""

from pdf_inspector import extract_text_spans, detect_anomalies


def run_inspection(pdf_path, label):
    """
    Runs the full inspection pipeline on a single PDF and prints a report.

    Args:
        pdf_path (str): path to the PDF to inspect.
        label (str): human-readable name for this PDF, used in the printed report.
    """
    print(f"\n{'=' * 60}")
    print(f"INSPECTING: {label} ({pdf_path})")
    print(f"{'=' * 60}")

    spans = extract_text_spans(pdf_path)
    print(f"Total text spans extracted: {len(spans)}")

    flagged = detect_anomalies(spans)

    if not flagged:
        print("RESULT: No anomalies detected. This file appears clean.")
    else:
        print(f"RESULT: {len(flagged)} suspicious span(s) detected!\n")
        for i, span in enumerate(flagged, start=1):
            print(f"  --- Flagged Span #{i} ---")
            print(f"  Text: \"{span['text'].strip()}\"")
            print(f"  Page: {span['page_number']}")
            print(f"  Reasons:")
            for reason in span["reasons"]:
                print(f"    - {reason}")
            print()


if __name__ == "__main__":
    run_inspection("../data/Miet-Pamecha-Resume.pdf", "My Resume")
    run_inspection("../data/clean_resume.pdf", "Clean Resume")
    run_inspection("../data/malicious_resume.pdf", "Malicious Resume")