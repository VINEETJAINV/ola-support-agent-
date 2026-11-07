#!/usr/bin/env python
"""
Script to validate KB documents meet requirements:
- Exactly 12 documents
- Each document has exactly 4 or 5 sentences
- One per required topic
"""
import os
import re
import sys

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "kb")

REQUIRED_TOPICS = {
    "ticket-priority-classification.txt": "ticket-priority classification",
    "sla-by-severity.txt": "SLA by severity",
    "escalation-matrix.txt": "escalation matrix",
    "refund-compensation.txt": "refund/compensation",
    "customer-communication-channels.txt": "customer-communication channels",
    "business-hours-holidays.txt": "business hours/holidays",
    "repeat-complaints.txt": "repeat complaints",
    "service-credits.txt": "service credits",
    "feedback-collection.txt": "feedback collection",
    "vip-handling.txt": "VIP handling",
    "outage-communication.txt": "outage communication",
    "data-retention.txt": "data retention",
}


def count_sentences(text: str) -> int:
    """Count sentences in text using regex."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'[.!?]+(?:\s+|$)', text.strip())
    # Filter out empty strings
    return len([s for s in sentences if s.strip()])


def check_kb():
    """Validate KB documents."""
    print("=" * 60)
    print("KB VALIDATION REPORT")
    print("=" * 60)
    
    files = sorted([f for f in os.listdir("kb") if f.endswith(".txt")])
    
    print(f"\nFound {len(files)} .txt files in kb/")
    
    if len(files) != 12:
        print(f"FAIL: Expected 12 files, found {len(files)}")
        return False
    
    print("File count: OK (12 files)")
    
    all_ok = True
    sentence_counts = {}
    
    for filename in sorted(files):
        filepath = os.path.join("kb", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        sentence_count = count_sentences(content)
        sentence_counts[filename] = sentence_count
        
        if 4 <= sentence_count <= 5:
            status = "OK"
        else:
            status = "FAIL"
            all_ok = False
        
        print(f"  {filename}: {sentence_count} sentences {status}")
    
    # Check all required topics present
    print("\nRequired topics check:")
    for filename, topic in REQUIRED_TOPICS.items():
        if filename in files:
            print(f"  {topic} ({filename}): OK")
        else:
            print(f"  {topic} ({filename}): MISSING")
            all_ok = False
    
    print(f"\nSentence count range: {min(sentence_counts.values())}-{max(sentence_counts.values())}")
    print(f"All documents 4-5 sentences: {'YES' if all(4 <= c <= 5 for c in sentence_counts.values()) else 'NO'}")
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    
    return all_ok


if __name__ == "__main__":
    os.makedirs("transcripts", exist_ok=True)
    
    all_ok = check_kb()
    
    # Save report
    with open("transcripts/kb_check.txt", "w", encoding="utf-8") as f:
        f.write("KB VALIDATION REPORT\n")
        f.write("=" * 60 + "\n")
        # Re-run to capture output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        check_kb()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        f.write(output)
    
    print("Report saved to transcripts/kb_check.txt")
    
    if not all_ok:
        sys.exit(1)