from .punctuation import restore_punctuation
from .qa_extractor import extract_qa_with_llama

def analyze_speech(raw_asr_text: str):
    """
    The complete pipeline:
    1. Punctuation restoration
    2. Q&A extraction using Llama
    3. Speech confidence scoring
    """
    # STEP 1 — Add punctuation
    restored_text = restore_punctuation(raw_asr_text)

    # STEP 2 — Extract Q&A from restored text
    qa_results = extract_qa_with_llama(restored_text)

    # STEP 3 — Compute speech confidence (avg of QA scores)
    if qa_results:
        avg_score = sum(item["score"] for item in qa_results) / len(qa_results)
    else:
        avg_score = 0

    return avg_score, qa_results, restored_text
