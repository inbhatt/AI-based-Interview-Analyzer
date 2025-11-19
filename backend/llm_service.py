import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"

def evaluate_answer(question, answer):
    """
    Sends the interview question & candidate answer to LLaMA 3.1
    and returns structured scoring + feedback.
    """

    prompt = f"""
    You are an AI interview evaluator.

    Analyze the candidate answer based on:
    1. Correctness
    2. Completeness
    3. Relevance
    4. Confidence (clarity, hesitation, filler words, uncertainty)
    5. Communication quality

    Interview Question: {question}
    Candidate Answer: {answer}

    Return ONLY a JSON object in this exact format:

    {{
        "is_correct": true/false,
        "content_score": number (0-100),
        "confidence_score": number (0-100),
        "overall_score": number (0-100),
        "mistakes": ["list the mistakes"],
        "feedback": "short paragraph feedback"
    }}
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    raw_text = response.json().get("response", "")

    # Extract JSON from LLM output
    try:
        data = json.loads(raw_text)
        return data
    except json.JSONDecodeError:
        return {
            "is_correct": False,
            "content_score": 0,
            "confidence_score": 0,
            "overall_score": 0,
            "mistakes": ["LLM returned invalid JSON"],
            "feedback": raw_text
        }
