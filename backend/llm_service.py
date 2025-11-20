import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"
def evaluate_answer(question, answer):
    prompt = f"""
    You are an AI interview evaluator.

    Analyze the candidate answer based on:
    1. Correctness
    2. Completeness
    3. Relevance
    4. Confidence
    5. Communication quality

    Interview Question: {question}
    Candidate Answer: {answer}

    Return ONLY a JSON object in this exact format:

    {{
      "confidence_score": number,
      "content_score": number,
      "overall_score": number,
      "feedback": "string",
      "mistakes": ["list", "of", "strings"]
    }}
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        data = response.json()
        raw_output = data.get("response", "").strip()

        return json.loads(raw_output)

    except Exception as e:
        print("❌ LLaMA Evaluation Error:", e)
        return {
            "confidence_score": 0,
            "content_score": 0,
            "overall_score": 0,
            "feedback": "LLM failed to generate a valid response.",
            "mistakes": []
        }
