import json
import subprocess

import requests


def call_ollama(prompt: str):
    """
    Calls the locally-running Ollama model and returns the response.
    """
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama3.1',
            'prompt': prompt,
            'stream': False
        },
        timeout=30
    )
    result = response.json()
    return result.get('response', '[]')

def extract_qa_with_llama(restored_text: str):
    """
    Extracts Q&A pairs from punctuation-restored interview transcript.
    """
    prompt = f"""
    You are an expert interview evaluator.

    Your task is:
    1. Identify ALL the interview questions in the transcript.
    2. Identify the candidate's answer for EACH question.
    3. Score EACH answer from 0–100 based on:
       - Relevance
       - Completeness
       - Clarity
       - Professionalism
    4. Provide an explanation as to why you gave that score for each answer

    SCORING RULES:
    - Score MUST be a number from 0 to 100.
    - Score MUST NOT be null.
    - Score MUST NOT be a string.
    - Every question MUST have a score.
    - Do NOT generate or invent questions or answers.
    - Each questions is followed by its answer
    - Do NOT rearrange answers to match with appropriate questions. 

    YOUR OUTPUT MUST BE VALID JSON ONLY.

    FORMAT:
    [
      {{
        "question": "string",
        "answer": "string",
        "score": 0-100,
        "explanation": "string",
      }}
    ]

    INTERVIEW TRANSCRIPT:
    \"\"\"{restored_text}\"\"\"
    """

    response_text = call_ollama(prompt)

    # Extract JSON safely
    try:
        start = response_text.index("[")
        end = response_text.rindex("]") + 1
        json_text = response_text[start:end]
        qa_list = json.loads(json_text)
    except:
        qa_list = []

    return qa_list
