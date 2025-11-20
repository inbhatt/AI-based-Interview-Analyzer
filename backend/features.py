import os
import cv2
import numpy as np
import mediapipe as mp
import speech_recognition as sr
import subprocess
import json
import re
from datetime import datetime
from deepface import DeepFace
import requests

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

# Confidence Calculation Weights
CONF_WEIGHTS = {
    "expression": 0.25,
    "eye_movement": 0.20,
    "speech": 0.35,
    "gesture": 0.20
}


def format_time(dt):
    """Helper function to format datetime"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def analyze_speech_with_llama(video_path):
    """
    Extracts audio, transcribes it, identifies Q&A pairs,
    and evaluates answers using Llama 3.1 via Ollama.
    Returns an overall speech confidence score.
    """
    recognizer = sr.Recognizer()
    audio_path = os.path.join(os.path.dirname(video_path), "audio.wav")

    # Extract audio using ffmpeg
    command = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y'
    process = subprocess.run(command, shell=True, capture_output=True, text=True, encoding="utf-8")

    if process.returncode != 0:
        print("FFmpeg Error:", process.stderr)
        return 70, []  # Default confidence if extraction fails

    if not os.path.exists(audio_path):
        print("Error: Audio file was not created.")
        return 70, []

    try:
        # Transcribe audio
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"Recognized Speech: {text}")

        # Step 1: Identify questions in the text using Llama
        questions = identify_questions_with_llama(text)

        if not questions:
            print("No questions detected in the audio.")
            # Fallback: Analyze speech quality (filler words, clarity)
            return analyze_speech_quality(text), []

        # Step 2: Extract Q&A pairs
        qa_pairs = extract_qa_pairs(text, questions)

        # Step 3: Evaluate each answer with Llama
        scores = []
        detailed_results = []

        for qa in qa_pairs:
            question = qa['question']
            answer = qa['answer']

            score = evaluate_answer_with_llama(question, answer)
            scores.append(score)

            detailed_results.append({
                'question': question,
                'answer': answer,
                'score': score
            })

            print(f"Q: {question[:50]}... | A Score: {score}%")

        # Calculate overall speech confidence
        if scores:
            overall_speech_confidence = sum(scores) / len(scores)
        else:
            overall_speech_confidence = 70  # Default

        # Clean up audio file
        try:
            os.remove(audio_path)
        except:
            pass

        return round(overall_speech_confidence, 2), detailed_results

    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio.")
        return 50, []
    except sr.RequestError as e:
        print(f"Could not request results from speech recognition service; {e}")
        return 70, []
    except Exception as e:
        print(f"Speech analysis error: {e}")
        return 70, []


def identify_questions_with_llama(text):
    """
    Uses Llama to identify questions in the transcribed text.
    Returns a list of questions found.
    """
    prompt = f"""Analyze the following interview transcript and identify all questions asked by the interviewer.
Return ONLY a JSON array of questions, with no additional text or explanation.

Transcript: "{text}"

Format your response as a JSON array like this:
["Question 1?", "Question 2?", "Question 3?"]

If no questions are found, return an empty array: []
"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.1',
                'prompt': prompt,
                'stream': False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '[]')

            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                return questions

        return []

    except Exception as e:
        print(f"Error identifying questions with Llama: {e}")
        # Fallback: Use simple pattern matching
        return identify_questions_fallback(text)


def identify_questions_fallback(text):
    """
    Fallback method to identify questions using pattern matching.
    """
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    questions = []

    for sentence in sentences:
        sentence = sentence.strip()
        # Check if sentence ends with question mark or starts with question words
        if sentence.endswith('?') or any(sentence.lower().startswith(q) for q in
                                         ['what', 'why', 'how', 'when', 'where', 'who',
                                          'can you', 'could you', 'tell me', 'describe']):
            if len(sentence) > 10:  # Filter out very short phrases
                questions.append(sentence + ('?' if not sentence.endswith('?') else ''))

    return questions


def extract_qa_pairs(text, questions):
    """
    Extracts question-answer pairs from the text.
    Each answer is the text between two consecutive questions.
    """
    qa_pairs = []

    for i, question in enumerate(questions):
        # Find the position of current question
        question_start = text.find(question)
        if question_start == -1:
            continue

        # Find the start of the answer (after the question)
        answer_start = question_start + len(question)

        # Find the end of the answer (start of next question or end of text)
        if i + 1 < len(questions):
            next_question = questions[i + 1]
            answer_end = text.find(next_question, answer_start)
            if answer_end == -1:
                answer_end = len(text)
        else:
            answer_end = len(text)

        # Extract answer
        answer = text[answer_start:answer_end].strip()

        # Only add if answer has substantial content
        if len(answer) > 10:
            qa_pairs.append({
                'question': question,
                'answer': answer
            })

    return qa_pairs


def evaluate_answer_with_llama(question, answer):
    """
    Uses Llama to evaluate how appropriate and accurate an answer is.
    Returns a percentage score (0-100).
    """
    prompt = f"""You are evaluating an interview answer. Rate the quality of the answer on a scale of 0-100 based on:
1. Relevance to the question
2. Clarity and coherence
3. Completeness
4. Professionalism

Question: "{question}"
Answer: "{answer}"

Respond with ONLY a number between 0 and 100, nothing else.
"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.1',
                'prompt': prompt,
                'stream': False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '70').strip()

            # Extract number from response
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                score = int(numbers[0])
                return min(100, max(0, score))  # Clamp between 0-100

        return 70  # Default score

    except Exception as e:
        print(f"Error evaluating answer with Llama: {e}")
        return 70


def analyze_speech_quality(text):
    """
    Fallback method to analyze speech quality based on filler words and clarity.
    """
    from .models import Speech

    speech_data = {s.word.lower(): s.percentage for s in Speech.objects.all()}
    words = text.lower().split()

    if len(words) == 0:
        return 70

    # Count filler words and calculate deduction
    total_deduction = 0
    for word, weight in speech_data.items():
        matches = text.lower().count(word)
        if matches > 0:
            total_deduction += (matches * weight / len(words) * 100)

    speech_confidence = max(0, 100 - total_deduction)
    return round(speech_confidence, 2)


def analyze_video(video_path):
    """
    Comprehensive video analysis including facial expressions,
    eye movement, hand gestures, and speech evaluation.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return get_default_results()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps > 0 else 1

    start_time = datetime.now()
    print(f"Starting video analysis. Time: {format_time(start_time)}")
    print(f"Video: {total_frames} frames, {fps} fps, {video_duration:.2f}s duration")

    # Initialize counters
    frame_count = 0
    expression_counts = {}
    expression_seconds = {}

    # Eye tracking
    eyes_forward_count = 0
    eyes_down_count = 0
    eyes_away_count = 0

    # Hand gesture tracking
    hand_present_count = 0
    hand_movement_detected = 0
    previous_hand_positions = []

    # Initialize MediaPipe
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Process every Nth frame for efficiency (adjust based on video length)
    frame_skip = max(1, int(fps / 5)) if fps > 0 else 1  # Process ~5 frames per second

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for efficiency
        if frame_count % frame_skip != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # ===== FACIAL EXPRESSION DETECTION =====
        try:
            analysis = DeepFace.analyze(
                rgb_frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend='opencv'
            )

            if analysis and len(analysis) > 0:
                expression = analysis[0]["dominant_emotion"]
                # Normalize expression names to match database
                expression = expression.lower()
                expression_counts[expression] = expression_counts.get(expression, 0) + 1

        except Exception as e:
            pass  # Continue if face detection fails for this frame

        # ===== EYE MOVEMENT DETECTION =====
        face_results = face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Eye landmarks (MediaPipe Face Mesh indices)
                left_eye_center = face_landmarks.landmark[468]  # Left eye center
                right_eye_center = face_landmarks.landmark[473]  # Right eye center
                nose_tip = face_landmarks.landmark[1]  # Nose tip

                # Calculate eye direction based on vertical position
                avg_eye_y = (left_eye_center.y + right_eye_center.y) / 2

                # Determine gaze direction
                if avg_eye_y > nose_tip.y + 0.03:  # Looking down
                    eyes_down_count += 1
                elif abs(avg_eye_y - nose_tip.y) < 0.03:  # Looking forward
                    eyes_forward_count += 1
                else:  # Looking away/up
                    eyes_away_count += 1

        # ===== HAND GESTURE DETECTION =====
        hand_results = hands.process(rgb_frame)

        if hand_results.multi_hand_landmarks:
            hand_present_count += 1

            # Calculate hand movement
            current_positions = []
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Use wrist position as reference point
                wrist = hand_landmarks.landmark[0]
                current_positions.append((wrist.x, wrist.y))

            # Detect movement by comparing with previous frame
            if previous_hand_positions:
                for curr, prev in zip(current_positions, previous_hand_positions):
                    distance = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
                    if distance > 0.02:  # Movement threshold
                        hand_movement_detected += 1
                        break

            previous_hand_positions = current_positions
        else:
            previous_hand_positions = []

    cap.release()
    face_mesh.close()
    hands.close()

    end_time = datetime.now()
    print(f"Video processing complete. Time: {format_time(end_time)}")
    print(f"Duration: {(end_time - start_time).total_seconds():.2f}s")

    # ===== CALCULATE EXPRESSION CONFIDENCE =====
    from .models import Expression

    expression_data = {e.name.lower(): e.percentage for e in Expression.objects.all()}

    # Convert frame counts to seconds
    processed_frames = frame_count // frame_skip
    for expr, count in expression_counts.items():
        expression_seconds[expr] = round((count / processed_frames) * video_duration, 2)

    # Calculate weighted expression confidence
    expression_confidence = 0
    total_expression_time = sum(expression_seconds.values())

    if total_expression_time > 0:
        for expr, seconds in expression_seconds.items():
            weight = expression_data.get(expr, 50)  # Default 50 if not in DB
            time_fraction = seconds / total_expression_time
            expression_confidence += (weight * time_fraction)
    else:
        expression_confidence = 60  # Default if no expressions detected

    # ===== CALCULATE EYE MOVEMENT CONFIDENCE =====
    total_eye_frames = eyes_forward_count + eyes_down_count + eyes_away_count

    if total_eye_frames > 0:
        # Forward gaze = confident (80%), Down = less confident (40%), Away = least confident (30%)
        eye_movement_confidence = (
                (eyes_forward_count / total_eye_frames) * 85 +
                (eyes_down_count / total_eye_frames) * 50 +
                (eyes_away_count / total_eye_frames) * 35
        )
    else:
        eye_movement_confidence = 60  # Default

    # ===== CALCULATE HAND GESTURE CONFIDENCE =====
    if processed_frames > 0:
        hand_presence_ratio = hand_present_count / processed_frames
        hand_movement_ratio = hand_movement_detected / processed_frames if hand_present_count > 0 else 0

        # Moderate hand movement is good (too much or too little is less confident)
        optimal_movement = 0.3  # 30% of frames should show movement
        movement_score = 100 - abs(hand_movement_ratio - optimal_movement) * 200

        # Presence of hands is positive
        presence_score = hand_presence_ratio * 100

        hand_gesture_confidence = (movement_score * 0.6 + presence_score * 0.4)
        hand_gesture_confidence = max(30, min(95, hand_gesture_confidence))  # Clamp between 30-95
    else:
        hand_gesture_confidence = 60

    # ===== SPEECH ANALYSIS WITH LLAMA =====
    print("Starting speech analysis with Llama...")
    speech_confidence, qa_results = analyze_speech_with_llama(video_path)

    # ===== CALCULATE OVERALL CONFIDENCE =====
    overall_confidence = (
            (CONF_WEIGHTS["expression"] * expression_confidence) +
            (CONF_WEIGHTS["eye_movement"] * eye_movement_confidence) +
            (CONF_WEIGHTS["speech"] * speech_confidence) +
            (CONF_WEIGHTS["gesture"] * hand_gesture_confidence)
    )

    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Expression: {expression_confidence:.2f}%")
    print(f"Eye Movement: {eye_movement_confidence:.2f}%")
    print(f"Speech: {speech_confidence:.2f}%")
    print(f"Hand Gesture: {hand_gesture_confidence:.2f}%")
    print(f"Overall: {overall_confidence:.2f}%")

    return {
        "expression_seconds": expression_seconds,
        "expression_confidence": round(expression_confidence, 2),
        "eye_movement_confidence": round(eye_movement_confidence, 2),
        "speech_confidence": round(speech_confidence, 2),
        "hand_gesture_confidence": round(hand_gesture_confidence, 2),
        "overall_confidence": round(overall_confidence, 2),
        "qa_analysis": qa_results  # Detailed Q&A evaluation
    }


def get_default_results():
    """Returns default results if video processing fails."""
    return {
        "expression_seconds": {"neutral": 0},
        "expression_confidence": 60.0,
        "eye_movement_confidence": 60.0,
        "speech_confidence": 60.0,
        "hand_gesture_confidence": 60.0,
        "overall_confidence": 60.0,
        "qa_analysis": []
    }