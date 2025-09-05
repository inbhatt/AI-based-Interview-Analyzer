import os
import cv2
import numpy as np
import mediapipe as mp
import speech_recognition as sr
import nltk
import re
from deepface import DeepFace
from django.shortcuts import render
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from datetime import datetime
from django.db.models import Avg
from .models import Expression, Eyes, HandsExpression, Speech
import glob

# Load NLP data
nltk.download("punkt")
nltk.download("punkt_tab")

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Confidence Calculation Weights
CONF_WEIGHTS = {
    "expression": 0.4,
    "eye_movement": 0.2,
    "speech": 0.2,
    "gesture": 0.2
}

@csrf_exempt
def upload_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        cleanup_files()
        video_file = request.FILES["video"]
        file_path = default_storage.save("backend/videos/" + video_file.name, ContentFile(video_file.read()))

        result = analyze_video(file_path)

        return JsonResponse({
            "confidence_result": result,
            "expression_counts": result["expression_counts"]  # Add this
        })

    return JsonResponse({"error": "Invalid request"}, status=400)

def cleanup_files():
    # Delete video file
    videos_folder = os.path.join(os.path.dirname(__file__), 'videos')
    
    # Get a list of all files in the videos folder
    files = glob.glob(os.path.join(videos_folder, '*'))  # Get all files

    for file in files:
        try:
            if os.path.isfile(file):
                os.remove(file)  # Remove the file
                print(f"Deleted file: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")




def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_duration = total_frames / fps if fps else 1  # Avoid division by zero

    start_time = datetime.now()
    print(f"Starting processing. Current Time: {format_time(start_time)}")

    frame_count = 0
    expression_counts = {}
    eyes_down_count = 0
    eyes_forward_count = 0
    no_hand_movement_count = 0
    hand_movement_count = 0

    face_mesh = mp_face_mesh.FaceMesh()
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print(f"Processing frame {frame_count}/{total_frames}")

        # Facial Expression Detection
        try:
            analysis = DeepFace.analyze(rgb_frame, actions=["emotion"], enforce_detection=False)
            if analysis:
                expression = analysis[0]["dominant_emotion"]
                expression_counts[expression] = expression_counts.get(expression, 0) + 1
        except Exception as e:
            print(f"DeepFace error: {e}")

        # Eye Movement Detection
        h, w, _ = frame.shape
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                left_eye = face_landmarks.landmark[159].y
                right_eye = face_landmarks.landmark[145].y
                nose = face_landmarks.landmark[1].y

                if left_eye > nose and right_eye > nose:
                    eyes_down_count += 1
                else:
                    eyes_forward_count += 1

        # Hand Gesture Detection
        if hand_results.multi_hand_landmarks:
            hand_movement_count += 1
        else:
            no_hand_movement_count += 1

    cap.release()
    end_time = datetime.now()
    print(f"Ended processing. Current Time: {format_time(end_time)}")
    print(f"Time taken: {str(end_time - start_time).split('.')[0]}")

    # **Fetching predefined confidence values from the database**
    expression_data = {e.name.lower(): e.percentage for e in Expression.objects.all()}
    eye_data = {e.side.lower(): e.percentage for e in Eyes.objects.all()}
    hand_data = {h.move.lower(): h.percentage for h in HandsExpression.objects.all()}

    # **Calculating confidence scores**
    expression_confidence = 0
    for expr, count in expression_counts.items():
        expr_weight = expression_data.get(expr.lower(), 0)  # Default to 0 if not found
        time_fraction = (count / frame_count) * video_duration if frame_count else 0
        expression_confidence += (expr_weight * time_fraction / video_duration)

    eye_movement_confidence = (
        eye_data.get("forward", 0) * (eyes_forward_count / frame_count)
        if frame_count else 0
    )
    eye_movement_confidence += (
        eye_data.get("down", 0) * (eyes_down_count / frame_count)
        if frame_count else 0
    )

    hand_gesture_confidence = (
        hand_data.get("moving", 0) * (hand_movement_count / frame_count)
        if frame_count else 0
    )

    # Speech Analysis
    speech_confidence = analyze_speech(video_path)

    # **Weighted Average Calculation**
    overall_confidence = (
        (CONF_WEIGHTS["expression"] * expression_confidence) +
        (CONF_WEIGHTS["eye_movement"] * eye_movement_confidence) +
        (CONF_WEIGHTS["speech"] * speech_confidence) +
        (CONF_WEIGHTS["gesture"] * hand_gesture_confidence)
    )

    return {
        "expression_counts": expression_counts,
        "expression_confidence": round(expression_confidence, 2),
        "eye_movement_confidence": round(eye_movement_confidence, 2),
        "speech_confidence": round(speech_confidence, 2),
        "hand_gesture_confidence": round(hand_gesture_confidence, 2),
        "overall_confidence": round(overall_confidence, 2),
    }


import subprocess

def analyze_speech(video_path):
    recognizer = sr.Recognizer()
    audio_path = os.path.join(os.path.dirname(video_path), "audio.wav")

    # Extract audio using ffmpeg
    command = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y'
    process = subprocess.run(command, shell=True, capture_output=True, text=True, encoding="utf-8")

    if process.returncode != 0:
        print("FFmpeg Error:", process.stderr)
        return 100  # Default confidence if extraction fails

    if not os.path.exists(audio_path):
        print("Error: Audio file was not created.")
        return 100  # Default confidence if audio file is missing

    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)  # Uncomment for actual speech recognition
            #text = "so uhh tell me about yourself like the most common you know what I mean job interview question of all time usually the thing that they ask you"
            print(f"Recognized Speech: {text}")

            # Fetch low-confidence words and their confidence percentages from the database
            speech_data = {s.word.lower(): s.percentage for s in Speech.objects.all()}

            words_count = len(nltk.word_tokenize(text))
            if words_count == 0:
                return 100  # If no words are spoken, assume full confidence

            # Count occurrences of each low-confidence word
            total_deduction = 0
            for word, weight in speech_data.items():
                matches = len(re.findall(rf"\b{word}\b", text, re.IGNORECASE))
                if matches > 0:
                    total_deduction += (matches * weight)

            # Normalize confidence to a 0-100 scale
            speech_confidence = max(0, 100 - total_deduction)

            return round(speech_confidence, 2)

    except Exception as e:
        print(f"Speech recognition error: {e}")
        return 100  # Default full confidence if speech cannot be analyzed.


def home(request: HttpRequest):
    return render(request, 'home.html')

def admin_panel(request):
    return render(request, 'admin.html')

def history(request):
    return render(request, 'history.html')

def report(request):
    return render(request, 'report.html')

def format_time(dt):
    """Format datetime to 'dd-MM-yy HH:mm:ss' format."""
    return dt.strftime("%d-%m-%y %H:%M:%S")
