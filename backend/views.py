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
from .models import Expression, Eyes, HandsExpression, Speech, User,CandidateRecord
import glob
from .forms import ProfileUpdateForm
import shutil
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.utils import timezone
from django.conf import settings
from datetime import datetime


from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import User

def signup_view(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        password = request.POST.get("password")
        role = request.POST.get("role")

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
        else:
            user = User.objects.create_user(email=email, name=name, role=role, password=password)
            messages.success(request, "Signup successful! Please login.")
            return redirect("login")

    return render(request, "auth.html")


def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        user = authenticate(request, email=email, password=password)

        if user is not None:
            login(request, user)
            
            return redirect("home")  # redirect to dashboard/home
        else:
            messages.error(request, "Invalid email or password!")

    return render(request, "auth.html")


def logout_view(request):
    logout(request)
    return redirect("login")




def candidate_profile(request):
    user = request.user

    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, request.FILES, instance=user)
        if form.is_valid():
            form.save()
            return redirect('home')  # redirect to home after saving
    else:
        form = ProfileUpdateForm(instance=user)

    context = {
        'user': user,
        'form': form
    }
    return render(request, 'admin.html', context)


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

# @csrf_exempt
# def upload_video(request):
#     if request.method == "POST" and request.FILES.get("video"):
#         cleanup_files()
#         video_file = request.FILES["video"]
#         file_path = default_storage.save("backend/videos/" + video_file.name, ContentFile(video_file.read()))

#         user = request.user
#         #result = analyze_video(file_path)
#         result = analyze_video(user)



#         return JsonResponse({
#             "confidence_result": result,
#             "expression_counts": result["expression_counts"]  # Add this
#         })

#     return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
def upload_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]

        # Save uploaded file
        relative_path = f"backend/videos/{video_file.name}"
        full_path = os.path.join(settings.MEDIA_ROOT, relative_path)

        # Make sure the folder exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Write file manually
        with open(full_path, "wb") as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # Copy to copies folder
        copy_folder = os.path.join(settings.MEDIA_ROOT, "video_copies")
        os.makedirs(copy_folder, exist_ok=True)
        shutil.copy(full_path, os.path.join(copy_folder, video_file.name))

        # Static analysis results
        result = {
            "expression_counts": {"happy": 2, "sad": 1},
            "expression_confidence": 80,
            "eye_movement_confidence": 85,
            "speech_confidence": 90,
            "hand_gesture_confidence": 75,
            "overall_confidence": 82
        }

        # Store in database
        CandidateRecord.objects.create(
            user=request.user,
            video_name=video_file.name,
            expression_confidence=result["expression_confidence"],
            eye_movement_confidence=result["eye_movement_confidence"],
            speech_confidence=result["speech_confidence"],
            hand_gesture_confidence=result["hand_gesture_confidence"],
            overall_confidence=result["overall_confidence"],
            date_time=datetime.now()
        )

        return JsonResponse({
            "confidence_result": result,
            "expression_counts": result["expression_counts"]
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




def analyze_video(user):
    # cap = cv2.VideoCapture(video_path)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # video_duration = total_frames / fps if fps else 1  # Avoid division by zero

    # start_time = datetime.now()
    # print(f"Starting processing. Current Time: {format_time(start_time)}")

    # frame_count = 0
    # expression_counts = {}
    # eyes_down_count = 0
    # eyes_forward_count = 0
    # no_hand_movement_count = 0
    # hand_movement_count = 0

    # face_mesh = mp_face_mesh.FaceMesh()
    # hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     frame_count += 1
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #     print(f"Processing frame {frame_count}/{total_frames}")

    #     # Facial Expression Detection
    #     try:
    #         analysis = DeepFace.analyze(rgb_frame, actions=["emotion"], enforce_detection=False)
    #         if analysis:
    #             expression = analysis[0]["dominant_emotion"]
    #             expression_counts[expression] = expression_counts.get(expression, 0) + 1
    #     except Exception as e:
    #         print(f"DeepFace error: {e}")

    #     # Eye Movement Detection
    #     h, w, _ = frame.shape
    #     face_results = face_mesh.process(rgb_frame)
    #     hand_results = hands.process(rgb_frame)

    #     if face_results.multi_face_landmarks:
    #         for face_landmarks in face_results.multi_face_landmarks:
    #             left_eye = face_landmarks.landmark[159].y
    #             right_eye = face_landmarks.landmark[145].y
    #             nose = face_landmarks.landmark[1].y

    #             if left_eye > nose and right_eye > nose:
    #                 eyes_down_count += 1
    #             else:
    #                 eyes_forward_count += 1

    #     # Hand Gesture Detection
    #     if hand_results.multi_hand_landmarks:
    #         hand_movement_count += 1
    #     else:
    #         no_hand_movement_count += 1

    # cap.release()
    # end_time = datetime.now()
    # print(f"Ended processing. Current Time: {format_time(end_time)}")
    # print(f"Time taken: {str(end_time - start_time).split('.')[0]}")

    # # **Fetching predefined confidence values from the database**
    # expression_data = {e.name.lower(): e.percentage for e in Expression.objects.all()}
    # eye_data = {e.side.lower(): e.percentage for e in Eyes.objects.all()}
    # hand_data = {h.move.lower(): h.percentage for h in HandsExpression.objects.all()}

    # # **Calculating confidence scores**
    # expression_confidence = 0
    # for expr, count in expression_counts.items():
    #     expr_weight = expression_data.get(expr.lower(), 0)  # Default to 0 if not found
    #     time_fraction = (count / frame_count) * video_duration if frame_count else 0
    #     expression_confidence += (expr_weight * time_fraction / video_duration)

    # eye_movement_confidence = (
    #     eye_data.get("forward", 0) * (eyes_forward_count / frame_count)
    #     if frame_count else 0
    # )
    # eye_movement_confidence += (
    #     eye_data.get("down", 0) * (eyes_down_count / frame_count)
    #     if frame_count else 0
    # )

    # hand_gesture_confidence = (
    #     hand_data.get("moving", 0) * (hand_movement_count / frame_count)
    #     if frame_count else 0
    # )

    # # Speech Analysis
    # speech_confidence = analyze_speech(video_path)

    # # **Weighted Average Calculation**
    # overall_confidence = (
    #     (CONF_WEIGHTS["expression"] * expression_confidence) +
    #     (CONF_WEIGHTS["eye_movement"] * eye_movement_confidence) +
    #     (CONF_WEIGHTS["speech"] * speech_confidence) +
    #     (CONF_WEIGHTS["gesture"] * hand_gesture_confidence)
    # )

    # return {
    #     "expression_counts": expression_counts,
    #     "expression_confidence": round(expression_confidence, 2),
    #     "eye_movement_confidence": round(eye_movement_confidence, 2),
    #     "speech_confidence": round(speech_confidence, 2),
    #     "hand_gesture_confidence": round(hand_gesture_confidence, 2),
    #     "overall_confidence": round(overall_confidence, 2),
    # }

    # Instead of running heavy AI analysis, set static (dummy) values
    expression_confidence = 0.85
    eye_movement_confidence = 0.78
    speech_confidence = 0.90
    hand_gesture_confidence = 0.82

    overall_confidence = round(
        (0.25 * expression_confidence) +
        (0.25 * eye_movement_confidence) +
        (0.25 * speech_confidence) +
        (0.25 * hand_gesture_confidence),
        2
    )

    # 🔹 Save record in database
    if user:
        from .models import CandidateRecord
        CandidateRecord.objects.create(
            user=user,
            expression_confidence=expression_confidence,
            eye_movement_confidence=eye_movement_confidence,
            speech_confidence=speech_confidence,
            hand_gesture_confidence=hand_gesture_confidence,
            overall_confidence=overall_confidence,
        )

    return {
        "expression_counts": {"happy": 10, "neutral": 5, "sad": 3},
        "expression_confidence": expression_confidence,
        "eye_movement_confidence": eye_movement_confidence,
        "speech_confidence": speech_confidence,
        "hand_gesture_confidence": hand_gesture_confidence,
        "overall_confidence": overall_confidence,
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


def home(request):
    user = request.user
    return render(request, 'home.html', {'user': user})



def auth(request: HttpRequest):
    return render(request, 'auth.html')

def format_time(dt):
    """Format datetime to 'dd-MM-yy HH:mm:ss' format."""
    return dt.strftime("%d-%m-%y %H:%M:%S")

