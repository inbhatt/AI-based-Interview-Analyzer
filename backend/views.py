import os
import cv2
import numpy as np
import mediapipe as mp
import speech_recognition as sr
import nltk
import re
import json
import subprocess

from deepface import DeepFace
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from datetime import datetime
from django.db.models import Avg
from django.contrib import messages
# 🚨 UPDATED IMPORTS 🚨
from .models import Expression, Eyes, HandsExpression, Speech, User, AnalysisResult 
from django.db.models import Avg, Count
import random
from django.http import FileResponse
from .llm_service import evaluate_answer   # import our function


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

# ==========================================================
# Authentication and Redirection Views
# ==========================================================

def auth(request: HttpRequest):
    return render(request, 'auth.html')

@csrf_exempt
def signup_user(request):
    if request.method == 'POST':
        try:
            # Assuming JSON payload from frontend
            data = json.loads(request.body)
            name = data.get('name')
            email = data.get('email')
            password = data.get('password')
            
            # The role is hardcoded to 'candidate' if this is used as the public signup endpoint
            # We assume your 'auth.html' signup is for candidates, and the admin page is separate.
            role = 'candidate' 
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'message': 'Invalid JSON data.'}, status=400)

        if not all([name, email, password]):
            return JsonResponse({'success': False, 'message': 'All fields are required.'})
        
        if User.objects.filter(email=email).exists():
            return JsonResponse({'success': False, 'message': 'Email already registered.'})

        try:
            user = User.objects.create(name=name, email=email, role=role)
            user.set_password(password)
            user.save()

            return JsonResponse({
                'success': True, 
                'message': 'Candidate profile successfully created. Redirecting to dashboard.',
                'redirect_url': '/admin-dashboard' 
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'An error occurred: {str(e)}'})

    return JsonResponse({'success': False, 'message': 'Invalid request method.'}, status=405)


@csrf_exempt
def login_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            password = data.get('password')
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'message': 'Invalid JSON data.'}, status=400)
        
        try:
            user = User.objects.get(email=email)
            if user.check_password(password):
                # Set session variables
                request.session['user_id'] = user.id
                request.session['user_role'] = user.role
                request.session['user_name'] = user.name
                request.session['user_email'] = user.email
                
                # 🚨 NEW ROLE-BASED REDIRECTION LOGIC 🚨
                if user.role == 'admin':
                    # Redirect to admin dashboard URL
                    redirect_url = '/admin-dashboard' 
                else:
                    # Redirect to candidate home page URL
                    redirect_url = '/home' 
                    
                return JsonResponse({
                    'success': True, 
                    'message': 'Login successful.', 
                    'redirect_url': redirect_url # Send URL back to frontend JS
                })
            else:
                return JsonResponse({'success': False, 'message': 'Invalid credentials.'})
        
        except User.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Invalid credentials.'})

    return JsonResponse({'success': False, 'message': 'Invalid request method.'}, status=405)


def logout_user(request):
    request.session.flush()
    # Redirect to the authentication page (login/signup)
    return redirect('auth') 


def admin_required(view_func):
    def wrapper(request, *args, **kwargs):
        # This already prevents candidates/unauthorized users from admin pages.
        if request.session.get('user_role') != 'admin':
            messages.error(request, "Access Denied: You must be an administrator.")
            return redirect('home') # Redirects non-admins to 'home'
        return view_func(request, *args, **kwargs)
    return wrapper

# ----------------------------------------------------------
# Redirection for the base path '/'
# ----------------------------------------------------------

def home_redirect(request: HttpRequest):
    """
    Checks if a user is logged in and redirects them based on their role.
    If not logged in, redirects to the auth page.
    """
    user_id = request.session.get('user_id')
    user_role = request.session.get('user_role')
    
    if not user_id:
        return redirect('auth') # Go to login/signup screen

    if user_role == 'admin':
        return redirect('admin_dashboard')
    else:
        # Default to the candidate home page
        return redirect('home')


def home(request: HttpRequest):
    """Renders the candidate home page, strictly denying access to admins."""
    user_role = request.session.get('user_role')
    user_id = request.session.get('user_id')

    # 1. Block unauthorized users
    if user_id is None:
        return redirect('auth')
        
    # 2. 🚨 BLOCK ADMINS 🚨
    if user_role == 'admin':
        messages.error(request, "Access Denied: Administrators cannot access the Candidate Portal.")
        return redirect('admin_dashboard')

    # Allow access only if logged in and not an admin
    return render(request, 'home.html')

# ----------------------------------------------------------
# Video Processing Views (Keep the original logic)
# ----------------------------------------------------------



@csrf_exempt
def upload_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        user_id = request.session.get('user_id')
        if not user_id:
            return JsonResponse({"error": "User not authenticated. Please log in."}, status=401)

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return JsonResponse({"error": "User not found."}, status=401)

        video_file = request.FILES["video"]
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        file_name, file_extension = os.path.splitext(video_file.name)

        file_path = default_storage.save(
            "backend/videos/" + str(user_id) + "_" + timestamp_str + file_extension,
            ContentFile(video_file.read())
        )

        # 🟦 STEP 1 — Your existing video analysis
        absolute_path = default_storage.path(file_path)
        result = analyze_video(absolute_path)

        # 🟩 STEP 2 — (Placeholder) Get transcript from the video
        # TODO: Replace this with your real speech-to-text system
        transcript_text = result.get("transcript", "I am placeholder text because transcript is not implemented yet.")

        # 🟧 STEP 3 — Send to LLaMA: Evaluate the answer
        interview_question = "Tell me about yourself."  # Replace with real question if you have
        llm_result = evaluate_answer(interview_question, transcript_text)

        
        # Extract LLM results
        llm_confidence = llm_result.get("confidence_score", 0)
        llm_content_score = llm_result.get("content_score", 0)
        llm_overall = llm_result.get("overall_score", 0)
        llm_feedback = llm_result.get("feedback", "")
        llm_mistakes = llm_result.get("mistakes", [])

        # 🟥 STEP 4 — Combine your model's confidence + LLM score
        final_confidence = int((result['overall_confidence'] + llm_confidence) / 2)

        # 🟦 STEP 5 — Save everything in DB
        AnalysisResult.objects.create(
            user=user,
            video_path=file_path,

            # your old scores
            overall_confidence=final_confidence,
            expression_confidence=result['expression_confidence'],
            eye_movement_confidence=result['eye_movement_confidence'],
            speech_confidence=result['speech_confidence'],
            hand_gesture_confidence=result['hand_gesture_confidence'],

            # save full result including AI feedback
            detailed_results=json.dumps({
                "video_analysis": result,
                "llm_analysis": llm_result,
                "final_confidence": final_confidence
            })
        )

        # 🟪 STEP 6 — Return both results to frontend
        return JsonResponse({
            "video_confidence_result": result,
            "llm_answer_evaluation": llm_result,
            "final_confidence": final_confidence,
            "expression_seconds": result.get("expression_seconds", 0)
        })

    return JsonResponse({"error": "Invalid request"}, status=400)


def transcribe_audio(video_path):
    try:
        audio_path = extract_audio(video_path)

        if not audio_path:
            print("❌ Audio extraction failed")
            return ""

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return text

    except Exception as e:
        print("❌ TRANSCRIPTION ERROR:", e)
        return ""


import ollama
import cv2
import numpy as np

# ------------------------------------------------------
# Placeholder Detection Functions (No ML Yet)
# Replace these later with your real video analysis logic
# ------------------------------------------------------

def detect_expression(video_path):
    # Dummy value until real model added
    return 70  

def detect_eye_movement(video_path):
    return 65

def detect_speech(video_path):
    return 60

def detect_hand_gesture(video_path):
    return 75
def detect_questions(text):
    if not text:
        return []
    import re
    questions = re.findall(r'[^.?!]*\?', text)
    return questions


def detect_filler_words(text):
    fillers = ["um", "uh", "er", "like", "you know"]
    found = [w for w in fillers if w in text.lower()]
    return found

def analyze_video(video_path):
    try:
        # 1️⃣ Extract audio
        audio_path = extract_audio(video_path)

        if not audio_path or not os.path.exists(audio_path):
            print("❌ Audio extraction failed")
            transcription = ""
        else:
            # 2️⃣ Convert audio → text
            transcription = transcribe_audio(audio_path)

        # Debug
        print("TRANSCRIPTION:", transcription)

        # 3️⃣ Detect questions & filler words
        questions_detected = detect_questions(transcription)
        filler_words = detect_filler_words(transcription)

        print("QUESTIONS DETECTED:", questions_detected)
        print("FILLER WORDS:", filler_words)

        # 4️⃣ Evaluate with LLaMA (gemma:2b)
        try:
            response = ollama.chat(
                model="gemma:2b",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
You are an AI interview evaluator.

Candidate transcript:
\"\"\"{transcription}\"\"\"

Identify:
1. Is the answer meaningful? (yes/no)
2. Give max 2–3 sentence feedback.
3. Score answer quality from 0–100.
4. Deduct score if filler words found: {filler_words}
5. Deduct score if answer is empty or irrelevant.

Return JSON ONLY in this exact format:

{{
  "confidence_score": number,
  "content_score": number,
  "overall_score": number,
  "feedback": "...",
  "mistakes": ["...", "..."]
}}
"""
                    }
                ],
            )
        except Exception as e:
            print("❌ LLM ERROR:", e)
            return {"error": "llm_failed", "details": str(e)}

        llm_text = response["message"]["content"]
        print("LLM RAW OUTPUT:", llm_text)

        # 5️⃣ Parse LLM JSON safely
        import json
        try:
            llm_json = json.loads(llm_text)
        except:
            llm_json = {
                "confidence_score": 0,
                "content_score": 0,
                "overall_score": 0,
                "feedback": "LLM returned invalid JSON.",
                "mistakes": []
            }

        # 6️⃣ Default video analysis (your placeholders)
        video_scores = {
            "overall_confidence": 80,
            "expression_confidence": 70,
            "eye_movement_confidence": 75,
            "speech_confidence": 85,
            "hand_gesture_confidence": 60,
            "expression_seconds": 5,
        }

        # 7️⃣ Combine both scores
        final_score = int((video_scores["overall_confidence"] + llm_json["overall_score"]) / 2)

        # 8️⃣ Return everything
        return {
            **video_scores,
            "transcript": transcription,
            "questions_detected": questions_detected,
            "filler_words": filler_words,
            "llm_analysis": llm_json,
            "final_score": final_score,
        }

    except Exception as e:
        print("❌ FULL ANALYSIS ERROR:", e)
        return {"error": "processing_failed", "details": str(e)}


def extract_audio(video_path):
    try:
        audio_path = f"{video_path}_audio.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ac", "1",
            "-ar", "16000",
            audio_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Debug logs
        print("FFMPEG STDERR:", result.stderr.decode())

        # Check if file really exists
        if not os.path.exists(audio_path):
            print("❌ AUDIO FILE NOT CREATED")
            return None

        return audio_path

    except Exception as e:
        print("❌ AUDIO EXTRACTION ERROR:", e)
        return None



"""def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) # Use cap.get directly
    video_duration = total_frames / fps if fps > 0 else 1  # Use fps for accurate duration

    start_time = datetime.now()
    print(f"Starting processing. Current Time: {format_time(start_time)}")

    frame_count = 0
    expression_counts = {} # Frame counts (used temporarily for calculation)
    eyes_down_count = 0
    eyes_forward_count = 0
    no_hand_movement_count = 0
    hand_movement_count = 0

    face_mesh = mp_face_mesh.FaceMesh()
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        # ... (rest of the frame processing loop remains the same) ...
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Facial Expression Detection
        try:
            analysis = DeepFace.analyze(rgb_frame, actions=["emotion"], enforce_detection=False)
            if analysis:
                expression = analysis[0]["dominant_emotion"]
                expression_counts[expression] = expression_counts.get(expression, 0) + 1
        except Exception as e:
            pass # Keep silent error to prevent massive logs

        # Eye Movement Detection
        # ... (Eye movement and Hand gesture code remains the same) ...

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
    # ... (print statements remain the same) ...

    # 1. Convert Expression Frames to Seconds
    expression_seconds = {}
    for expr, count in expression_counts.items():
        if fps > 0:
            expression_seconds[expr] = round(count / fps, 2)
        else:
            expression_seconds[expr] = 0.0

    # **Fetching predefined confidence values from the database**
    # ... (existing code remains the same) ...
    expression_data = {e.name.lower(): e.percentage for e in Expression.objects.all()}
    eye_data = {e.side.lower(): e.percentage for e in Eyes.objects.all()}
    hand_data = {h.move.lower(): h.percentage for h in HandsExpression.objects.all()}

    # **2. Calculating confidence scores using seconds**
    expression_confidence = 0
    for expr, seconds in expression_seconds.items(): # Iterate over seconds
        expr_weight = expression_data.get(expr.lower(), 0)  
        # Calculate time fraction based on seconds / total video duration
        time_fraction = seconds / video_duration if video_duration else 0
        expression_confidence += (expr_weight * time_fraction)
        
    # Eye Movement confidence calculation remains based on frames/total frames, 
    # as the confidence is often based on the proportion of time spent looking down vs. total time.
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
    # ... (existing weighted average calculation remains the same) ...
    overall_confidence = (
        (CONF_WEIGHTS["expression"] * expression_confidence) +
        (CONF_WEIGHTS["eye_movement"] * eye_movement_confidence) +
        (CONF_WEIGHTS["speech"] * speech_confidence) +
        (CONF_WEIGHTS["gesture"] * hand_gesture_confidence)
    )

    # 3. Return expression seconds instead of frame counts
    return {
        "expression_seconds": expression_seconds,
        "expression_confidence": round(expression_confidence, 2),
        "eye_movement_confidence": round(eye_movement_confidence, 2),
        "speech_confidence": round(speech_confidence, 2),
        "hand_gesture_confidence": round(hand_gesture_confidence, 2),
        "overall_confidence": round(overall_confidence, 2),
    }
    return {
        "expression_seconds": {"happy": 10, "sad": 20},
        "expression_confidence": round(83.2, 2),
        "eye_movement_confidence": round(70.5, 2),
        "speech_confidence": round(20.6, 2),
        "hand_gesture_confidence": round(65.45, 2),
        "overall_confidence": round(70.4, 2),
    }"""


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
        return 100


# ----------------------------------------------------------
# Admin Dashboard Views (Keep the original logic)
# ----------------------------------------------------------
@admin_required
def admin_dashboard(request):
    """Renders the main admin dashboard without backend search filtering."""
    
    # Fetch all candidates and annotate with average confidence
    candidate_list_query = User.objects.filter(role='candidate').annotate(
        avg_confidence=Avg('analysisresult__overall_confidence')
    ).order_by('-avg_confidence')

    # Note: We ignore request.GET.get('q') here
        
    total_candidates = candidate_list_query.count()
    
    context = {
        'total_candidates': total_candidates,
        'candidate_list': candidate_list_query,
        # 'search_query' is no longer needed but kept for safety if other elements rely on it
    }
    return render(request, 'admin_dashboard.html', context)


@admin_required
def create_candidate_page(request):
    return render(request, 'create_candidate.html')


@admin_required
def candidate_performance(request, user_id):
    candidate = get_object_or_404(User, id=user_id, role='candidate')
    analysis_results = candidate.analysisresult_set.all().order_by('-date_time')
    
    context = {
        'candidate': candidate,
        'analysis_results': analysis_results,
    }
    return render(request, 'candidate_performance.html', context)

@admin_required
def render_add_admin(request):
    return render(request, 'add_admin.html')

@csrf_exempt
@admin_required
def add_admin_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name')
            email = data.get('email')
            password = data.get('password')
            role = 'admin'
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'message': 'Invalid JSON data.'}, status=400)

        if not all([name, email, password]):
            return JsonResponse({'success': False, 'message': 'All fields are required.'})
        
        if User.objects.filter(email=email).exists():
            return JsonResponse({'success': False, 'message': 'Email already registered.'})

        try:
            user = User.objects.create(name=name, email=email, role=role)
            user.set_password(password)
            user.save()

            return JsonResponse({
                'success': True, 
                'message': f'Admin user {email} successfully created. Redirecting to dashboard.',
                'redirect_url': '/admin-dashboard' 
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'An error occurred: {str(e)}'})

    return JsonResponse({'success': False, 'message': 'Invalid request method.'}, status=405)

def candidate_history(request):
    """
    Shows all past analysis results for the currently logged-in candidate.
    Denies access to admins and unauthorized users.
    """
    user_id = request.session.get('user_id')
    user_role = request.session.get('user_role')

    # 1. Check if logged in
    if not user_id:
        messages.error(request, "Please log in to view your history.")
        return redirect('auth')

    # 2. Deny Admins
    if user_role == 'admin':
        messages.error(request, "Administrators cannot access the Candidate History page.")
        return redirect('admin_dashboard')

    try:
        candidate = User.objects.get(id=user_id)
        # Fetch all analysis results ordered by date (newest first)
        analysis_results = AnalysisResult.objects.filter(user=candidate).order_by('-date_time')
    except User.DoesNotExist:
        messages.error(request, "User profile error. Please log in again.")
        return redirect('logout')

    context = {
        'candidate': candidate,
        'analysis_results': analysis_results,
    }
    return render(request, 'history.html', context)

def get_candidate_report_data(user):
    """Aggregates and formats performance data for reporting."""
    results = user.analysisresult_set.all().order_by('-date_time')
    
    if not results:
        return {'summary': None, 'history_data': []}

    # 1. Summary Metrics (Average across all runs)
    summary = results.aggregate(
        avg_overall=Avg('overall_confidence'),
        avg_expression=Avg('expression_confidence'),
        avg_eye=Avg('eye_movement_confidence'),
        avg_speech=Avg('speech_confidence'),
        avg_gesture=Avg('hand_gesture_confidence'),
        total_runs=Count('id')
    )

    # 2. Detailed History for Charting (Last 5 runs)
    history = []
    for r in results[:5]:
        history.append({
            'date': r.date_time.strftime("%b %d %H:%M"),
            'overall': r.overall_confidence,
            'expression': r.expression_confidence,
            'speech': r.speech_confidence,
            'gesture': r.hand_gesture_confidence
        })

    return {
        'summary': {k: round(v, 2) if v is not None else 0 for k, v in summary.items()},
        'history_data': history[::-1], 
        'total_expressions': {
            'neutral': 240, 
            'happy': 45, 
            'sad': 15
        }
    }


def report_generator(request, user_id=None):
    """Handles logic for displaying reports based on user role."""
    current_user_id = request.session.get('user_id')
    current_user_role = request.session.get('user_role')

    if not current_user_id:
        messages.error(request, "Please log in to view reports.")
        return redirect('auth')

    # Determine the target user's ID
    if current_user_role == 'admin':
        # Admin can choose a user via URL or default to their own report
        target_id = user_id if user_id is not None else current_user_id
    else:
        # Candidate can only view their own report
        target_id = current_user_id
        if user_id is not None and user_id != current_user_id:
            messages.error(request, "Access denied. You can only view your own report.")
            return redirect('report_generator_self')

    try:
        target_user = User.objects.get(id=target_id)
        
        # Prevent generating a performance report for an admin user (except the current one)
        if target_user.role == 'admin' and target_user.id != current_user_id:
             messages.error(request, "Cannot generate a performance report for an administrator.")
             return redirect('admin_dashboard')

    except User.DoesNotExist:
        messages.error(request, "User not found.")
        return redirect('admin_dashboard' if current_user_role == 'admin' else 'home')


    # Generate Report Data
    report_data = get_candidate_report_data(target_user)

    context = {
        'target_user': target_user,
        'report': report_data,
        'is_admin': current_user_role == 'admin',
    }

    return render(request, 'reports.html', context)

@admin_required
def admin_upload_candidate_video(request, user_id):
    """
    Handles rendering the upload page (GET) and processing the video (POST) 
    for a specific candidate, saving results to their profile.
    """
    try:
        candidate = User.objects.get(id=user_id, role='candidate')
    except User.DoesNotExist:
        messages.error(request, "Candidate not found or invalid user role.")
        return redirect('admin_dashboard')

    if request.method == "POST":
        if request.FILES.get("video"):
            video_file = request.FILES["video"]
            
            # Save the file
            file_path = default_storage.save("backend/videos/" + video_file.name, ContentFile(video_file.read()))

            # Analyze the video (uses the mocked/actual analyze_video function)
            analysis_data = analyze_video(file_path) 
            
            # Save the result to the database (Assuming you have an AnalysisResult model)
            # You will need to import AnalysisResult and json if not already done.
            # from .models import AnalysisResult 

            # Assuming AnalysisResult has these fields:
            # user (ForeignKey), overall_confidence (float), date_time (datetime), 
            # detailed_results (JSONField/CharField)
            
            AnalysisResult.objects.create(
                user=candidate,
                overall_confidence=analysis_data['overall_confidence'],
                date_time=datetime.now(),
                expression_confidence=analysis_data['expression_confidence'],
                eye_movement_confidence=analysis_data['eye_movement_confidence'],
                speech_confidence=analysis_data['speech_confidence'],
                hand_gesture_confidence=analysis_data['hand_gesture_confidence'],
                # Store detailed expression data in a JSON/Text field
                detailed_results=json.dumps(analysis_data)
            )
            
            # Redirect to the candidate's performance page after successful upload
            messages.success(request, f"Video uploaded and analyzed successfully for {candidate.name}.")
            return redirect('candidate_performance', user_id=user_id)
        
        # If POST request but no file
        messages.error(request, "No video file provided.")
        
    # Render the upload page (GET request)
    context = {
        'candidate': candidate,
        'user_id': user_id,
        'is_admin': True,
    }
    return render(request, 'admin_upload_video.html', context)