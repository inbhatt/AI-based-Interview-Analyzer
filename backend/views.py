import os
import cv2
import numpy as np
import mediapipe as mp
import requests
import speech_recognition as sr
import nltk
import re
import json
import subprocess

from deepface import DeepFace
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from datetime import datetime
from django.contrib import messages
from .models import User, AnalysisResult
from django.db.models import Avg, Count

from .punctuation import restore_punctuation
from .qa_extractor import extract_qa_with_llama

# Load NLP data
nltk.download("punkt")
nltk.download("punkt_tab")

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Confidence Calculation Weights
CONF_WEIGHTS = {
    "expression": 0.30,
    "eye_movement": 0.20,
    "speech": 0.35,
    "gesture": 0.15
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
        file_path = default_storage.save("backend/videos/" + str(user_id) + "_" + timestamp_str + file_extension, ContentFile(video_file.read()))

        result = analyze_video(file_path)

        AnalysisResult.objects.create(
            user=user,
            video_path=file_path,
            # 🚨 SAVING ALL FIVE SCORES 🚨
            overall_confidence=result['overall_confidence'],
            expression_confidence=result['expression_confidence'],
            eye_movement_confidence=result['eye_movement_confidence'],
            speech_confidence=result['speech_confidence'],
            hand_gesture_confidence=result['hand_gesture_confidence'],
            speech_details=json.dumps(result.get("qa_analysis", [])),
            detailed_results=json.dumps(result) # Store the full analysis dictionary
        )

        return JsonResponse({
            "confidence_result": result,
            "expression_seconds": result["expression_seconds"] # Use new seconds field
        })

    return JsonResponse({"error": "Invalid request"}, status=400)


'''def analyze_video(video_path):
    """
    MOCKED FUNCTION: Returns random analysis results for the demo.
    """
    # 1. Generate random scores (between 55% and 95%)
    exp_conf = round(random.uniform(55.0, 95.0), 2)
    eye_conf = round(random.uniform(55.0, 95.0), 2)
    speech_conf = round(random.uniform(55.0, 95.0), 2)
    hand_conf = round(random.uniform(55.0, 95.0), 2)
    
    # 2. Calculate overall confidence using the weights (same as your original logic)
    CONF_WEIGHTS = {
        "expression": 0.4,
        "eye_movement": 0.2,
        "speech": 0.2,
        "gesture": 0.2
    }
    overall_confidence = (
        (CONF_WEIGHTS["expression"] * exp_conf) +
        (CONF_WEIGHTS["eye_movement"] * eye_conf) +
        (CONF_WEIGHTS["speech"] * speech_conf) +
        (CONF_WEIGHTS["gesture"] * hand_conf)
    )
    overall_confidence = round(overall_confidence, 2) # Normalize to 100
    
    # 3. Generate random expression seconds
    # Assuming video duration is around 30-60 seconds for a demo
    total_duration = random.randint(30, 60)
    
    # Randomly assign portions of the total duration to key expressions
    seconds_neutral = random.randint(int(total_duration * 0.4), int(total_duration * 0.7))
    seconds_happy = random.randint(5, 15)
    
    # Ensure sad/other is not too high
    seconds_sad = random.randint(0, 10) 
    
    # Adjust neutral to ensure total is approximately correct
    seconds_neutral = total_duration - seconds_happy - seconds_sad
    if seconds_neutral < 0:
        seconds_neutral = 5 # Failsafe
    
    expression_seconds = {
        "neutral": seconds_neutral,
        "happy": seconds_happy,
        "sad": seconds_sad,
        "surprise": random.randint(0, 5),
    }

    # Final scores object
    return {
        "expression_seconds": expression_seconds,
        "expression_confidence": exp_conf,
        "eye_movement_confidence": eye_conf,
        "speech_confidence": speech_conf,
        "hand_gesture_confidence": hand_conf,
        "overall_confidence": overall_confidence,
        "qa_analysis": []
    }'''
def format_time(dt):
    """Helper function to format datetime"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def analyze_speech_with_llama(video_path):
    """
    Extracts audio, restores punctuation, identifies Q&A pairs,
    and evaluates answers using Llama via Ollama.
    Returns overall speech confidence and full Q&A list.
    """
    recognizer = sr.Recognizer()
    audio_path = os.path.join(os.path.dirname(video_path), "audio.wav")

    # STEP 1 — Extract audio from video
    command = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y'
    process = subprocess.run(command, shell=True, capture_output=True, text=True)

    if process.returncode != 0 or not os.path.exists(audio_path):
        print("FFmpeg failed to extract audio.")
        return 70, []

    try:
        # STEP 2 — Transcribe speech
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            raw_text = recognizer.recognize_google(audio_data)

        print(f"🗣️ Recognized Speech (Raw): {raw_text}")

        # STEP 3 — Punctuate using multilingual model
        punctuated_text = restore_punctuation(raw_text)
        print(f"✍️ After Punctuation: {punctuated_text}")

        # STEP 4 — Run Llama to extract Q&A and score
        qa_results = extract_qa_with_llama(punctuated_text)

        # STEP 5 — Compute average confidence score
        if qa_results:
            avg_score = sum(q["score"] for q in qa_results) / len(qa_results)
        else:
            avg_score = 70  # Fallback

        # Clean up temp audio
        os.remove(audio_path)

        return round(avg_score, 2), qa_results

    except Exception as e:
        print(f"Speech analysis error: {e}")
        return 60, []


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
def live_recording_page(request, user_id):
    """
    Renders the live recording interface for a specific candidate.
    """
    try:
        candidate = User.objects.get(id=user_id, role='candidate')
    except User.DoesNotExist:
        messages.error(request, "Candidate not found.")
        return redirect('admin_dashboard')

    context = {
        'candidate': candidate,
        'user_id': user_id,
    }
    return render(request, 'live_recording.html', context)


@csrf_exempt
@admin_required
def save_live_recording(request, user_id):
    """
    Receives the recorded video blob from the frontend,
    saves it, analyzes it, and returns the results.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        candidate = User.objects.get(id=user_id, role='candidate')
    except User.DoesNotExist:
        return JsonResponse({"error": "Candidate not found."}, status=404)

    # Get the video file from request
    video_file = request.FILES.get('video')

    if not video_file:
        return JsonResponse({"error": "No video file provided."}, status=400)

    try:
        # Save the video file
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        file_name = f"live_recording_{user_id}_{timestamp_str}.webm"
        file_path = default_storage.save(f"backend/videos/{file_name}", ContentFile(video_file.read()))

        print(f"Live recording saved: {file_path}")

        # Analyze the video
        result = analyze_video(file_path)

        # Save analysis results to database
        AnalysisResult.objects.create(
            user=candidate,
            video_path=file_path,
            overall_confidence=result['overall_confidence'],
            expression_confidence=result['expression_confidence'],
            eye_movement_confidence=result['eye_movement_confidence'],
            speech_confidence=result['speech_confidence'],
            hand_gesture_confidence=result['hand_gesture_confidence'],
            speech_details=json.dumps(result.get("qa_analysis", [])),
            detailed_results=json.dumps(result)
        )

        return JsonResponse({
            "success": True,
            "message": "Recording analyzed successfully!",
            "confidence_result": result,
            "expression_seconds": result.get("expression_seconds", {}),
            "redirect_url": f"/candidate/{user_id}/performance"
        })

    except Exception as e:
        print(f"Error processing live recording: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            "error": f"An error occurred while processing the recording: {str(e)}"
        }, status=500)

@admin_required
def create_candidate_page(request):
    return render(request, 'create_candidate.html')


@admin_required
def candidate_performance(request, user_id):
    candidate = get_object_or_404(User, id=user_id, role='candidate')
    analysis_results = [
        {
            "obj": r,
            "local_time": timezone.localtime(r.date_time)
        }
        for r in candidate.analysisresult_set.all().order_by('-date_time')
    ]

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
                speech_details=json.dumps(analysis_data.get("qa_analysis", [])),
                detailed_results=json.dumps(analysis_data)
            )
            
            # Redirect to the candidate's performance page after successful upload
            messages.success(request, f"Video uploaded and analyzed successfully for {candidate.name}.")
            return redirect('candidate_performance.html', user_id=user_id)
        
        # If POST request but no file
        messages.error(request, "No video file provided.")
        
    # Render the upload page (GET request)
    context = {
        'candidate': candidate,
        'user_id': user_id,
        'is_admin': True,
    }
    return render(request, 'admin_upload_video.html', context)