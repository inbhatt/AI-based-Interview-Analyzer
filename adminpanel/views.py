from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .models import Candidate, InterviewReport, AdminUser
from .forms import AdminLoginForm, CandidateForm

# --- Admin login view ---
def admin_login(request):
    if request.method == "POST":
        form = AdminLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            admin_user = authenticate(request, username=email, password=password)
            if admin_user and admin_user.is_admin:
                login(request, admin_user)
                return redirect('admin_dashboard')
            else:
                messages.error(request, "Invalid email or password.")
    else:
        form = AdminLoginForm()
    return render(request, 'adminpanel/login.html', {'form': form})

def admin_logout(request):
    logout(request)
    return redirect('admin_login')

# --- Dashboard ---
@login_required
def admin_dashboard(request):
    candidates = Candidate.objects.all()
    avg_confidence = (
        sum(c.overall_confidence for c in candidates) / len(candidates)
        if candidates else 0
    )
    return render(
        request,
        'adminpanel/dashboard.html',
        {'candidates': candidates, 'avg_confidence': avg_confidence},
    )

# --- Candidate views ---
@login_required
def candidate_list(request):
    candidates = Candidate.objects.all()
    return render(request, 'adminpanel/candidate_list.html', {'candidates': candidates})

@login_required
def candidate_detail(request, candidate_id):
    candidate = get_object_or_404(Candidate, id=candidate_id)
    reports = candidate.reports.all()
    return render(request, 'adminpanel/candidate_detail.html', {'candidate': candidate, 'reports': reports})

@login_required
def add_candidate(request):
    if request.method == "POST":
        form = CandidateForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Candidate added successfully!")
            return redirect('candidate_list')
    else:
        form = CandidateForm()
    return render(request, 'adminpanel/add_candidate.html', {'form': form})

@login_required
def live_record(request):
    return render(request, 'adminpanel/live_record.html')
