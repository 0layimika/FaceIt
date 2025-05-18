from datetime import date, datetime

from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from openpyxl.styles import Font

from .models import *
from .forms import *
import bcrypt
from keras_facenet import FaceNet
import cv2
from .decorators import lecturer_login_required, student_login_required
import numpy as np
import openpyxl


embed = FaceNet()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def cosine_similarity(a,b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
def get_embedding():
    camera = cv2.VideoCapture(0)
    print("Press 'C' to capture and 'Q' to quit")
    embeddings = None
    face_detected = False  # Flag to ensure face is captured only once

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Convert frame to grayscale (necessary for HOG+SVM)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using HOG+SVM
        faces, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Capture face embedding if a face is detected and not already captured
            if not face_detected:
                face_region = frame[y:y + h, x:x + w]
                face_embedding = embed.embeddings([face_region])[0]
                embeddings = face_embedding
                face_detected = True  # Set the flag to true to prevent re-capturing the face
                print("Face captured")
                break  # Once the face is captured, exit the loop

        # Show the frame with the face bounding box around the detected face
        cv2.imshow("Register Face", frame)

        # Check for key press to capture the face or quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and embeddings is not None:
            break  # If a face is captured and embeddings extracted, break
        elif key == ord('q'):
            break  # Exit if 'Q' is pressed

    camera.release()
    cv2.destroyAllWindows()
    return embeddings

def register_student(request):
    if request.method == "POST":
        if request.POST['password'] == request.POST['confirm']:
            try:
                Student.objects.get(matric_number=request.POST["matric_number"])
                return render(request,'attendance/home.html',{'error':"Your matric number is already registered in the system"})
            except Student.DoesNotExist:
                password = request.POST["password"]
                encrypted = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                request.session['data'] = {
                    "matric_number":request.POST["matric_number"],
                    "name" : request.POST["name"],
                    "email" : request.POST["email"],
                    "password": encrypted
                }
                return redirect('attendance:scan_face')
        return render(request,'attendance/home.html',{"error":"Passwords do not match"})
    return render(request, 'attendance/home.html')

def scan_face(request):
    if request.method == "POST":
        student_data = request.session.get('data')
        if not student_data:
            return redirect('attendance:register')  # Ensure registration details exist in the session

        # Extract face embeddings
        embeddings = get_embedding()
        if embeddings is None:
            return render(request, 'attendance/scan_face.html', {"error": "Face not detected. Try again."})

        # Save student details and face embeddings in the database
        student = Student.objects.create(
            matric_number=student_data['matric_number'],
            name=student_data['name'],
            email=student_data['email'],
            password=student_data['password'],
            face_embedding=embeddings.tolist()  # Save embeddings as a list (convert to JSON if needed)
        )
        del request.session['data']
        request.session['student_id'] = student.id
        return redirect('attendance:student_page')
        return render(request, 'attendance/realmain.html')  # Redirect to success page or dashboard
    return render(request, 'attendance/scan_face.html')

def register_lecturer(request):
    if request.method == "POST":
        if request.POST['password'] == request.POST['confirm']:
            try:
                Lecturer.objects.get(email=request.POST['email'])
                return render(request, 'attendance/register_teacher.html',
                          {'error': "Your email is already registered in the system"})
            except Lecturer.DoesNotExist:
                password = request.POST["password"]
                encrypted = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                Lecturer.objects.create(name=request.POST['name'], email=request.POST['email'], password=encrypted)
                return render(request, 'attendance/realmain.html')
        return render(request, 'attendance/register_teacher.html',
                          {'error': "Passwords do nor match"})
    return render(request, 'attendance/register_teacher.html')

@student_login_required
def student_page(request):
    student = Student.objects.get(id=request.session['student_id'])
    courses = student.courses
    return render(request,'attendance/student_main.html',{"courses":courses,"student":student})

def login_student(request):
    if request.method == "POST":
        try:
            student = Student.objects.get(email=request.POST['email'])
        except Student.DoesNotExist:
            try:
                student = Student.objects.get(matric_number=request.POST['email'])
            except Student.DoesNotExist:
                return render(request,'attendance/student_login.html',{"error":"Invalid login id"})
        password = request.POST['password']
        verify = bcrypt.checkpw(password.encode('utf-8'),student.password.encode('utf-8'))
        if verify:
            request.session['student_id'] = student.id
            return redirect('attendance:student_page')
        else:
            return render(request,'attendance/student_login.html',{"error":"Incorrect password"})
    return render(request, 'attendance/student_login.html')

def login_lecturer(request):
    if request.method == "POST":
        try:
            lecturer = Lecturer.objects.get(email=request.POST['email'])
        except Lecturer.DoesNotExist:
            return render(request,'attendance/lecturer_login.html',{"error":"Invalid login id"})
        password = request.POST['password']
        verify = bcrypt.checkpw(password.encode('utf-8'),lecturer.password.encode('utf-8'))
        if verify:
            request.session['lecturer_id'] = lecturer.id
            courses = lecturer.courses.all()
            return redirect('attendance:lecturer_page')
        else:
            return render(request,'attendance/lecturer_login.html',{"error":"Incorrect password"})
    return render(request, 'attendance/lecturer_login.html')

@lecturer_login_required
def lecturer_page(request):
    lecturer = Lecturer.objects.get(id=request.session['lecturer_id'])
    courses = lecturer.courses.all
    return render(request, 'attendance/realmain.html', {"lecturer": lecturer, "courses": courses})


@student_login_required
def register_courses(request):
    student = Student.objects.get(id=request.session['student_id'])
    if not isinstance(student, Student):
        return redirect('attendance:login_student')
    form = CourseRegistrationForm(request.POST)
    if request.method== "POST":
        form = CourseRegistrationForm(request.POST)
        if form.is_valid():
            selected_courses = form.cleaned_data['courses']
            student.courses.set(selected_courses)
            student.save()
            return redirect('attendance:student_page')
        else:
            form = CourseRegistrationForm(initial={"courses": student.courses.all()})
    else:
        return render(request, "attendance/course_registration.html",{"form":form})

@lecturer_login_required
def start_class_session(request, id):
    if request.method == "POST":
        course = get_object_or_404(Course, pk=id)
        lecturer = Lecturer.objects.get(pk=request.session['lecturer_id'])
        enrolled_students = course.students.all()
        if not course.lecturers.filter(id=lecturer.id).exists():
            return redirect('attendance:realmain')  # Redirect if unauthorized
        attendance_summary = take_attendance(course, lecturer, enrolled_students)
        print(attendance_summary)
        return render(request, "attendance/summary.html", {
            "course":course,
            "summary":attendance_summary,
            "date":date.today()
        })
    return redirect('attendance:realmain')

def logout(request):
    request.session.flush()
    return redirect('attendance:student_page')

def take_attendance(course,lecturer, enrolled_students):
    session = ClassSession.objects.create(course=course, lecturer=lecturer)
    camera = cv2.VideoCapture(0)
    marked_students = set()


    print("Press 'Q' to quit attendance session.")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

        for (x, y, w, h) in faces:
            face_region = frame[y:y + h, x:x + w]
            face_region = cv2.resize(face_region, (160, 160))  # Ensure consistent size for embedding
            detected_embedding = embed.embeddings([face_region])[0]

            best_match = None
            highest_similarity = 0

            for student in enrolled_students:
                if student.face_embedding is None:
                    continue

                stored_embedding = np.array(student.face_embedding)
                similarity = cosine_similarity(detected_embedding, stored_embedding)

                if similarity > 0.7 and similarity > highest_similarity:
                    best_match = student
                    highest_similarity = similarity

            if best_match and best_match.id not in marked_students:
                marked_students.add(best_match.id)
                print(f"Matched: {best_match.name} (Similarity: {highest_similarity:.2f})")
                Attendance.objects.get_or_create(
                    student=best_match,
                    course=course,
                    date=date.today(),
                    defaults={"status": True}
                )

                cv2.putText(frame, f"{best_match.name} ({highest_similarity:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    summary = []
    for student in enrolled_students:
        attendance_record = Attendance.objects.filter(
            student=student,
            course=course,
            date=date.today()
        ).first()
        summary.append({
            'name': student.name,
            "mat":student.matric_number,
            'status': 1 if attendance_record and attendance_record.status else 0
        })

    return summary

@lecturer_login_required
def view_course_sessions(request, course_id):
    course = get_object_or_404(Course, pk=course_id)
    lecturer = Lecturer.objects.get(pk=request.session['lecturer_id'])

    if not course.lecturers.filter(id=lecturer.id).exists():
        return redirect('attendance:realmain')  # unauthorized

    query_date = request.GET.get('date')
    if query_date:
        try:
            filter_date = datetime.strptime(query_date, '%Y-%m-%d').date()
            sessions = course.sessions.filter(date=filter_date, lecturer=lecturer).order_by('-date')
        except ValueError:
            sessions = course.sessions.filter(lecturer=lecturer).order_by('-date')  # fallback
    else:
        sessions = course.sessions.filter(lecturer=lecturer).order_by('-date')

    return render(request, 'attendance/sessions.html', {
        'course': course,
        'sessions': sessions,
        'query_date': query_date or ''
    })


@lecturer_login_required
def session_attendance_summary(request, session_id):
    session = get_object_or_404(ClassSession, pk=session_id)
    lecturer = Lecturer.objects.get(pk=request.session['lecturer_id'])

    if session.lecturer != lecturer:
        return redirect('attendance:lecturer_page')

    # Get enrolled students for the course at that time
    enrolled_students = session.course.students.all()

    # Build summary list just like in live session
    summary = []
    for student in enrolled_students:
        attendance_record = Attendance.objects.filter(
            student=student,
            course=session.course,
            date=session.date
        ).first()
        summary.append({
            'name': student.name,
            'mat': student.matric_number,
            'status': 1 if attendance_record and attendance_record.status else 0
        })

    return render(request, "attendance/summary.html", {
        "course": session.course,
        "summary": summary,
        "date": session.date
    })


@lecturer_login_required
def download_summary(request, session_id):
    session = get_object_or_404(ClassSession, pk=session_id)
    lecturer = Lecturer.objects.get(pk=request.session['lecturer_id'])

    if session.lecturer != lecturer:
        return redirect('attendance:lecturer_page')

    # Get students and attendance
    students = session.course.students.all()
    attendance_data = []
    for student in students:
        record = Attendance.objects.filter(
            student=student,
            course=session.course,
            date=session.date
        ).first()
        attendance_data.append({
            'name': student.name,
            'matric': student.matric_number,
            'status': "Present" if record and record.status else "Absent"
        })

    # Create Excel workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Attendance"

    # Write headers
    headers = ["Name", "Matric Number", "Status"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    # Write data
    for entry in attendance_data:
        ws.append([entry['name'], entry['matric'], entry['status']])

    # Prepare response
    response = HttpResponse(content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    filename = f"{session.course.course_code}_{session.date}_attendance.xlsx"
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    wb.save(response)
    return response

@student_login_required
def student_attendance_summary(request):
    student = Student.objects.get(pk=request.session['student_id'])
    courses = student.courses.all()

    attendance_data = []

    for course in courses:
        sessions = ClassSession.objects.filter(course=course)
        total_classes = sessions.count()

        presents = Attendance.objects.filter(
            student=student,
            course=course,
            status=True
        ).count()

        absents = total_classes - presents
        percent = round((presents / total_classes) * 100, 2) if total_classes > 0 else 0

        attendance_data.append({
            "course_name": course.course_name,
            "course_code": course.course_code,
            "total_classes": total_classes,
            "present": presents,
            "absent": absents,
            "percent": percent,
        })

    return render(request, "attendance/student_summary.html", {
        "attendance_data": attendance_data,
        "student": student,
    })