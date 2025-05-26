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
from mtcnn import MTCNN
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import StreamingHttpResponse, JsonResponse

embed = FaceNet()
detector = MTCNN()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


camera_active = False
captured_embedding = None


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None

        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected - Click Capture",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(faces) == 0:
            cv2.putText(frame, "Position your face in the camera",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


def gen_frames():
    camera = VideoCamera()
    while camera_active:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_stream(request):
    global camera_active
    camera_active = True
    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


@csrf_exempt
@require_http_methods(["POST"])
def capture_face_embedding(request):
    """Capture face and generate embedding for registration"""
    global captured_embedding

    try:
        # Use your existing get_embedding logic but simplified
        camera = cv2.VideoCapture(0)
        embeddings = None
        max_attempts = 30
        attempts = 0
        best_face_area = 0

        while attempts < max_attempts and embeddings is None:
            ret, frame = camera.read()
            if not ret:
                break

            attempts += 1

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using HOG+SVM
            faces = detector.detect_faces(frame)

            # Process the largest face detected
            for face in faces:
                x, y, w, h = face['box']
                face_area = w * h

                if face_area > best_face_area:
                    try:
                        face_region = frame[y:y + h, x:x + w]
                        if face_region.size > 0 and w > 50 and h > 50:
                            face_embedding = embed.embeddings([face_region])[0]
                            embeddings = face_embedding
                            best_face_area = face_area
                            break
                    except Exception as e:
                        print(f"Error processing face region: {e}")
                        continue
            if embeddings is not None:
                break

            import time
            time.sleep(0.05)

        camera.release()

        if embeddings is not None:
            captured_embedding = embeddings
            return JsonResponse({
                'success': True,
                'message': 'Face captured successfully!',
                'embedding_length': len(embeddings)
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'No suitable face detected. Please try again with better lighting.'
            })

    except Exception as e:
        return JsonResponse({'success': False, 'message': f'Error: {str(e)}'})


@csrf_exempt
def stop_camera(request):
    """Stop the camera stream"""
    global camera_active
    camera_active = False
    return JsonResponse({'success': True, 'message': 'Camera stopped'})


def register_student(request):
    if request.method == "POST":
        if request.POST['password'] == request.POST['confirm']:
            try:
                Student.objects.get(matric_number=request.POST["matric_number"])
                return render(request, 'attendance/home.html',
                              {'error': "Your matric number is already registered in the system"})
            except Student.DoesNotExist:
                password = request.POST["password"]
                encrypted = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                request.session['data'] = {
                    "matric_number": request.POST["matric_number"],
                    "name": request.POST["name"],
                    "email": request.POST["email"],
                    "password": encrypted
                }
                return redirect('attendance:scan_face')
        return render(request, 'attendance/home.html', {"error": "Passwords do not match"})
    return render(request, 'attendance/home.html')


def scan_face(request):
    """Display the face scanning page for registration"""
    if 'data' not in request.session:
        return redirect('attendance:register')

    return render(request, 'attendance/scan_face.html', {
        'student_name': request.session['data']['name'],
        'matric_number': request.session['data']['matric_number']
    })


@csrf_exempt
def complete_registration(request):
    """Complete student registration after face capture"""
    global captured_embedding, camera_active

    if request.method == "POST":
        if 'data' not in request.session:
            return JsonResponse({'success': False, 'message': 'Session expired. Please start registration again.'})

        if captured_embedding is None:
            return JsonResponse(
                {'success': False, 'message': 'No face embedding captured. Please capture your face first.'})

        try:
            # Create student with the captured embedding
            student_data = request.session['data']

            student = Student.objects.create(
                matric_number=student_data['matric_number'],
                name=student_data['name'],
                email=student_data['email'],
                password=student_data['password'],
                face_embedding=captured_embedding.tolist()  # Convert numpy array to list for storage
            )

            # Clean up
            camera_active = False
            captured_embedding = None
            del request.session['data']

            return JsonResponse({
                'success': True,
                'message': f'Registration completed successfully for {student.name}!',
                'redirect_url': '/attendance/'  # Adjust to your home URL
            })

        except Exception as e:
            return JsonResponse({'success': False, 'message': f'Registration failed: {str(e)}'})

    return JsonResponse({'success': False, 'message': 'Invalid request method'})


def get_embedding():
    """Legacy function - now handled by the web interface"""
    global captured_embedding
    if captured_embedding is not None:
        embedding = captured_embedding
        captured_embedding = None  # Reset after use
        return embedding
    return None
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
    return render(request, 'attendance/student_main.html', {"courses": courses, "student": student})


def login_student(request):
    if request.method == "POST":
        try:
            student = Student.objects.get(email=request.POST['email'])
        except Student.DoesNotExist:
            try:
                student = Student.objects.get(matric_number=request.POST['email'])
            except Student.DoesNotExist:
                return render(request, 'attendance/student_login.html', {"error": "Invalid login id"})
        password = request.POST['password']
        verify = bcrypt.checkpw(password.encode('utf-8'), student.password.encode('utf-8'))
        if verify:
            request.session['student_id'] = student.id
            return redirect('attendance:student_page')
        else:
            return render(request, 'attendance/student_login.html', {"error": "Incorrect password"})
    return render(request, 'attendance/student_login.html')


def login_lecturer(request):
    if request.method == "POST":
        try:
            lecturer = Lecturer.objects.get(email=request.POST['email'])
        except Lecturer.DoesNotExist:
            return render(request, 'attendance/lecturer_login.html', {"error": "Invalid login id"})
        password = request.POST['password']
        verify = bcrypt.checkpw(password.encode('utf-8'), lecturer.password.encode('utf-8'))
        if verify:
            request.session['lecturer_id'] = lecturer.id
            courses = lecturer.courses.all()
            return redirect('attendance:lecturer_page')
        else:
            return render(request, 'attendance/lecturer_login.html', {"error": "Incorrect password"})
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
    if request.method == "POST":
        form = CourseRegistrationForm(request.POST)
        if form.is_valid():
            selected_courses = form.cleaned_data['courses']
            student.courses.set(selected_courses)
            student.save()
            return redirect('attendance:student_page')
        else:
            form = CourseRegistrationForm(initial={"courses": student.courses.all()})
    else:
        return render(request, "attendance/course_registration.html", {"form": form})


def logout(request):
    request.session.flush()
    return redirect('attendance:student_page')


attendance_camera_active = False
current_session = None
marked_students = set()
enrolled_students_list = []


class AttendanceCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global marked_students, enrolled_students_list

        ret, frame = self.video.read()
        if not ret:
            return None

        # Detect faces using MTCNN (works on color image, not grayscale)
        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']

            # Ensure bounding box is within frame boundaries
            x, y = max(0, x), max(0, y)
            face_region = frame[y:y + h, x:x + w]

            try:
                face_region = cv2.resize(face_region, (160, 160))  # Resize for embedding model
                detected_embedding = embed.embeddings([face_region])[0]
                best_match = None
                highest_similarity = 0

                # Compare with enrolled students
                for student in enrolled_students_list:
                    if student.face_embedding is None:
                        continue

                    stored_embedding = np.array(student.face_embedding)
                    similarity = cosine_similarity(detected_embedding, stored_embedding)

                    if similarity >= 0.65 and similarity > highest_similarity:
                        best_match = student
                        highest_similarity = similarity

                # Mark attendance if match found and not already marked
                if best_match and best_match.id not in marked_students:
                    marked_students.add(best_match.id)

                    attendance, created = Attendance.objects.get_or_create(
                        student=best_match,
                        course=current_session.course,
                        session=current_session,
                        defaults={"status": True}
                    )

                    if not created and not attendance.status:
                        attendance.status = True
                        attendance.save()
                    print(f"Marked: {best_match.name} (Similarity: {highest_similarity:.2f})")

                # Draw bounding box and label
                if best_match:
                    if best_match.id in marked_students:
                        color = (0, 255, 0)
                        label = f"{best_match.name} - MARKED ({highest_similarity:.2f})"
                    else:
                        color = (0, 255, 255)
                        label = f"{best_match.name} ({highest_similarity:.2f})"
                else:
                    color = (0, 0, 255)
                    label = "Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                print(f"Error processing face: {e}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Add session info overlay
        cv2.putText(frame, f"Attendance Session Active", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Students Marked: {len(marked_students)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


def gen_attendance_frames():
    camera = AttendanceCamera()
    while attendance_camera_active:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def attendance_video_stream(request):
    global attendance_camera_active
    attendance_camera_active = True
    return StreamingHttpResponse(gen_attendance_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


@lecturer_login_required
def start_class_session(request, id):
    if request.method == "POST":
        course = get_object_or_404(Course, pk=id)
        lecturer = Lecturer.objects.get(pk=request.session['lecturer_id'])
        enrolled_students = course.students.all()

        if not course.lecturers.filter(id=lecturer.id).exists():
            return redirect('attendance:realmain')  # Redirect if unauthorized

        # Start the attendance session
        return start_attendance_session(request, course, lecturer, enrolled_students)

    return redirect('attendance:realmain')


def start_attendance_session(request, course, lecturer, enrolled_students):
    global current_session, marked_students, enrolled_students_list

    # Create class session
    current_session = ClassSession.objects.create(course=course, lecturer=lecturer)
    marked_students = set()
    enrolled_students_list = list(enrolled_students)
    print(enrolled_students_list)

    # Render the attendance page
    return render(request, 'attendance/take_attendance.html', {
        'course': course,
        'lecturer': lecturer,
        'enrolled_students': enrolled_students,
        'session_id': current_session.id
    })


@csrf_exempt
@require_http_methods(["POST"])
def stop_attendance_session(request):
    global attendance_camera_active, current_session, marked_students, enrolled_students_list

    attendance_camera_active = False
    print(marked_students)
    if current_session is None:
        return JsonResponse({'success': False, 'message': 'No active session'})

    # Mark absent students
    for student in enrolled_students_list:
        if student.id not in marked_students:
            attendance, created = Attendance.objects.get_or_create(
                student=student,
                course=current_session.course,
                session=current_session,
                defaults={"status": False}
            )

            if not created and attendance.status != False:
                attendance.status = False
                attendance.save()

    # Store session ID for redirect
    session_id = current_session.id

    # Reset global variables
    current_session = None
    marked_students = set()
    enrolled_students_list = []

    return JsonResponse({
        'success': True,
        'message': 'Attendance session completed',
        'redirect_url': f'/attendance/session/{session_id}'
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
            session=session
        ).first()
        if attendance_record is not None:
            status = 1 if attendance_record.status else 0
        else:
            status = 0
        # print (f"{student.name} - Attendance Record: {attendance_record} - Status: {getattr(attendance_record, 'status', 'N/A')}")
        summary.append({
            'name': student.name,
            'mat': student.matric_number,
            'status': status
        })
    print(summary)

    return render(request, "attendance/summary.html", {
        "course": session.course,
        "summary": summary,
        "date": session.date
    })


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
