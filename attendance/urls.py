from django.urls import path
from .views import *

app_name = "attendance"
urlpatterns = [
    path('register', register_student, name="register"),
    path('scan_face', scan_face, name="scan_face"),
    path('lecturer_register', register_lecturer, name='register_lecturer'),
    path('login_student', login_student, name="login_student"),
    path('login_lecturer', login_lecturer, name="login_lecturer"),
    path('course_registration', register_courses, name="course_registration"),
    path('start-session/<int:id>', start_class_session, name='start-session'),
    path('', student_page, name='student_page'),
    path('lecturer', lecturer_page, name='lecturer_page'),
    path('logout', logout, name='logout'),
    path('<int:course_id>/sessions/', view_course_sessions, name='view-sessions'),
    path('session/<int:session_id>', session_attendance_summary, name="session-summary"),
    path('session/<int:session_id>/download', download_summary, name="session-download"),
    path('summary', student_attendance_summary, name="attendance-summary")

]
