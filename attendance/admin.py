from django.contrib import admin
from .models import *

admin.site.register(Student)
admin.site.register(Lecturer)
admin.site.register(Course)
admin.site.register(Attendance)
admin.site.register(ClassSession)
