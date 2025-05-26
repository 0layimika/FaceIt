from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

class UserManager(BaseUserManager):
    def create_user(self, matric_number, password=None, **extra_fields):
        if not matric_number:
            raise ValueError("The Matriculation Number is required.")
        user = self.model(matric_number=matric_number, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

class Student(AbstractBaseUser):
    matric_number = models.CharField(max_length=15, unique=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True, blank=True, null=True)
    face_embedding = models.JSONField(null=True, blank=True)
    courses = models.ManyToManyField('Course', related_name='students')
    password = models.CharField(max_length=256)
    USERNAME_FIELD = 'matric_number'
    REQUIRED_FIELDS = []

    objects = UserManager()

    def __str__(self):
        return self.name

class Lecturer(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)
    courses = models.ManyToManyField('Course', related_name='lecturers')

    def __str__(self):
        return self.name

class Course(models.Model):
    course_code = models.CharField(max_length=10, unique=True)
    course_name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.course_code


class ClassSession(models.Model):
    course = models.ForeignKey('Course', on_delete=models.CASCADE, related_name='sessions')
    lecturer = models.ForeignKey('Lecturer', on_delete=models.CASCADE, related_name='sessions')
    date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.course.course_code} - {self.date}"

class Attendance(models.Model):
    course = models.ForeignKey('Course', on_delete=models.CASCADE, related_name='attendance_records')
    student = models.ForeignKey('Student', on_delete=models.CASCADE, related_name='attendance_records')
    session = models.ForeignKey(ClassSession, on_delete=models.CASCADE, null=True)
    status = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.student.name} - {self.course.course_code}"

    class Meta:
        unique_together = ('student', 'session')