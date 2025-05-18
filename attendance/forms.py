from django import forms
from .models import Course

class CourseRegistrationForm(forms.Form):
    courses = forms.ModelMultipleChoiceField(
        queryset=Course.objects.all(),
        widget=forms.CheckboxSelectMultiple,
        required=True,
        label="select courses"
    )