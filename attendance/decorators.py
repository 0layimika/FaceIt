from functools import wraps
from django.shortcuts import redirect

def lecturer_login_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if 'lecturer_id' in request.session:
            return view_func(request, *args, **kwargs)
        return redirect('attendance:login_lecturer')
    return wrapper

def student_login_required(view_func):
    @wraps(view_func)
    def wrapper(request,*args,**kwargs):
        if 'student_id' in request.session:
            return view_func(request, *args, **kwargs)
        return redirect('attendance:login_student')
    return wrapper
