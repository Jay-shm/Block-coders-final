from django.urls import path
from . import views
from .views import mark_attendance

app_name = 'edu'

urlpatterns = [
    path('student/', views.student_dashboard, name = "student_dashboard"),
    path('teacher/', views.teacher_dashboard, name = "teacher_dashboard"),
    path('stu_add/', views.stu_add, name = "stu_add"),
    path('stu_details/', views.stu_details, name = "stu_details"),
    path('mark_attendance/', mark_attendance, name='mark_attendance'),
    path('add/', views.add, name = "add"),
]
