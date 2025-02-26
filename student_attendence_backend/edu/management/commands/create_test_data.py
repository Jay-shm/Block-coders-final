from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from edu.models import StudentProfile, Subject

User = get_user_model()

class Command(BaseCommand):
    help = 'Creates test data for attendance system'

    def handle(self, *args, **kwargs):
        # Create test subject
        subject, created = Subject.objects.get_or_create(
            code='CS101',
            defaults={
                'name': 'Computer Science 101',
                'total_hours': 40
            }
        )
        self.stdout.write(f'{"Created" if created else "Found"} subject: {subject.name}')

        # Create test student
        test_email = 'test.student@example.com'
        test_roll_no = 'TEST001'

        user, created = User.objects.get_or_create(
            email=test_email,
            defaults={
                'full_name': 'Test Student',
                'profession': 'student',
            }
        )
        if created:
            user.set_password('test123')  # Set a password for login testing
            user.save()

        student, created = StudentProfile.objects.get_or_create(
            user=user,
            defaults={
                'roll_no': test_roll_no,
                'date_of_birth': '2000-01-01',
                'contact_details': '1234567890',
                'address': 'Test Address'
            }
        )
        self.stdout.write(f'{"Created" if created else "Found"} student: {student.user.full_name}')
