import requests
import json

# Configuration
BASE_URL = 'http://127.0.0.1:8000'  # Change this to match your server
ATTENDANCE_URL = f'{BASE_URL}/mark_attendance/'

def test_mark_attendance():
    # Test data
    attendance_data = {
        'UID': 'TEST001',  # This matches the roll_no we created
        'subject_code': 'CS101',
        'ispresent': True
    }

    # Send POST request
    try:
        response = requests.post(
            ATTENDANCE_URL, 
            json=attendance_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Print response
        print(f'Status Code: {response.status_code}')
        print('Response:')
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    test_mark_attendance()
