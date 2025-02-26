# Block-coders
# Student Attendance System

A comprehensive web-based attendance management system with facial recognition capabilities built by Block-coders team.

## Project Structure

- **frontend/** - Web interface components
    - Student, teacher and employee dashboards
    - Login and registration pages 
    - Attendance management interfaces

- **Image_processing&recog/**
    - Face recognition API implementation
    - Real-time face detection and processing

- **server/** - Backend server implementation

- **student_attendence_backend/** - Student attendance specific backend logic

## Features

- Facial recognition based attendance tracking
- Multiple user roles (Students, Teachers, Employees)  
- Custom dashboards for different user types
- Real-time attendance monitoring
- User registration and authentication

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create `.env` file with:
```
DATABASE_URL=your_database_url
SECRET_KEY=your_secret_key
```

### 4. Running Components

Start Face Recognition API:
```bash
cd Image_processing&recog/face_recognition_api
uvicorn main:app --reload
```

Start Backend Server:
```bash
cd server
npm install
npm start
```

Open frontend `index.html` in browser

## Technologies Used

- Frontend: HTML, CSS
- Face Recognition: Python, FastAPI, WebSocket
- Backend: Node.js/Python
- ML Libraries: TensorFlow, OpenCV, scikit-learn

## Notes

- Always activate virtual environment before running
- Install Visual Studio Build Tools before dlib installation on Windows
- Keep requirements.txt updated when adding dependencies
- Never commit sensitive information or large files

## Contributing

1. Fork the repository
2. Create feature branch 
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License - see LICENSE file for details