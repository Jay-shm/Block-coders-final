# face_recognition_api/README.md

# Face Recognition API

This project is a Face Recognition API built using FastAPI. It allows users to recognize faces from images and provides a WebSocket client for real-time testing.

## Project Structure

- **datasets/**: This directory stores images of different people and is auto-created during the execution of the application.
- **models/**: This directory stores trained face embeddings and is also auto-created.
- **utils/face_processing.py**: Contains helper functions for face detection and embeddings extraction.
- **websocket_client/**: Contains files for testing the application via WebSocket.
  - **client.py**: Python WebSocket client.
  - **client.html**: Web-based WebSocket client interface.
- **main.py**: The main FastAPI application.
- **requirements.txt**: Lists the dependencies required for installation.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To start the FastAPI application, run:

```
uvicorn main:app --reload
```

You can then access the API at `http://127.0.0.1:8000`.

## WebSocket Client

To test the application using the WebSocket client, open `client.html` in a web browser.