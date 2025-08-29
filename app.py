import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- Basic Setup ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Mediapipe Face Mesh Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- In-Memory Leaderboard (for demo purposes) ---
# In a real app, you would use a database like SQLite or PostgreSQL.
leaderboard = [
    {"username": "GrinMaster", "score": 950},
    {"username": "GiggleQueen", "score": 820},
    {"username": "BeamBot", "score": 700},
]

# --- Core Smile Detection Logic ---
def calculate_smile_score(image):
    """
    Analyzes an image to detect a face and calculate a smile score.
    Returns a score from 0 to 100 based on the smile width relative to face width.
    """
    try:
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return 0  # No face detected

        landmarks = results.multi_face_landmarks[0].landmark
        
        # --- Landmark indices for smile calculation ---
        # Mouth corners
        left_mouth_corner = landmarks[61]
        right_mouth_corner = landmarks[291]
        
        # Points for face width normalization (e.g., cheekbones or jawline)
        left_face_edge = landmarks[234] 
        right_face_edge = landmarks[454]

        # Get pixel coordinates
        ih, iw, _ = image.shape
        left_mouth_px = (left_mouth_corner.x * iw, left_mouth_corner.y * ih)
        right_mouth_px = (right_mouth_corner.x * iw, right_mouth_corner.y * ih)
        left_face_px = (left_face_edge.x * iw, left_face_edge.y * ih)
        right_face_px = (right_face_edge.x * iw, right_face_edge.y * ih)

        # Calculate distances using Euclidean formula
        smile_width = np.linalg.norm(np.array(left_mouth_px) - np.array(right_mouth_px))
        face_width = np.linalg.norm(np.array(left_face_px) - np.array(right_face_px))
        
        # --- Scoring Logic ---
        # Normalize smile width by face width to make it independent of distance to camera
        smile_ratio = smile_width / face_width
        
        # Map the ratio to a score (e.g., 0-100). These thresholds are adjustable!
        # A neutral face has a ratio around 0.35-0.45. A big smile is > 0.5.
        min_ratio = 0.40  # Neutral face
        max_ratio = 0.60  # Big smile
        
        score = ((smile_ratio - min_ratio) / (max_ratio - min_ratio)) * 100
        score = max(0, min(100, score)) # Clamp score between 0 and 100

        return int(score)

    except Exception as e:
        print(f"Error in smile detection: {e}")
        return 0

# --- API Endpoints ---
@app.route('/')
def index():
    """Renders the main game page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_smile():
    """Receives a webcam frame, analyzes it, and returns the score."""
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    np_image = np.array(image)
    
    # Process the image to get the score
    score = calculate_smile_score(np_image)
    coins = score * 5 # Simple coin logic
    
    return jsonify({'score': score, 'coins': coins})

@app.route('/leaderboard', methods=['GET', 'POST'])
def handle_leaderboard():
    """Fetches or updates the leaderboard."""
    global leaderboard
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', 'Anonymous')
        score = data.get('score', 0)
        leaderboard.append({'username': username, 'score': score})
        # Sort by score (descending) and keep top 10
        leaderboard = sorted(leaderboard, key=lambda x: x['score'], reverse=True)[:10]
        return jsonify({'status': 'success'})
        
    # GET request
    return jsonify(sorted(leaderboard, key=lambda x: x['score'], reverse=True))

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)