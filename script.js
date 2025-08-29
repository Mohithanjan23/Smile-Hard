document.addEventListener('DOMContentLoaded', async () => {
    // --- DOM Elements ---
    const video = document.getElementById('webcam');
    const overlay = document.getElementById('overlay-canvas');
    const scoreDisplay = document.getElementById('score');
    const smileMeterBar = document.getElementById('smile-meter-bar');
    const saveBtn = document.getElementById('save-btn');

    // --- AI MODEL LOADING ---
    async function loadModels() {
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
        console.log("Loading AI models...");
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        ]);
        console.log("Models loaded!");
    }

    // --- WEBCAM & DETECTION ---
    async function startVideo() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
            video.srcObject = stream;
            video.addEventListener('play', () => {
                overlay.width = video.clientWidth;
                overlay.height = video.clientHeight;
                startDetection();
            });
        } catch (err) {
            console.error("Webcam access error:", err);
        }
    }

    function startDetection() {
        setInterval(async () => {
            const detections = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
            
            if (detections) {
                const score = calculateSmileScore(detections.landmarks.positions);
                updateUI(score);
            } else {
                updateUI(0);
            }
        }, 200); // Detect 5 times per second
    }

    // --- SMILE SCORE CALCULATION ---
    function calculateSmileScore(landmarks) {
        // Landmark indices for smile calculation
        const leftMouth = landmarks[48];
        const rightMouth = landmarks[54];
        const leftEye = landmarks[36];
        const rightEye = landmarks[45];

        const mouthWidth = Math.hypot(leftMouth.x - rightMouth.x, leftMouth.y - rightMouth.y);
        const eyeDistance = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);

        // Normalize smile width by the distance between the eyes
        const smileRatio = mouthWidth / eyeDistance;

        // Map the ratio to a score (0-100). These thresholds may need tweaking.
        const minRatio = 0.6; // Neutral face
        const maxRatio = 0.9; // Big smile
        
        let score = ((smileRatio - minRatio) / (maxRatio - minRatio)) * 100;
        score = Math.max(0, Math.min(100, score)); // Clamp score between 0-100
        
        return Math.floor(score);
    }

    // --- UI UPDATE ---
    function updateUI(score) {
        scoreDisplay.textContent = score;
        smileMeterBar.style.width = `${score}%`;
    }

    // --- SAVE SNAPSHOT ---
    saveBtn.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        
        // Draw mirrored webcam image
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform

        // --- Overlay Score ---
        const score = scoreDisplay.textContent;
        ctx.font = "48px 'Press Start 2P'";
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 6;
        
        const text = `SCORE: ${score}`;
        const textMetrics = ctx.measureText(text);
        
        // Position text at the bottom center
        const x = (canvas.width - textMetrics.width) / 2;
        const y = canvas.height - 40;

        ctx.strokeText(text, x, y);
        ctx.fillText(text, x, y);

        // Trigger download
        const link = document.createElement('a');
        link.download = `smile-hard-snapshot-${Date.now()}.png`;
        link.href = canvas.toDataURL();
        link.click();
    });

    // --- INITIALIZATION ---
    async function initialize() {
        await loadModels();
        await startVideo();
    }
    initialize();
});