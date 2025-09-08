const webcamFeed = document.getElementById('webcamFeed');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const detectionCanvas = document.getElementById('detectionCanvas');
const responseLog = document.getElementById('responseLog');
const responseLog1 = document.getElementById('responseLog1');

// Store unique detected classes
const detectedClassesSet = new Set();

const ctx = detectionCanvas.getContext('2d');
const videoElement = document.createElement('video'); // Hidden video element for capturing frames

// Auto-detect backend URL for local network/mobile use
// By default, use the same host as the frontend, but allow override for local testing
let backendUrl = '';
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    backendUrl = 'http://127.0.0.1:8000'; // For local desktop testing
} else {
    // Use the same host as the frontend, but port 8000
    backendUrl = `${window.location.protocol}//${window.location.hostname}:8000`;
}
// To force a specific IP (e.g., for debugging), uncomment and set below:
// backendUrl = 'http://192.168.1.10:8000';
const detectionIntervalMs = 1000; // Send frame every 1 second

let mediaStream = null;
let captureIntervalId = null;

startButton.addEventListener('click', async () => {
    try {
        // Prefer back camera on mobile devices
        const constraints = {
            video: {
                facingMode: { ideal: 'environment' }
            }
        };
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamFeed.srcObject = mediaStream;
        webcamFeed.play();

        startButton.disabled = true;
        stopButton.disabled = false;
        responseLog.textContent = 'Webcam started. Waiting for video dimensions...\n';

        webcamFeed.addEventListener('loadedmetadata', () => {
            // Set canvas dimensions to match the video feed
            detectionCanvas.width = webcamFeed.videoWidth;
            detectionCanvas.height = webcamFeed.videoHeight;
            videoElement.width = webcamFeed.videoWidth;
            videoElement.height = webcamFeed.videoHeight;
            responseLog.textContent += `Video dimensions: ${webcamFeed.videoWidth}x${webcamFeed.videoHeight}\n`;
            startFrameCapture();
        }, { once: true });

    } catch (error) {
        console.error('Error accessing webcam:', error);
        responseLog.textContent = `Error: ${error.message}\n`;
        alert('Could not access the webcam. Please ensure it\'s connected and permissions are granted.');
    }
});

stopButton.addEventListener('click', () => {
    stopFrameCapture();
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        webcamFeed.srcObject = null;
    }
    startButton.disabled = false;
    stopButton.disabled = true;
    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    responseLog.textContent = 'Webcam stopped.\n';
});

function startFrameCapture() {
    if (captureIntervalId) {
        clearInterval(captureIntervalId);
    }
    captureIntervalId = setInterval(async () => {
        if (webcamFeed.readyState === webcamFeed.HAVE_ENOUGH_DATA) {
            // Draw the current video frame onto a hidden canvas (using videoElement's dimensions)
            const offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width = webcamFeed.videoWidth;
            offscreenCanvas.height = webcamFeed.videoHeight;
            const offscreenCtx = offscreenCanvas.getContext('2d');
            offscreenCtx.drawImage(webcamFeed, 0, 0, webcamFeed.videoWidth, webcamFeed.videoHeight);

            // Convert the canvas content to a Blob (image file format)
            offscreenCanvas.toBlob(async (blob) => {
                if (!blob) {
                    console.error("Failed to create blob from canvas.");
                    return;
                }
                const formData = new FormData();
                formData.append('file', blob, 'webcam_frame.jpg');

                try {
                    const response = await fetch(`${backendUrl}/detect`, {
                        method: 'POST',
                        body: formData,
                    });

                    const result = await response.json();

                    // Show detailed detection results in responseLog every interval
                    if (Array.isArray(result) && result.length > 0 && result[0].class) {
                        let logText = 'Detected Objects (' + new Date().toLocaleTimeString() + '):\n';
                        result.forEach((det, idx) => {
                            logText += `#${idx + 1}:\n`;
                            logText += `  Class: ${det.class}\n`;
                            logText += `  Confidence: ${(det.confidence * 100).toFixed(1)}%\n`;
                            logText += `  Box: [${det.box.join(', ')}]\n`;
                            // Only store high accuracy detections (confidence > 0.7)
                            if (det.confidence > 0.5) {
                                detectedClassesSet.add(det.class);
                            }
                        });
                        responseLog.textContent = logText;
                    } else {
                        responseLog.textContent = 'No objects detected.';
                    }

                    // Display the set of unique detected classes on screen
                    if (detectedClassesSet.size > 0) {
                        responseLog1.textContent = 'Unique objects detected so far:\n' + Array.from(detectedClassesSet).join(', ');
                    } else {
                        responseLog1.textContent = 'No unique objects detected yet.';
                    }

                    if (response.ok) {
                        drawDetections(result);
                    } else {
                        console.error('Backend Error:', result);
                        // alert(`Error from backend: ${result.detail || response.statusText}`);
                    }
                } catch (error) {
                    console.error('Fetch Error:', error);
                    responseLog.textContent += `\nError connecting to backend: ${error.message}`;
                    // alert(`Error: ${error.message}. Make sure the backend is running.`);
                }
            }, 'image/jpeg', 0.8); // 0.8 quality for JPEG
        }
    }, detectionIntervalMs);
}

function stopFrameCapture() {
    if (captureIntervalId) {
        clearInterval(captureIntervalId);
        captureIntervalId = null;
    }
}

function drawDetections(detections) {
    // Ensure canvas dimensions match the video feed
    detectionCanvas.width = webcamFeed.videoWidth;
    detectionCanvas.height = webcamFeed.videoHeight;

    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    ctx.lineWidth = 2;
    ctx.font = '18px Arial';

    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.box;
        const label = detection.class;
        const confidence = detection.confidence;

        ctx.strokeStyle = 'red';
        ctx.fillStyle = 'red';
        ctx.rect(x1, y1, x2 - x1, y2 - y1);
        ctx.stroke();

        ctx.fillText(`${label} (${(confidence * 100).toFixed(1)}%)`, x1, y1 > 10 ? y1 - 5 : 15);
    });
}
