const webcamFeed = document.getElementById('webcamFeed');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const detectionCanvas = document.getElementById('detectionCanvas');
const responseLog = document.getElementById('responseLog');
const responseLog1 = document.getElementById('responseLog1');

const ctx = detectionCanvas.getContext('2d');
const videoElement = document.createElement('video'); // Hidden video element for capturing frames

const backendUrl = 'http://127.0.0.1:8000';
const detectionIntervalMs = 200; // Send frame every 200ms

let mediaStream = null;
let captureIntervalId = null;

startButton.addEventListener('click', async () => {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
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
                                        responseLog1.textContent = result.data

                    responseLog.textContent = JSON.stringify(result, null, 2);

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
