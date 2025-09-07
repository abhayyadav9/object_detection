const imageUpload = document.getElementById('imageUpload');
const detectButton = document.getElementById('detectButton');
const uploadedImage = document.getElementById('uploadedImage');
const detectionCanvas = document.getElementById('detectionCanvas');
const responseLog = document.getElementById('responseLog');
const ctx = detectionCanvas.getContext('2d');

const backendUrl = 'http://127.0.0.1:8000';

let currentImage = null;

imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage.src = e.target.result;
            uploadedImage.onload = () => {
                currentImage = uploadedImage;
                // Set canvas dimensions to match the image
                detectionCanvas.width = currentImage.width;
                detectionCanvas.height = currentImage.height;
                ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
                detectButton.disabled = false;
            };
        };
        reader.readAsDataURL(file);
    } else {
        uploadedImage.src = "#";
        detectButton.disabled = true;
        currentImage = null;
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        responseLog.textContent = '';
    }
});

detectButton.addEventListener('click', async () => {
    if (!currentImage || !imageUpload.files[0]) {
        alert("Please upload an image first.");
        return;
    }

    detectButton.disabled = true;
    responseLog.textContent = 'Detecting objects...\n';
    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height); // Clear previous drawings

    const formData = new FormData();
    formData.append('file', imageUpload.files[0]);

    try {
        const response = await fetch(`${backendUrl}/detect`, {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        responseLog.textContent = JSON.stringify(result, null, 2);

        if (response.ok) {
            drawDetections(result);
        } else {
            alert(`Error from backend: ${result.detail || response.statusText}`);
            logger.error('Backend Error:', result);
        }
    } catch (error) {
        console.error('Fetch Error:', error);
        responseLog.textContent += `\nError connecting to backend or processing request: ${error.message}`;
        alert(`Error: ${error.message}. Make sure the backend is running at ${backendUrl}`);
    } finally {
        detectButton.disabled = false;
    }
});

function drawDetections(detections) {
    if (!currentImage) return;

    // Ensure canvas matches the displayed image size
    const imgWidth = currentImage.naturalWidth;
    const imgHeight = currentImage.naturalHeight;
    const displayWidth = currentImage.width;
    const displayHeight = currentImage.height;

    detectionCanvas.width = displayWidth;
    detectionCanvas.height = displayHeight;

    const scaleX = displayWidth / imgWidth;
    const scaleY = displayHeight / imgHeight;

    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    ctx.lineWidth = 2;
    ctx.font = '18px Arial';

    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.box;
        const label = detection.class;
        const confidence = detection.confidence;

        // Scale coordinates to fit the displayed image on canvas
        const scaledX1 = x1 * scaleX;
        const scaledY1 = y1 * scaleY;
        const scaledX2 = x2 * scaleX;
        const scaledY2 = y2 * scaleY;

        const width = scaledX2 - scaledX1;
        const height = scaledY2 - scaledY1;

        ctx.strokeStyle = 'red';
        ctx.fillStyle = 'red';
        ctx.rect(scaledX1, scaledY1, width, height);
        ctx.stroke();

        ctx.fillText(`${label} (${(confidence * 100).toFixed(1)}%)`, scaledX1, scaledY1 > 10 ? scaledY1 - 5 : 15);
    });
}
