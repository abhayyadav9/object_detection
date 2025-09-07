from fastapi import FastAPI

app = FastAPI(
    title="Real-time Object Detection Backend",
    description="A backend service for real-time video streaming, object detection, and self-training.",
    version="1.0.0",
)

# Import and include routers for different functionalities
from .routes import stream, logs, retrain, status, detect
app.include_router(stream.router)
app.include_router(logs.router)
app.include_router(retrain.router)
app.include_router(status.router)
app.include_router(detect.router)

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint providing a welcome message for the API."""
    return {"message": "Welcome to the Real-time Object Detection Backend!"}
