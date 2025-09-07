import argparse
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import os

# Configure logger for this script
logger.remove()
logger.add(os.sys.stderr, level="INFO")

def export_yolo_to_tensorrt(
    model_path: str,
    imgsz: int = 640,
    half: bool = True,
    device: str = "0" # GPU device ID
):
    """
    Exports a YOLOv8 model to ONNX and then to TensorRT engine.

    Args:
        model_path (str): Path to the YOLOv8 PyTorch (.pt) or ONNX (.onnx) model.
        imgsz (int): Image size for inference (e.g., 640 for 640x640).
        half (bool): Use half-precision (FP16) for TensorRT engine (requires compatible GPU).
        device (str): GPU device ID to use for export (e.g., "0" for the first GPU).
    """
    logger.info(f"Starting YOLOv8 model export to TensorRT for model: {model_path}")
    
    try:
        # Load the YOLO model
        model = YOLO(model_path)
        logger.info(f"YOLO model loaded from {model_path}")

        # Export to TensorRT. Ultralytics handles ONNX conversion internally.
        # This requires a CUDA-enabled environment with TensorRT installed.
        export_path = Path(model_path).parent / f"{Path(model_path).stem}.engine"
        
        logger.info(f"Attempting to export model to TensorRT engine. Output will be saved to {export_path.parent}")
        model.export(format="engine", imgsz=imgsz, half=half, device=device, verbose=False)
        
        # The export method in ultralytics typically saves the engine in the same directory
        # or a 'runs/detect/export' directory. We need to find the actual path.
        # A more robust check would involve listing files in the expected output directory.
        # For now, let's assume it saves next to the original model or in a standard 'runs' folder.
        # Ultralytics often creates a `runs/detect/export` directory.
        
        # Let's try to locate the exported engine file.
        # The actual path might be something like: runs/detect/export/yolov8n.engine
        # We need to find the correct path after export.
        
        # A simpler way is to check the `model.export()` return value or infer the path.
        # Ultralytics usually saves it to `model_path.replace('.pt', '.engine')` or in `runs/detect/export`

        # Let's assume the engine is saved in the `backend/app/models/yolo` directory
        # with the same base name but .engine extension
        expected_engine_path = Path(model_path).parent / f"{Path(model_path).stem}.engine"

        if expected_engine_path.exists():
            logger.success(f"Successfully exported model to TensorRT engine: {expected_engine_path}")
            return str(expected_engine_path)
        else:
            logger.warning(f"TensorRT engine not found at expected path: {expected_engine_path}. ")
            logger.warning("Ultralytics might have saved it in a different location (e.g., runs/detect/export/). Please check your output directories.")
            # Attempt to find in common ultralytics export locations
            runs_dir = Path("runs/detect")
            if runs_dir.exists():
                for file in runs_dir.rglob("*.engine"):
                    logger.success(f"Found TensorRT engine at: {file}")
                    return str(file)
            logger.error("Could not locate the exported TensorRT engine. Ensure TensorRT is correctly installed and configured.")
            return None

    except Exception as e:
        logger.error(f"Error during TensorRT export: {e}")
        logger.error("Please ensure you have a CUDA-enabled GPU and TensorRT installed and configured correctly in your environment.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to TensorRT engine.")
    parser.add_argument(
        "--model_path", 
        type=str,
        default="backend/app/models/yolo/yolov8n.pt",
        help="Path to the YOLOv8 PyTorch (.pt) or ONNX (.onnx) model."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference (e.g., 640 for 640x640)."
    )
    parser.add_argument(
        "--half",
        type=bool,
        default=True,
        help="Use half-precision (FP16) for TensorRT engine (requires compatible GPU)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0", # Default to first GPU
        help="GPU device ID to use for export (e.g., '0' for the first GPU)."
    )
    args = parser.parse_args()

    export_yolo_to_tensorrt(args.model_path, args.imgsz, args.half, args.device)
