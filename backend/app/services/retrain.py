import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from ultralytics import YOLO
import yaml
import cv2
from datetime import datetime

from backend.app.config import config
from backend.app.services.detection import detector # To use the detector for pseudo-labeling

class PseudoLabelGenerator:
    """
    Handles the generation of pseudo-labels from unlabeled data using the main object detector.
    Detections with confidence above a configured threshold are saved as YOLO-format labels.
    """
    def __init__(self):
        """
        Initializes the PseudoLabelGenerator, ensuring the output directory for pseudo-labels exists.
        """
        self.pseudo_labels_dir = Path(config.PSEUDO_LABELS_DIR)
        self.unlabeled_data_dir = Path(config.UNLABELED_DATA_DIR)
        self.pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PseudoLabelGenerator initialized. Output dir: {self.pseudo_labels_dir}")

    async def generate_pseudo_labels(self, frame: Any, frame_name: str) -> None:
        """
        Generates pseudo-labels for a given frame. If detections meet the confidence
        threshold, the frame and its corresponding YOLO-format labels are saved.

        Args:
            frame (Any): The input image frame to process for pseudo-labeling.
            frame_name (str): A unique name for the frame (used for saving image and label files).
        """
        # Use the main detector instance to perform object detection on the frame.
        detections = await detector.detect(frame)

        # Filter detections based on the pseudo-labeling confidence threshold.
        high_confidence_detections = [
            det for det in detections if det["confidence"] >= config.PSEUDO_LABEL_CONFIDENCE
        ]

        if high_confidence_detections:
            image_filename = self.pseudo_labels_dir / f"{frame_name}.jpg"
            label_filename = self.pseudo_labels_dir / f"{frame_name}.txt"
            
            # Save the original frame to the pseudo-labels directory.
            cv2.imwrite(str(image_filename), frame)

            # Generate and save labels in YOLO format (class x_center y_center width height).
            with open(label_filename, "w") as f:
                img_h, img_w, _ = frame.shape
                for det in high_confidence_detections:
                    x1, y1, x2, y2 = det["box"]
                    label_cls = detector.model.names.index(det["class"]) # Get integer class ID

                    # Convert absolute bounding box coordinates to normalized YOLO format.
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    f.write(f"{label_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            logger.info(f"Generated pseudo-labels for {frame_name} with {len(high_confidence_detections)} high confidence detections.")

class StudentModelTrainer:
    """
    Manages the retraining process for a student YOLOv8 model.
    It combines human-labeled data and pseudo-labeled data, prepares a dataset
    configuration, and initiates the training loop using Ultralytics YOLO.
    """
    def __init__(self):
        """
        Initializes the StudentModelTrainer with paths for labeled, pseudo-labeled data,
        and the output directory for the retrained model.
        """
        self.labeled_data_dir = Path(config.LABELED_DATA_DIR)
        self.pseudo_labels_dir = Path(config.PSEUDO_LABELS_DIR)
        self.model_output_dir = Path("backend/app/models/yolo") # Where the retrained model will be saved
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"StudentModelTrainer initialized. Labeled data: {self.labeled_data_dir}, Pseudo labels: {self.pseudo_labels_dir}")

    def _prepare_dataset_config(self, data_path: Path, class_names: List[str]) -> Path:
        """
        Creates a `dataset.yaml` file required by Ultralytics YOLO for training.
        This file specifies the paths to training images/labels and class names.

        Args:
            data_path (Path): The root directory containing the combined training data.
            class_names (List[str]): A list of all class names used in the dataset.

        Returns:
            Path: The path to the generated `dataset.yaml` file.
        """
        dataset_config_path = data_path / "dataset.yaml"
        
        # Create dummy structure for YOLO to find images/labels (e.g., data_path/images, data_path/labels)
        # Actual images and labels will be copied into data_path directly for simplicity in this setup.
        (data_path / "images").mkdir(exist_ok=True)
        (data_path / "labels").mkdir(exist_ok=True)

        data = {
            "path": str(data_path.resolve()),  # Absolute path to the dataset root
            "train": "images",                  # Relative path to training images from 'path'
            "val": "images",                    # Relative path to validation images from 'path'
            "names": {i: name for i, name in enumerate(class_names)} # Mapping of class IDs to names
        }

        with open(dataset_config_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        logger.info(f"Created dataset config: {dataset_config_path}")
        return dataset_config_path

    async def retrain_model(self) -> Dict[str, Any]:
        """
        Orchestrates the retraining of the student model. This involves:
        1. Combining human-labeled and pseudo-labeled data into a single directory.
        2. Generating a `dataset.yaml` configuration file for YOLO.
        3. Loading a base YOLO model for fine-tuning.
        4. Initiating the training process.

        Returns:
            Dict[str, Any]: A dictionary containing the retraining status, a message,
                            and the path to the newly trained model if successful.
        """
        logger.info("Starting student model retraining process...")

        # 1. Prepare a combined directory for training data.
        combined_data_dir = Path("data/combined_training_data")
        shutil.rmtree(combined_data_dir, ignore_errors=True) # Clean up previous combined data
        combined_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created combined training data directory: {combined_data_dir}")

        # Copy human-labeled data to the combined directory.
        if self.labeled_data_dir.exists():
            for item in self.labeled_data_dir.iterdir():
                if item.is_file():
                    shutil.copy(item, combined_data_dir)
                elif item.is_dir():
                    shutil.copytree(item, combined_data_dir / item.name, dirs_exist_ok=True)
            logger.info(f"Copied labeled data from {self.labeled_data_dir}")

        # Copy pseudo-labeled data to the combined directory.
        if self.pseudo_labels_dir.exists():
            for item in self.pseudo_labels_dir.iterdir():
                if item.is_file():
                    shutil.copy(item, combined_data_dir)
                elif item.is_dir():
                    shutil.copytree(item, combined_data_dir / item.name, dirs_exist_ok=True)
            logger.info(f"Copied pseudo-labeled data from {self.pseudo_labels_dir}")

        # Ensure there is data to train with.
        if not any(combined_data_dir.iterdir()):
            logger.warning("No data found for retraining. Aborting.")
            return {"status": "aborted", "message": "No data found for retraining.", "model_path": None}

        # 2. Prepare the dataset configuration file (`dataset.yaml`) for YOLO training.
        # Retrieves class names from the currently loaded detector model.
        class_names = list(detector.model.names.values()) if detector.model else []
        if not class_names:
            logger.error("Could not get class names from detector model. Aborting retraining.")
            return {"status": "aborted", "message": "Could not retrieve class names for dataset config.", "model_path": None}
            
        dataset_config_path = self._prepare_dataset_config(combined_data_dir, class_names)

        # 3. Load a base YOLO model for fine-tuning.
        try:
            # Use the currently configured YOLO model path as the base, or fallback to 'yolov8n.pt'.
            base_model_path = config.YOLO_MODEL_PATH if Path(config.YOLO_MODEL_PATH).exists() else "yolov8n.pt"
            student_model = YOLO(base_model_path)
            logger.info(f"Loaded base model for retraining: {base_model_path}")
        except Exception as e:
            logger.error(f"Failed to load base YOLO model for retraining: {e}")
            return {"status": "failed", "message": f"Failed to load base model: {e}", "model_path": None}

        # 4. Initiate the training of the student model.
        try:
            logger.info(f"Training student model for {config.RETRAIN_EPOCHS} epochs with image size {config.RETRAIN_IMG_SIZE} and batch size {config.RETRAIN_BATCH_SIZE} on device {detector.device}...")
            results = student_model.train(
                data=str(dataset_config_path), # Path to the dataset configuration file
                epochs=config.RETRAIN_EPOCHS,    # Number of training epochs
                imgsz=config.RETRAIN_IMG_SIZE,   # Image size for training
                batch=config.RETRAIN_BATCH_SIZE, # Batch size for training
                device=detector.device,          # Training device (GPU/CPU)
                project=str(self.model_output_dir.parent), # Parent directory for saving results
                name=f"yolo/{datetime.now().strftime('%Y%m%d_%H%M%S')}_student", # Specific run directory name
                half=config.HALF_PRECISION and detector.device != "cpu", # Half-precision training
                exist_ok=True # Allow overwriting existing results with the same name
            )
            trained_model_path = results.save_dir / "weights" / "best.pt"
            logger.success(f"Student model retraining completed. Model saved to: {trained_model_path}")

            # Note: For real-time deployment, updating the active model might require a service restart
            # or a more sophisticated model hot-swapping mechanism. For now, the new model's path is returned.

            return {"status": "success", "message": "Student model retrained successfully.", "model_path": str(trained_model_path)}
        except Exception as e:
            logger.error(f"Student model retraining failed: {e}")
            return {"status": "failed", "message": f"Retraining failed: {e}", "model_path": None}


pseudo_label_generator = PseudoLabelGenerator() # Singleton instance for pseudo-label generation
student_model_trainer = StudentModelTrainer()   # Singleton instance for student model training
