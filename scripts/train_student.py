import argparse
from pathlib import Path
from loguru import logger
import os
import asyncio

from backend.app.services.retrain import student_model_trainer
from backend.app.config import config

# Configure logger for this script
logger.remove()
logger.add(os.sys.stderr, level="INFO")

async def train_student_model_script(
    labeled_data_dir: str,
    pseudo_labels_dir: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    half: bool
):
    """
    Standalone script to train the student model.

    Args:
        labeled_data_dir (str): Directory containing human-labeled data.
        pseudo_labels_dir (str): Directory containing pseudo-labeled data.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        imgsz (int): Image size for training.
        device (str): Device to use for training (e.g., "cpu", "cuda:0").
        half (bool): Use half-precision (FP16) training if True and GPU is available.
    """
    logger.info("Starting student model training from script...")

    # Temporarily override config with script arguments if provided
    original_labeled_data_dir = config.LABELED_DATA_DIR
    original_pseudo_labels_dir = config.PSEUDO_LABELS_DIR
    original_retrain_epochs = config.RETRAIN_EPOCHS
    original_retrain_batch_size = config.RETRAIN_BATCH_SIZE
    original_retrain_img_size = config.RETRAIN_IMG_SIZE
    original_device = config.DEVICE
    original_half_precision = config.HALF_PRECISION

    config.LABELED_DATA_DIR = labeled_data_dir
    config.PSEUDO_LABELS_DIR = pseudo_labels_dir
    config.RETRAIN_EPOCHS = epochs
    config.RETRAIN_BATCH_SIZE = batch_size
    config.RETRAIN_IMG_SIZE = imgsz
    config.DEVICE = device
    config.HALF_PRECISION = half

    try:
        # Re-initialize trainer with potentially updated config values
        # In a real scenario, you might pass these directly to the trainer or re-instantiate it.
        trainer = student_model_trainer # Using the singleton, assumes config is dynamic or reloaded
        result = await trainer.retrain_model()
        logger.info(f"Script-triggered retraining completed with status: {result['status']}")
        return result
    except Exception as e:
        logger.error(f"Error during script-triggered retraining: {e}")
        return {"status": "failed", "message": f"Script retraining failed: {e}", "model_path": None}
    finally:
        # Restore original config values
        config.LABELED_DATA_DIR = original_labeled_data_dir
        config.PSEUDO_LABELS_DIR = original_pseudo_labels_dir
        config.RETRAIN_EPOCHS = original_retrain_epochs
        config.RETRAIN_BATCH_SIZE = original_retrain_batch_size
        config.RETRAIN_IMG_SIZE = original_retrain_img_size
        config.DEVICE = original_device
        config.HALF_PRECISION = original_half_precision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the student model using combined data.")
    parser.add_argument(
        "--labeled_data_dir", 
        type=str, 
        default=config.LABELED_DATA_DIR,
        help="Directory containing human-labeled data."
    )
    parser.add_argument(
        "--pseudo_labels_dir", 
        type=str, 
        default=config.PSEUDO_LABELS_DIR,
        help="Directory containing pseudo-labeled data."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=config.RETRAIN_EPOCHS,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=config.RETRAIN_BATCH_SIZE,
        help="Training batch size."
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=config.RETRAIN_IMG_SIZE,
        help="Image size for training."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=config.DEVICE,
        help="Device to use for training (e.g., 'cpu', 'cuda:0')."
    )
    parser.add_argument(
        "--half", 
        type=bool, 
        default=config.HALF_PRECISION,
        help="Use half-precision (FP16) training if True and GPU is available."
    )
    args = parser.parse_args()

    # Run the async function using asyncio
    asyncio.run(train_student_model_script(
        args.labeled_data_dir, 
        args.pseudo_labels_dir, 
        args.epochs, 
        args.batch_size, 
        args.imgsz, 
        args.device, 
        args.half
    ))
