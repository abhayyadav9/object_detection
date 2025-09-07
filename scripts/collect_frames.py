import cv2
import os
import argparse
from datetime import datetime
from pathlib import Path
from loguru import logger

# Configure logger for this script
logger.remove()
logger.add(os.sys.stderr, level="INFO")

def collect_frames(
    source: str,
    output_dir: str,
    interval: int = 1,
    max_frames: int = 0
):
    """
    Collects frames from a video source and saves them to a specified directory.

    Args:
        source (str): Video source (e.g., "0" for webcam, or a video file path).
        output_dir (str): Directory to save the collected frames.
        interval (int): Save every N-th frame.
        max_frames (int): Maximum number of frames to collect (0 for no limit).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Collecting frames to: {output_path}")

    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        frame_count = 0
        saved_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or error reading frame.")
                break

            frame_count += 1

            if frame_count % interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                frame_filename = output_path / f"frame_{timestamp}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                saved_frame_count += 1
                logger.info(f"Saved frame {saved_frame_count} as {frame_filename}")

                if max_frames > 0 and saved_frame_count >= max_frames:
                    logger.info(f"Reached maximum of {max_frames} frames. Stopping collection.")
                    break
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
                logger.info("'q' pressed. Stopping frame collection.")
                break

    except Exception as e:
        logger.error(f"Error during frame collection: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Frame collection finished. Total frames saved: {saved_frame_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect frames from a video stream.")
    parser.add_argument(
        "--source", 
        type=str, 
        default="0", 
        help="Video source (e.g., '0' for webcam, or path to a video file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/unlabeled",
        help="Directory to save collected frames."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Save every N-th frame."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum number of frames to collect (0 for no limit)."
    )
    args = parser.parse_args()

    collect_frames(args.source, args.output_dir, args.interval, args.max_frames)
