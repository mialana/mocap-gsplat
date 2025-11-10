import argparse
import os
import cv2

def extract_frames(input_dir, start_frame=0, end_frame=None):
    # Create the FRAMES directory inside the input directory
    frames_dir = os.path.join(input_dir, "FRAMES")
    os.makedirs(frames_dir, exist_ok=True)

    # Iterate over all .avi files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".avi"):
            file_path = os.path.join(input_dir, file_name)
            capture_device = cv2.VideoCapture(file_path)
            stream_id = os.path.splitext(file_name)[0]  # Use the file name as stream ID

            frame_count = 0
            total_frames = int(capture_device.get(cv2.CAP_PROP_FRAME_COUNT))
            end_frame = end_frame if end_frame is not None else total_frames

            for frame_count in range(start_frame, end_frame):
                capture_device.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = capture_device.read()
                if not ret:
                    break

                # Determine the folder for the current frame
                folder_index = frame_count
                folder_name = f"frame{folder_index:04d}"
                folder_path = os.path.join(frames_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Save the frame as a PNG file
                frame_file_name = f"{stream_id}_frame{frame_count:04d}.png"
                frame_file_path = os.path.join(folder_path, frame_file_name)
                cv2.imwrite(frame_file_path, frame)

            capture_device.release()

def main():
    parser = argparse.ArgumentParser(description="Extract frames from .avi files and organize them.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing .avi files.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame number (inclusive).")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame number (exclusive).")
    args = parser.parse_args()

    extract_frames(args.input_dir, args.start_frame, args.end_frame)

if __name__ == "__main__":
    main()

