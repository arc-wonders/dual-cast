import cv2
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import numpy as np

class YOLOv8Detector:
    def __init__(self, model_path="D:/dual cast/models/agent_phase.pt"):
        """
        Initialize YOLOv8 detector with custom trained model
        
        Args:
            model_path (str): Path to the trained YOLOv8 model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_image(self, image_path, output_dir="results", conf_threshold=0.85):
        """
        Detect objects in a single image
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Directory to save results
            conf_threshold (float): Confidence threshold for detections
            
        Returns:
            dict: Detection results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run inference
        results = self.model(image_path, conf=conf_threshold)
        
        # Process results
        image_name = Path(image_path).stem
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(confidence),
                        "class_id": class_id,
                        "class_name": class_name
                    }
                    detections.append(detection)
        
        # Prepare result data
        result_data = {
            "image_path": image_path,
            "image_name": image_name,
            "timestamp": datetime.now().isoformat(),
            "total_detections": len(detections),
            "detections": detections
        }
        
        # Save annotated image
        annotated_img = results[0].plot()
        annotated_path = os.path.join(output_dir, f"{image_name}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated_img)
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"{image_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"Results saved: {json_path}")
        print(f"Annotated image saved: {annotated_path}")
        
        return result_data
    
    def detect_video(self, video_path, output_dir="results", conf_threshold=0.85, frame_skip=30, create_output_video=True):
        """
        Enhanced video processing with two-pass approach from reference code
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save results
            conf_threshold (float): Confidence threshold for detections
            frame_skip (int): Process every Nth frame for inference
            create_output_video (bool): Whether to create annotated output video
            
        Returns:
            dict: Detection results for all frames
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        video_name = Path(video_path).stem
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {original_fps:.2f}, Resolution: {width}x{height}")
        print(f"Processing every {frame_skip} frames for inference...")
        
        # Initialize video writer for output video
        out = None
        if create_output_video:
            output_video_path = os.path.join(output_dir, f"{video_name}_with_detections.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (width, height))
            print(f"Output video will be saved to: {output_video_path}")
        
        # Store detection results for all frames (for video output)
        all_frame_detections = {}
        processed_frame_results = []
        
        # First pass: run inference on selected frames
        frame_count = 0
        processed_frames = 0
        
        print("First pass: Running inference on selected frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every nth frame for inference
            if frame_count % frame_skip == 0:
                try:
                    # Run inference
                    results = self.model(frame, conf=conf_threshold)
                    
                    # Process detections for current frame
                    frame_detections = []
                    for r in results:
                        boxes = r.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                class_id = int(box.cls[0].cpu().numpy())
                                class_name = self.model.names[class_id]
                                
                                detection = {
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "confidence": float(confidence),
                                    "class_id": class_id,
                                    "class_name": class_name
                                }
                                frame_detections.append(detection)
                    
                    # Store detections for this frame
                    all_frame_detections[frame_count] = frame_detections
                    
                    # Store frame results
                    frame_data = {
                        "original_frame_number": frame_count,
                        "processed_frame_number": processed_frames,
                        "timestamp": frame_count / original_fps,
                        "detections": frame_detections
                    }
                    processed_frame_results.append(frame_data)
                    
                    processed_frames += 1
                    
                    detection_count = len(frame_detections)
                    if detection_count > 0:
                        print(f"Processed frame {frame_count}/{total_frames} (timestamp: {frame_count/original_fps:.2f}s) - Found {detection_count} detections")
                    else:
                        print(f"Processed frame {frame_count}/{total_frames} (timestamp: {frame_count/original_fps:.2f}s) - No detections")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    all_frame_detections[frame_count] = []
            
            frame_count += 1
        
        # Second pass: create output video with detections
        if create_output_video and out is not None:
            print("Second pass: Creating output video with detections...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Find the closest processed frame for detections
                closest_processed_frame = self._find_closest_processed_frame(frame_count, all_frame_detections, frame_skip)
                
                if closest_processed_frame is not None and closest_processed_frame in all_frame_detections:
                    detections = all_frame_detections[closest_processed_frame]
                    frame = self._draw_detections_on_frame(frame, detections)
                
                out.write(frame)
                frame_count += 1
                
                # Progress indicator
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Video creation progress: {progress:.1f}%")
            
            out.release()
            print(f"✅ Output video created successfully: {output_video_path}")
        
        # Release resources
        cap.release()
        
        # Prepare final results
        video_results = {
            "video_path": video_path,
            "video_name": video_name,
            "timestamp": datetime.now().isoformat(),
            "video_info": {
                "original_fps": original_fps,
                "frame_skip": frame_skip,
                "width": width,
                "height": height,
                "total_original_frames": total_frames,
                "total_processed_frames": processed_frames,
                "duration_seconds": total_frames / original_fps
            },
            "processing_settings": {
                "confidence_threshold": conf_threshold,
                "frame_skip": frame_skip,
                "create_output_video": create_output_video
            },
            "frames": processed_frame_results
        }
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"{video_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(video_results, f, indent=2)
        
        print(f"Video processing complete!")
        print(f"Results saved: {json_path}")
        if create_output_video:
            print(f"Annotated video saved: {output_video_path}")
        print(f"Processed {processed_frames} out of {total_frames} frames")
        
        return video_results
    
    def _find_closest_processed_frame(self, current_frame, detections_dict, frame_skip):
        """Find the closest processed frame to apply detections"""
        # Find the most recent processed frame
        processed_frames = [f for f in detections_dict.keys() if f <= current_frame]
        if processed_frames:
            return max(processed_frames)
        return None
    
    def _draw_detections_on_frame(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            if 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw rectangle with thicker line for video
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add label
                class_name = detection.get('class_name', 'Unknown')
                confidence = detection.get('confidence', 0)
                label = f"{class_name} ({confidence:.2f})"
                
                # Add background for text readability
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    def detect_folder(self, folder_path, output_dir="results", conf_threshold=0.85):
        """
        Detect objects in all images in a folder
        
        Args:
            folder_path (str): Path to folder containing images
            output_dir (str): Directory to save results
            conf_threshold (float): Confidence threshold for detections
            
        Returns:
            dict: Detection results for all images
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tga'}
        
        # Find all images in folder
        folder_path = Path(folder_path)
        image_files = [f for f in folder_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {folder_path}")
            return None
        
        print(f"Found {len(image_files)} images in {folder_path}")
        
        # Process each image
        all_results = []
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            # Create subdirectory for this image
            image_output_dir = os.path.join(output_dir, "individual_images")
            
            # Process image
            result = self.detect_image(str(image_file), image_output_dir, conf_threshold)
            all_results.append(result)
        
        # Compile folder results
        folder_results = {
            "folder_path": str(folder_path),
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_files),
            "images_processed": len(all_results),
            "results": all_results
        }
        
        # Save combined JSON results
        json_path = os.path.join(output_dir, f"folder_results.json")
        with open(json_path, 'w') as f:
            json.dump(folder_results, f, indent=2)
        
        print(f"Folder processing complete!")
        print(f"Combined results saved: {json_path}")
        
        return folder_results

def get_user_input():
    """Get user input interactively"""
    print("\n" + "="*60)
    print("           YOLOv8 Object Detection System")
    print("="*60)
    
    # Get input type
    print("\nSelect input type:")
    print("1. Single Image")
    print("2. Video File")
    print("3. Image Folder")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    input_type_map = {'1': 'image', '2': 'video', '3': 'folder'}
    input_type = input_type_map[choice]
    
    # Get input path
    if input_type == 'image':
        input_path = input("\nEnter path to image file: ").strip().strip('"')
    elif input_type == 'video':
        input_path = input("\nEnter path to video file: ").strip().strip('"')
    else:  # folder
        input_path = input("\nEnter path to image folder: ").strip().strip('"')
    
    # Validate input path
    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' does not exist!")
        return None
    
    # Get model path
    default_model = "D:/dual cast/models/agent_phase.pt"
    print(f"\nModel path (default: {default_model})")
    model_path = input("Enter custom model path or press Enter for default: ").strip().strip('"')
    if not model_path:
        model_path = default_model
    
    # Validate model path
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist!")
        return None
    
    # Get output directory
    default_output = "results"
    print(f"\nOutput directory (default: {default_output})")
    output_dir = input("Enter output directory or press Enter for default: ").strip().strip('"')
    if not output_dir:
        output_dir = default_output
    
    # Get confidence threshold
    print(f"\nConfidence threshold (default: 0.85)")
    conf_input = input("Enter confidence threshold (0.0-1.0) or press Enter for default: ").strip()
    try:
        conf_threshold = float(conf_input) if conf_input else 0.85
        if not (0 <= conf_threshold <= 1):
            print("Warning: Confidence threshold should be between 0 and 1. Using default 0.85")
            conf_threshold = 0.85
    except ValueError:
        print("Invalid confidence threshold. Using default 0.85")
        conf_threshold = 0.85
    
    # Get frame skip and video options for video processing
    frame_skip = None
    create_output_video = True
    if input_type == 'video':
        print(f"\nVideo processing options:")
        print("Frame skip determines how many frames to skip between detections.")
        print("Higher values = faster processing but fewer detections.")
        
        skip_input = input("Enter frame skip interval (default: 30): ").strip()
        try:
            frame_skip = int(skip_input) if skip_input else 30
            if frame_skip <= 0:
                print("Warning: Frame skip should be greater than 0. Using default 30")
                frame_skip = 30
        except ValueError:
            print("Invalid frame skip value. Using default 30")
            frame_skip = 30
        
        video_output = input("Create annotated output video? (y/n, default: y): ").strip().lower()
        create_output_video = video_output != 'n'
    
    return {
        'input_type': input_type,
        'input_path': input_path,
        'model_path': model_path,
        'output_dir': output_dir,
        'conf_threshold': conf_threshold,
        'frame_skip': frame_skip,
        'create_output_video': create_output_video
    }

def interactive_mode():
    """Run in interactive mode"""
    print("Starting YOLOv8 Detection System...")
    
    while True:
        # Get user input
        config = get_user_input()
        if config is None:
            continue
        
        try:
            # Initialize detector
            print(f"\nLoading model: {config['model_path']}")
            detector = YOLOv8Detector(config['model_path'])
            
            # Show processing info
            print(f"\nProcessing Configuration:")
            print(f"  Input Type: {config['input_type'].upper()}")
            print(f"  Input Path: {config['input_path']}")
            print(f"  Output Directory: {config['output_dir']}")
            print(f"  Confidence Threshold: {config['conf_threshold']}")
            if config['frame_skip'] is not None:
                print(f"  Frame Skip: {config['frame_skip']}")
                print(f"  Create Output Video: {config['create_output_video']}")
            
            # Process based on input type
            print(f"\nStarting detection...")
            if config['input_type'] == 'video':
                detector.detect_video(
                    config['input_path'], 
                    config['output_dir'], 
                    config['conf_threshold'], 
                    config['frame_skip'],
                    config['create_output_video']
                )
            elif config['input_type'] == 'image':
                detector.detect_image(config['input_path'], config['output_dir'], config['conf_threshold'])
            elif config['input_type'] == 'folder':
                detector.detect_folder(config['input_path'], config['output_dir'], config['conf_threshold'])
            
            print("\n" + "="*60)
            print("           DETECTION COMPLETED SUCCESSFULLY!")
            print("="*60)
            
        except Exception as e:
            print(f"\nError during processing: {e}")
            
        # Ask if user wants to continue
        print("\n" + "-"*60)
        while True:
            continue_choice = input("Do you want to process another file? (y/n): ").strip().lower()
            if continue_choice in ['y', 'yes', 'n', 'no']:
                break
            print("Please enter 'y' for yes or 'n' for no.")
        
        if continue_choice in ['n', 'no']:
            print("\nThank you for using YOLOv8 Detection System!")
            break

def main():
    """Main function with both interactive and command line modes"""
    import sys
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # Command line mode
        parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
        parser.add_argument("--input", "-i", required=True, help="Input path (video, image, or folder)")
        parser.add_argument("--type", "-t", choices=["video", "image", "folder"], 
                           required=True, help="Input type")
        parser.add_argument("--model", "-m", default="D:/dual cast/models/agent_phase.pt", 
                           help="Path to YOLOv8 model")
        parser.add_argument("--output", "-o", default="results", 
                           help="Output directory")
        parser.add_argument("--conf", "-c", type=float, default=0.85, 
                           help="Confidence threshold (default: 0.85)")
        parser.add_argument("--skip", "-s", type=int, default=30,
                           help="Frame skip interval for video processing (default: 30)")
        parser.add_argument("--no-video", action="store_true",
                           help="Don't create annotated output video")
        
        args = parser.parse_args()
        
        # Initialize detector
        detector = YOLOv8Detector(args.model)
        
        # Process based on input type
        if args.type == "video":
            detector.detect_video(
                args.input, 
                args.output, 
                args.conf, 
                args.skip,
                not args.no_video
            )
        elif args.type == "image":
            detector.detect_image(args.input, args.output, args.conf)
        elif args.type == "folder":
            detector.detect_folder(args.input, args.output, args.conf)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()