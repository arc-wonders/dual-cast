import cv2
import numpy as np
import time
import os
import argparse
from enum import Enum
from typing import Tuple, Optional
from inference_sdk import InferenceHTTPClient

class GamePhase(Enum):
    AGENT_SELECTION = "agent_selection"
    BUYING_PHASE = "buying_phase"
    GAMEPLAY = "gameplay"
    UNKNOWN = "unknown"

class ValorantPhaseDetector:
    def __init__(self, api_key="hY9qOmC03Dpg4JNVNeOp"):
        self.current_phase = GamePhase.UNKNOWN
        self.confidence_threshold = 0.7
        
        # Initialize YOLO client
        self.yolo_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        
        # Color ranges for different UI elements (HSV) - for agent selection
        self.color_ranges = {
            'valorant_red': (np.array([0, 120, 120]), np.array([10, 255, 255])),
            'valorant_blue': (np.array([100, 120, 120]), np.array([120, 255, 255])),
            'valorant_gold': (np.array([15, 120, 120]), np.array([30, 255, 255])),
        }
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better detection"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        return blurred
    
    def detect_text_regions(self, img: np.ndarray) -> list:
        """Detect text regions in the image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter based on size (typical text regions)
            if w > 20 and h > 10 and w < img.shape[1]//2:
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    def detect_agent_selection(self, img: np.ndarray) -> float:
        """Detect agent selection phase using original logic"""
        height, width = img.shape[:2]
        confidence = 0.0
        
        # Look for agent portraits (usually arranged in a grid)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect circular/rectangular agent portraits
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                 param1=50, param2=30, minRadius=30, maxRadius=80)
        
        agent_count = 0
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            agent_count = len(circles)
            
            # Agent selection requires MORE than 5 agents (not buying phase with 5 agents)
            if agent_count > 5:
                confidence += 0.4
            elif agent_count <= 5:
                # If 5 or fewer agents, likely buying phase, reduce confidence significantly
                confidence -= 0.3
        
        # Look for "SELECT AGENT" or similar text regions in top portion
        top_region = img[:height//3, :]
        text_regions = self.detect_text_regions(top_region)
        
        # Agent selection usually has prominent text at top
        large_text_regions = [r for r in text_regions if r[2] > 100 and r[3] > 20]
        if len(large_text_regions) >= 1:
            confidence += 0.3
        
        # Check for characteristic color patterns (agent ability colors)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Look for multiple distinct colors (different agent abilities)
        color_count = 0
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            if np.sum(mask) > 1000:  # Sufficient color presence
                color_count += 1
        
        if color_count >= 2:
            confidence += 0.3
        
        return max(min(confidence, 1.0), 0.0)
    
    def detect_yolo_phases(self, image_path: str) -> Tuple[GamePhase, float, dict]:
        """Use YOLO model to detect gameplay and buying phases"""
        try:
            # Run YOLO inference
            result = self.yolo_client.infer(image_path, model_id="valorant-lobqc/1")
            
            # Parse YOLO results
            detections = result.get('predictions', [])
            
            # Count different object types
            object_counts = {}
            total_confidence = 0
            
            for detection in detections:
                class_name = detection.get('class', '').lower()
                confidence = detection.get('confidence', 0)
                
                if class_name not in object_counts:
                    object_counts[class_name] = []
                object_counts[class_name].append(confidence)
                total_confidence += confidence
            
            # Determine phase based on detected objects
            gameplay_confidence = 0.0
            buying_confidence = 0.0
            
            # Gameplay indicators
            gameplay_objects = ['crosshair', 'minimap', 'health', 'armor', 'weapon_equipped', 'ability_icon']
            for obj in gameplay_objects:
                if obj in object_counts:
                    avg_conf = sum(object_counts[obj]) / len(object_counts[obj])
                    gameplay_confidence += avg_conf * 0.2  # Weight each indicator
            
            # Buying phase indicators
            buying_objects = ['weapon_icon', 'buy_menu', 'credits', 'shop_item']
            for obj in buying_objects:
                if obj in object_counts:
                    avg_conf = sum(object_counts[obj]) / len(object_counts[obj])
                    buying_confidence += avg_conf * 0.25  # Weight each indicator
            
            # Normalize confidences
            gameplay_confidence = min(gameplay_confidence, 1.0)
            buying_confidence = min(buying_confidence, 1.0)
            
            # Determine best phase
            if buying_confidence > gameplay_confidence and buying_confidence >= self.confidence_threshold:
                return GamePhase.BUYING_PHASE, buying_confidence, {
                    'detections': object_counts,
                    'total_objects': len(detections),
                    'yolo_result': result
                }
            elif gameplay_confidence >= self.confidence_threshold:
                return GamePhase.GAMEPLAY, gameplay_confidence, {
                    'detections': object_counts,
                    'total_objects': len(detections),
                    'yolo_result': result
                }
            else:
                return GamePhase.UNKNOWN, max(gameplay_confidence, buying_confidence), {
                    'detections': object_counts,
                    'total_objects': len(detections),
                    'yolo_result': result
                }
                
        except Exception as e:
            print(f"⚠️  YOLO detection failed: {str(e)}")
            return GamePhase.UNKNOWN, 0.0, {'error': str(e)}
    
    def analyze_frame(self, img: np.ndarray) -> Tuple[GamePhase, float]:
        """Analyze a single frame and return detected phase with confidence"""
        if img is None or img.size == 0:
            return GamePhase.UNKNOWN, 0.0
        
        # First check for agent selection using original logic
        agent_selection_confidence = self.detect_agent_selection(img)
        
        if agent_selection_confidence >= self.confidence_threshold:
            return GamePhase.AGENT_SELECTION, agent_selection_confidence
        
        # For other phases, we would need to save the image temporarily for YOLO
        # This is a limitation since YOLO expects file paths
        return GamePhase.UNKNOWN, 0.0
    
    def analyze_screenshot(self, image_path: str) -> Tuple[GamePhase, float, dict]:
        """Analyze a single screenshot and return detailed results"""
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found")
            return GamePhase.UNKNOWN, 0.0, {}
        
        # Load image for agent selection detection
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image '{image_path}'")
            return GamePhase.UNKNOWN, 0.0, {}
        
        print(f"\n🔍 Analyzing: {image_path}")
        print(f"📐 Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Step 1: Check for agent selection using original logic
        agent_selection_confidence = self.detect_agent_selection(img)
        
        if agent_selection_confidence >= self.confidence_threshold:
            print("\n📊 Detection Results:")
            print("-" * 40)
            print(f"✅ DETECTED Agent Selection        {agent_selection_confidence:.2f} ({agent_selection_confidence:.1%})")
            print(f"🎯 Final Detection: Agent Selection")
            print(f"🔢 Confidence: {agent_selection_confidence:.2f} ({agent_selection_confidence:.1%})")
            
            return GamePhase.AGENT_SELECTION, agent_selection_confidence, {
                'method': 'original_logic',
                'agent_selection_score': agent_selection_confidence
            }
        
        # Step 2: Use YOLO for gameplay and buying phase detection
        print("🤖 Running YOLO detection for gameplay/buying phases...")
        yolo_phase, yolo_confidence, yolo_details = self.detect_yolo_phases(image_path)
        
        # Combine results
        all_scores = {
            'agent_selection': agent_selection_confidence,
            'yolo_detection': yolo_confidence,
            'detected_objects': yolo_details.get('detections', {}),
            'total_yolo_objects': yolo_details.get('total_objects', 0)
        }
        
        print("\n📊 Detection Results:")
        print("-" * 40)
        print(f"{'❌' if agent_selection_confidence < self.confidence_threshold else '✅'} Agent Selection            {agent_selection_confidence:.2f} ({agent_selection_confidence:.1%})")
        
        if yolo_phase != GamePhase.UNKNOWN:
            status = "✅ DETECTED" if yolo_confidence >= self.confidence_threshold else "❌"
            print(f"{status} {yolo_phase.value.replace('_', ' ').title():<20} {yolo_confidence:.2f} ({yolo_confidence:.1%})")
        else:
            print(f"❌ Gameplay/Buying (YOLO)      {yolo_confidence:.2f} ({yolo_confidence:.1%})")
        
        # Print detected objects
        if yolo_details.get('detections'):
            print(f"\n🎯 YOLO Detected Objects:")
            for obj_type, confidences in yolo_details['detections'].items():
                avg_conf = sum(confidences) / len(confidences)
                print(f"   • {obj_type}: {len(confidences)} instances (avg: {avg_conf:.2f})")
        
        final_phase = yolo_phase if yolo_confidence >= self.confidence_threshold else GamePhase.UNKNOWN
        final_confidence = yolo_confidence
        
        print(f"\n🎯 Final Detection: {final_phase.value.replace('_', ' ').title()}")
        print(f"🔢 Confidence: {final_confidence:.2f} ({final_confidence:.1%})")
        
        return final_phase, final_confidence, all_scores
    
    def batch_analyze_screenshots(self, folder_path: str, show_images: bool = False):
        """Analyze multiple screenshots in a folder"""
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found")
            return
        
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"No image files found in '{folder_path}'")
            return
        
        print(f"\n🗂️  Found {len(image_files)} images in '{folder_path}'")
        print("="*60)
        
        results = {}
        for i, image_file in enumerate(sorted(image_files), 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
            image_path = os.path.join(folder_path, image_file)
            phase, confidence, scores = self.analyze_screenshot(image_path)
            results[image_file] = (phase, confidence, scores)
            
            # Optional: Display image with results
            if show_images:
                self.display_image_with_results(image_path, phase, confidence, scores)
            
            print("="*60)
        
        # Summary
        print(f"\n📈 SUMMARY ({len(image_files)} images)")
        print("-" * 40)
        phase_counts = {}
        for filename, (phase, conf, _) in results.items():
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        for phase, count in sorted(phase_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(image_files)) * 100
            print(f"{phase.value.replace('_', ' ').title():<20} {count:>3} images ({percentage:.1f}%)")
        
        return results
    
    def display_image_with_results(self, image_path: str, phase: GamePhase, confidence: float, scores: dict):
        """Display image with detection results overlay"""
        img = cv2.imread(image_path)
        if img is None:
            return
        
        # Resize image for display if too large
        height, width = img.shape[:2]
        if width > 1920 or height > 1080:
            scale = min(1920/width, 1080/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Add overlay with results
        overlay = img.copy()
        
        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (600, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Add text
        y_pos = 40
        cv2.putText(img, f"Detected: {phase.value.replace('_', ' ').title()}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_pos += 30
        cv2.putText(img, f"Confidence: {confidence:.2f} ({confidence:.1%})", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 25
        cv2.putText(img, "Detection Method:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_pos += 20
        method = scores.get('method', 'yolo')
        cv2.putText(img, f"  {method.replace('_', ' ').title()}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Show YOLO detected objects if available
        if 'detected_objects' in scores and scores['detected_objects']:
            y_pos += 25
            cv2.putText(img, "YOLO Objects:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            for obj_type, confidences in list(scores['detected_objects'].items())[:5]:  # Show max 5
                y_pos += 18
                avg_conf = sum(confidences) / len(confidences)
                text = f"  {obj_type}: {len(confidences)}x ({avg_conf:.2f})"
                cv2.putText(img, text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 100), 1)
        
        # Display image
        window_name = f"Detection Result - {os.path.basename(image_path)}"
        cv2.imshow(window_name, img)
        print(f"👁️  Displaying image. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Valorant Game Phase Detection with YOLO')
    parser.add_argument('--mode', choices=['screenshot', 'batch'], 
                       default='screenshot', help='Detection mode')
    parser.add_argument('--image', type=str, help='Path to screenshot for analysis')
    parser.add_argument('--folder', type=str, help='Folder containing screenshots for batch analysis')
    parser.add_argument('--show-images', action='store_true', 
                       help='Show images during analysis (for batch mode)')
    parser.add_argument('--confidence', type=float, default=0.7, 
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--api-key', type=str, default="hY9qOmC03Dpg4JNVNeOp",
                       help='Roboflow API key for YOLO model')
    
    args = parser.parse_args()
    
    # Create detector
    detector = ValorantPhaseDetector(api_key=args.api_key)
    detector.confidence_threshold = args.confidence
    
    print("🎮 VALORANT PHASE DETECTOR (with YOLO)")
    print("=" * 50)
    print("🔍 Agent Selection: Original Computer Vision Logic")
    print("🤖 Gameplay/Buying: YOLO Deep Learning Model")
    print("=" * 50)
    
    if args.mode == 'screenshot':
        if not args.image:
            print("❌ Error: Please provide --image path for screenshot mode")
            return
        
        print("📸 SCREENSHOT MODE - Single image analysis")
        detector.analyze_screenshot(args.image)
        
        # Optional: Show image
        if args.show_images:
            img = cv2.imread(args.image)
            if img is not None:
                # Resize if too large
                height, width = img.shape[:2]
                if width > 1920 or height > 1080:
                    scale = min(1920/width, 1080/height)
                    img = cv2.resize(img, (int(width*scale), int(height*scale)))
                
                cv2.imshow(f'Screenshot: {os.path.basename(args.image)}', img)
                print("👁️  Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
    elif args.mode == 'batch':
        if not args.folder:
            print("❌ Error: Please provide --folder path for batch mode")
            return
        
        print("🗂️  BATCH MODE - Multiple image analysis")
        detector.batch_analyze_screenshots(args.folder, show_images=args.show_images)


def interactive_mode():
    """Interactive mode for easy testing"""
    print("🎮 VALORANT PHASE DETECTOR (with YOLO)")
    print("=" * 50)
    
    # Get API key
    api_key = input("Enter Roboflow API key (press Enter for default): ").strip()
    if not api_key:
        api_key = "hY9qOmC03Dpg4JNVNeOp"
    
    detector = ValorantPhaseDetector(api_key=api_key)
    
    while True:
        print("\n🎮 VALORANT PHASE DETECTOR")
        print("=" * 40)
        print("1. 📸 Analyze Screenshot")
        print("2. 🗂️  Batch Analyze Folder")
        print("3. ⚙️  Settings")
        print("4. ❌ Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            image_path = input("\n📸 Enter screenshot path: ").strip()
            if image_path:
                phase, conf, scores = detector.analyze_screenshot(image_path)
                show_img = input("Display image? (y/n): ").lower().startswith('y')
                if show_img:
                    detector.display_image_with_results(image_path, phase, conf, scores)
            
        elif choice == '2':
            folder_path = input("\n🗂️  Enter folder path: ").strip()
            if folder_path:
                show_imgs = input("Show images during analysis? (y/n): ").lower().startswith('y')
                detector.batch_analyze_screenshots(folder_path, show_images=show_imgs)
            
        elif choice == '3':
            print(f"\n⚙️  Current confidence threshold: {detector.confidence_threshold:.2f}")
            new_threshold = input("Enter new threshold (0.0-1.0) or press Enter to skip: ").strip()
            if new_threshold:
                try:
                    detector.confidence_threshold = float(new_threshold)
                    print(f"✅ Threshold updated to {detector.confidence_threshold:.2f}")
                except ValueError:
                    print("❌ Invalid threshold value")
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    import sys
    
    # If no command line arguments, run interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()