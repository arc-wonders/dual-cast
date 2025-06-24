import cv2
import numpy as np
import time
import os
import argparse
from enum import Enum
from typing import Tuple, Optional
import mss
import threading

class GamePhase(Enum):
    AGENT_SELECTION = "agent_selection"
    BUYING_PHASE = "buying_phase"
    GAMEPLAY = "gameplay"
    GAME_END = "game_end"
    UNKNOWN = "unknown"

class ValorantPhaseDetector:
    def __init__(self):
        self.current_phase = GamePhase.UNKNOWN
        self.confidence_threshold = 0.7
        self.phase_history = []
        self.max_history = 5
        
        # Screen capture settings
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor
        
        # Color ranges for different UI elements (HSV)
        self.color_ranges = {
            'valorant_red': (np.array([0, 120, 120]), np.array([10, 255, 255])),
            'valorant_blue': (np.array([100, 120, 120]), np.array([120, 255, 255])),
            'valorant_gold': (np.array([15, 120, 120]), np.array([30, 255, 255])),
        }
        
    def capture_screen(self) -> np.ndarray:
        """Capture the current screen"""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
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
        """Detect agent selection phase"""
        height, width = img.shape[:2]
        confidence = 0.0
        
        # Look for agent portraits (usually arranged in a grid)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect circular/rectangular agent portraits
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                 param1=50, param2=30, minRadius=30, maxRadius=80)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Agent selection typically has multiple agent portraits
            if len(circles) >= 5:
                confidence += 0.4
        
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
        
        return min(confidence, 1.0)
    
    def detect_buying_phase(self, img: np.ndarray) -> float:
     
     """Improved detect weapon buying phase by counting guns detected"""
     height, width = img.shape[:2]
     confidence = 0.0
    
    # Convert to grayscale for easier feature detection
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Look for weapon icons (silhouettes of guns) using edge detection
     edges = cv2.Canny(gray, 50, 150)
    
    # Find contours of potential weapon slots
     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count the number of potential weapon icons
     weapon_count = 0
     for contour in contours:
         x, y, w, h = cv2.boundingRect(contour)
         aspect_ratio = w / h if h > 0 else 0
         area = w * h
        
        # Look for rectangular shapes that might indicate weapon icons
         if 2.0 < aspect_ratio < 4.0 and area > 500 and w > 60 and h > 20:
            weapon_count += 1
    
    # If we detect more than 3 weapons, it's likely the buying phase
     if weapon_count > 3:
         confidence += 0.7
    
    # Look for credit/money display
     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
     white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([255, 30, 255]))
     white_pixels = np.sum(white_mask) / 255
    
    # If there is enough white text for credits, increase confidence
     if white_pixels > 5000:
         confidence += 0.3
     
     return min(confidence, 1.0)
 
    def detect_gameplay(self, img: np.ndarray) -> float:
     height, width = img.shape[:2]
     confidence = 0.0
    
    # 1. Look for crosshair detection (focused on the center of the screen)
     center_region = img[height//2-30:height//2+30, width//2-30:width//2+30]
     gray_center = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
    
    # Detect edges in the center to find crosshair
     edges = cv2.Canny(gray_center, 50, 150)
     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=8, maxLineGap=3)
    
    # If we detect two or more lines intersecting, it indicates a crosshair
     if lines is not None and len(lines) >= 2:
         confidence += 0.4
    
    # 2. Look for health indicators (e.g., green/red bars) near the bottom of the screen
     bottom_region = img[int(height*0.8):, :]
     hsv_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)
    
     green_mask = cv2.inRange(hsv_bottom, np.array([40, 100, 100]), np.array([80, 255, 255]))  # Health (green)
     red_mask = cv2.inRange(hsv_bottom, np.array([0, 100, 100]), np.array([10, 255, 255]))    # Damage (red)
    
     green_pixels = np.sum(green_mask) / 255
     red_pixels = np.sum(red_mask) / 255
    
     if green_pixels > 500 or red_pixels > 300:
         confidence += 0.3
    
    # 3. Look for minimap (corner of the screen) - distinguish from agent selection phase
     minimap_regions = [
         img[:height//5, :width//5],      # Top-left
         img[:height//5, 4*width//5:],    # Top-right
         img[4*height//5:, :width//5],    # Bottom-left
         img[4*height//5:, 4*width//5:]   # Bottom-right
     ]
    
     for region in minimap_regions:
         if region.size > 0:
             gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
             
            # Look for circular minimap (distinct boundary)
             circles = cv2.HoughCircles(gray_region, cv2.HOUGH_GRADIENT, 1, 50,
                                          param1=50, param2=30, minRadius=20, maxRadius=80)
            
             if circles is not None and len(circles[0]) >= 1:
                 confidence += 0.2
                 break
            
            # Alternative: look for rectangular minimap with content
             edges = cv2.Canny(gray_region, 30, 100)
             contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             small_details = len([c for c in contours if 50 < cv2.contourArea(c) < 1000])
            
             if small_details >= 3:
                 confidence += 0.2
                 break
    
    # 4. Look for ability cooldown indicators in the bottom section (icon with distinct colors)
     ability_region = img[int(height*0.6):int(height*0.9), int(width*0.3):int(width*0.7)]
    
     if ability_region.size > 0:
         hsv_abilities = cv2.cvtColor(ability_region, cv2.COLOR_BGR2HSV)
        
         blue_mask = cv2.inRange(hsv_abilities, np.array([100, 100, 100]), np.array([130, 255, 255]))
         purple_mask = cv2.inRange(hsv_abilities, np.array([130, 100, 100]), np.array([160, 255, 255]))
        
         if np.sum(blue_mask) > 200 or np.sum(purple_mask) > 200:
             confidence += 0.2
     
    # 5. Avoid false positives from agent selection phase by detecting movement or shooting indicators
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     edges = cv2.Canny(gray, 30, 100)
     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
     large_rectangles = 0
     for contour in contours:
         x, y, w, h = cv2.boundingRect(contour)
         aspect_ratio = w / h if h > 0 else 0
         area = w * h
        
        # Ignore large rectangular regions that might indicate agent selection
         if (2.0 < aspect_ratio < 5.0 and area > 5000):
             large_rectangles += 1
    
     # If too many large rectangles are detected, it might be agent selection (not gameplay)
     if large_rectangles >= 4:
         confidence *= 0.5
    
     return min(confidence, 1.0)
  
    def detect_game_end(self, img: np.ndarray) -> float:
        """Detect game end/result phase"""
        height, width = img.shape[:2]
        confidence = 0.0
        
        # Look for large text in center (Victory/Defeat)
        center_region = img[height//4:3*height//4, width//4:3*width//4]
        text_regions = self.detect_text_regions(center_region)
        
        # Game end screen typically has large prominent text
        large_text = [r for r in text_regions if r[2] > 150 and r[3] > 30]
        if len(large_text) >= 1:
            confidence += 0.4
        
        # Look for scoreboard (player stats)
        # Scoreboards typically have tabular data
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal lines (table rows)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count horizontal line segments
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        horizontal_count = len([c for c in contours if cv2.boundingRect(c)[2] > 100])
        
        if horizontal_count >= 3:  # Multiple rows suggest scoreboard
            confidence += 0.4
        
        # Look for player names/stats (multiple text regions in organized layout)
        all_text_regions = self.detect_text_regions(img)
        
        # Game end screens typically have many text elements (player names, stats, etc.)
        if len(all_text_regions) >= 10:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def analyze_frame(self, img: np.ndarray) -> Tuple[GamePhase, float]:
        """Analyze a single frame and return detected phase with confidence"""
        if img is None or img.size == 0:
            return GamePhase.UNKNOWN, 0.0
        
        # Preprocess image
        processed = self.preprocess_image(img)
        
        # Test each phase detection
        phase_scores = {
            GamePhase.AGENT_SELECTION: self.detect_agent_selection(img),
            GamePhase.BUYING_PHASE: self.detect_buying_phase(img),
            GamePhase.GAMEPLAY: self.detect_gameplay(img),
            GamePhase.GAME_END: self.detect_game_end(img)
        }
        
        # Find phase with highest confidence
        best_phase = max(phase_scores.items(), key=lambda x: x[1])
        
        if best_phase[1] >= self.confidence_threshold:
            return best_phase[0], best_phase[1]
        else:
            return GamePhase.UNKNOWN, best_phase[1]
    
    def analyze_screenshot(self, image_path: str) -> Tuple[GamePhase, float, dict]:
        """Analyze a single screenshot and return detailed results"""
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found")
            return GamePhase.UNKNOWN, 0.0, {}
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image '{image_path}'")
            return GamePhase.UNKNOWN, 0.0, {}
        
        print(f"\n🔍 Analyzing: {image_path}")
        print(f"📐 Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Get detailed scores for all phases
        phase_scores = {
            GamePhase.AGENT_SELECTION: self.detect_agent_selection(img),
            GamePhase.BUYING_PHASE: self.detect_buying_phase(img),
            GamePhase.GAMEPLAY: self.detect_gameplay(img),
            GamePhase.GAME_END: self.detect_game_end(img)
        }
        
        # Find best match
        best_phase = max(phase_scores.items(), key=lambda x: x[1])
        detected_phase = best_phase[0] if best_phase[1] >= self.confidence_threshold else GamePhase.UNKNOWN
        
        # Print detailed results
        print("\n📊 Detection Results:")
        print("-" * 40)
        for phase, score in sorted(phase_scores.items(), key=lambda x: x[1], reverse=True):
            status = "✅ DETECTED" if phase == detected_phase and score >= self.confidence_threshold else "❌"
            print(f"{status} {phase.value.replace('_', ' ').title():<20} {score:.2f} ({score:.1%})")
        
        print(f"\n🎯 Final Detection: {detected_phase.value.replace('_', ' ').title()}")
        print(f"🔢 Confidence: {best_phase[1]:.2f} ({best_phase[1]:.1%})")
        
        return detected_phase, best_phase[1], phase_scores
    
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
        for image_file in sorted(image_files):
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
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Add text
        y_pos = 40
        cv2.putText(img, f"Detected: {phase.value.replace('_', ' ').title()}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_pos += 30
        cv2.putText(img, f"Confidence: {confidence:.2f} ({confidence:.1%})", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 25
        cv2.putText(img, "All Scores:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show all phase scores
        for i, (p, score) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)):
            y_pos += 20
            color = (0, 255, 0) if p == phase else (100, 100, 100)
            text = f"{p.value.replace('_', ' ')}: {score:.2f}"
            cv2.putText(img, text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Display image
        window_name = f"Detection Result - {os.path.basename(image_path)}"
        cv2.imshow(window_name, img)
        print(f"👁️  Displaying image. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def smooth_detection(self, phase: GamePhase) -> GamePhase:
        """Smooth detection using history to reduce flickering"""
        self.phase_history.append(phase)
        
        if len(self.phase_history) > self.max_history:
            self.phase_history.pop(0)
        
        # Use majority vote from recent history
        if len(self.phase_history) >= 3:
            phase_counts = {}
            for p in self.phase_history:
                phase_counts[p] = phase_counts.get(p, 0) + 1
            
            most_common = max(phase_counts.items(), key=lambda x: x[1])
            return most_common[0]
        
        return phase
        """Smooth detection using history to reduce flickering"""
        self.phase_history.append(phase)
        
        if len(self.phase_history) > self.max_history:
            self.phase_history.pop(0)
        
        # Use majority vote from recent history
        if len(self.phase_history) >= 3:
            phase_counts = {}
            for p in self.phase_history:
                phase_counts[p] = phase_counts.get(p, 0) + 1
            
            most_common = max(phase_counts.items(), key=lambda x: x[1])
            return most_common[0]
        
        return phase
    
    def run_live_detection(self, callback=None, show_display=False):
        """Run continuous live phase detection"""
        print("🔴 Starting LIVE Valorant phase detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            try:
                # Capture screen
                frame = self.capture_screen()
                
                if frame is not None:
                    # Analyze frame
                    detected_phase, confidence = self.analyze_frame(frame)
                    
                    # Smooth detection
                    smoothed_phase = self.smooth_detection(detected_phase)
                    
                    # Update current phase if changed
                    if smoothed_phase != self.current_phase:
                        self.current_phase = smoothed_phase
                        print(f"🎮 Phase changed to: {smoothed_phase.value.replace('_', ' ').title()} (confidence: {confidence:.2f})")
                        
                        # Call callback if provided
                        if callback:
                            callback(smoothed_phase, confidence)
                    
                    # Optional display
                    if show_display:
                        display_frame = frame.copy()
                        
                        # Add overlay
                        cv2.rectangle(display_frame, (10, 10), (400, 100), (0, 0, 0), -1)
                        cv2.rectangle(display_frame, (10, 10), (400, 100), (0, 255, 0), 2)
                        
                        cv2.putText(display_frame, f"Phase: {smoothed_phase.value.replace('_', ' ').title()}", 
                                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(display_frame, "Press 'q' to quit, 's' to save", 
                                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                        
                        # Resize for display
                        display_frame = cv2.resize(display_frame, (1280, 720))
                        cv2.imshow('Valorant Live Detection', display_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            # Save screenshot
                            timestamp = int(time.time())
                            filename = f"valorant_screenshot_{timestamp}_{smoothed_phase.value}.png"
                            cv2.imwrite(filename, frame)
                            print(f"💾 Screenshot saved: {filename}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n⏹️  Stopping live detection...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(1)
        
        cv2.destroyAllWindows()
        """Run continuous phase detection"""
        print("Starting Valorant phase detection...")
        print("Press 'q' to quit")
        
        while True:
            try:
                # Capture screen
                frame = self.capture_screen()
                
                if frame is not None:
                    # Analyze frame
                    detected_phase, confidence = self.analyze_frame(frame)
                    
                    # Smooth detection
                    smoothed_phase = self.smooth_detection(detected_phase)
                    
                    # Update current phase if changed
                    if smoothed_phase != self.current_phase:
                        self.current_phase = smoothed_phase
                        print(f"Phase changed to: {smoothed_phase.value} (confidence: {confidence:.2f})")
                        
                        # Call callback if provided
                        if callback:
                            callback(smoothed_phase, confidence)
                    
                    # Display result (optional - comment out for performance)
                    # cv2.putText(frame, f"Phase: {smoothed_phase.value}", (10, 30), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # 
                    # # Resize for display
                    # display_frame = cv2.resize(frame, (1280, 720))
                    # cv2.imshow('Valorant Phase Detection', display_frame)
                    
                    # Check for quit
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\nStopping detection...")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
        
        cv2.destroyAllWindows()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Valorant Game Phase Detection')
    parser.add_argument('--mode', choices=['live', 'screenshot', 'batch'], 
                       default='live', help='Detection mode')
    parser.add_argument('--image', type=str, help='Path to screenshot for analysis')
    parser.add_argument('--folder', type=str, help='Folder containing screenshots for batch analysis')
    parser.add_argument('--show-display', action='store_true', 
                       help='Show visual display (for live mode)')
    parser.add_argument('--show-images', action='store_true', 
                       help='Show images during analysis (for batch mode)')
    parser.add_argument('--confidence', type=float, default=0.7, 
                       help='Confidence threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Create detector
    detector = ValorantPhaseDetector()
    detector.confidence_threshold = args.confidence
    
    print("🎮 VALORANT PHASE DETECTOR")
    print("=" * 50)
    
    if args.mode == 'live':
        print("🔴 LIVE MODE - Real-time detection")
        detector.run_live_detection(
            callback=phase_change_callback, 
            show_display=args.show_display
        )
        
    elif args.mode == 'screenshot':
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

# Example usage functions
def phase_change_callback(phase: GamePhase, confidence: float):
    """Callback function called when phase changes in live mode"""
    print(f"🎮 Game phase: {phase.value.replace('_', ' ').title()}")
    print(f"📊 Confidence: {confidence:.1%}")
    
    # Add specific actions for each phase
    if phase == GamePhase.AGENT_SELECTION:
        print("💡 Tip: Choose your agent wisely!")
    elif phase == GamePhase.BUYING_PHASE:
        print("💰 Tip: Buy weapons and utility!")
    elif phase == GamePhase.GAMEPLAY:
        print("⚡ Tip: Focus and communicate with team!")
    elif phase == GamePhase.GAME_END:
        print("🏆 Tip: Review your performance!")

def interactive_mode():
    """Interactive mode for easy testing"""
    detector = ValorantPhaseDetector()
    
    while True:
        print("\n🎮 VALORANT PHASE DETECTOR")
        print("=" * 40)
        print("1. 🔴 Live Detection")
        print("2. 📸 Analyze Screenshot")
        print("3. 🗂️  Batch Analyze Folder")
        print("4. ⚙️  Settings")
        print("5. ❌ Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            print("\n🔴 Starting live detection...")
            print("Options:")
            show_display = input("Show visual display? (y/n): ").lower().startswith('y')
            detector.run_live_detection(callback=phase_change_callback, show_display=show_display)
            
        elif choice == '2':
            image_path = input("\n📸 Enter screenshot path: ").strip()
            if image_path:
                phase, conf, scores = detector.analyze_screenshot(image_path)
                show_img = input("Display image? (y/n): ").lower().startswith('y')
                if show_img:
                    detector.display_image_with_results(image_path, phase, conf, scores)
            
        elif choice == '3':
            folder_path = input("\n🗂️  Enter folder path: ").strip()
            if folder_path:
                show_imgs = input("Show images during analysis? (y/n): ").lower().startswith('y')
                detector.batch_analyze_screenshots(folder_path, show_images=show_imgs)
            
        elif choice == '4':
            print(f"\n⚙️  Current confidence threshold: {detector.confidence_threshold:.2f}")
            new_threshold = input("Enter new threshold (0.0-1.0) or press Enter to skip: ").strip()
            if new_threshold:
                try:
                    detector.confidence_threshold = float(new_threshold)
                    print(f"✅ Threshold updated to {detector.confidence_threshold:.2f}")
                except ValueError:
                    print("❌ Invalid threshold value")
            
        elif choice == '5':
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