import cv2
import numpy as np
import json
import argparse
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import os
import sys

# Try to import YOLO - you can replace this with your preferred detection library
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Using mock detection for demo.")

class PlayerState(Enum):
    NOT_SELECTED = "not_selected"
    HOVERING = "hovering"
    LOCKED = "locked"

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float
    
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box"""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection
        return intersection / union if union > 0 else 0.0

@dataclass
class PlayerStatus:
    state: PlayerState = PlayerState.NOT_SELECTED
    agent: Optional[str] = None
    last_hover_agent: Optional[str] = None
    hover_frames: int = 0
    lock_frames: int = 0
    last_position: Optional[Tuple[float, float]] = None

class ValorantImageAnalyzer:
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize Valorant Image Analyzer
        
        Args:
            model_path: Path to YOLO model file (.pt), if None uses mock detection
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if model_path and YOLO_AVAILABLE:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Loaded YOLO model from {model_path}")
            else:
                print(f"Model file {model_path} not found. Using mock detection.")
        else:
            print("Using mock detection mode (no YOLO model provided)")
        
        # Valorant agent names
        self.agent_names = {
            "jett", "reyna", "phoenix", "raze", "yoru", "neon",  # Duelists
            "sova", "breach", "skye", "kayo", "fade", "gekko",   # Initiators
            "omen", "brimstone", "astra", "viper", "harbor",    # Controllers
            "sage", "cypher", "killjoy", "chamber", "deadlock"  # Sentinels
        }
        
        # Detection class names your YOLO model should recognize
        self.detection_classes = [
            "agent_card", "highlighted_agent", "lock_in_button", 
            "locked_icon", "timer_bar", "player_name"
        ] + [f"agent_{agent}" for agent in self.agent_names]
    
    def mock_detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Mock detection function for demo purposes
        Replace this with your actual YOLO model inference
        """
        height, width = image.shape[:2]
        
        # Simulate some detections based on image analysis
        detections = []
        
        # Simple color-based detection for demo
        # In reality, your YOLO model would do this detection
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Mock detection: Look for bright regions (could be highlighted agents)
        bright_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours[:3]):  # Limit to 3 detections
            if cv2.contourArea(contour) > 1000:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                
                # Mock different types of detections
                if i == 0:
                    class_name = "highlighted_agent_jett"
                elif i == 1:
                    class_name = "highlighted_agent_sova"
                else:
                    class_name = "agent_card"
                
                detection = {
                    "x1": float(x),
                    "y1": float(y),
                    "x2": float(x + w),
                    "y2": float(y + h),
                    "class": class_name,
                    "confidence": 0.85 + i * 0.05
                }
                detections.append(detection)
        
        # Add mock locked icon if bright regions found
        if len(detections) > 0:
            first_det = detections[0]
            lock_detection = {
                "x1": first_det["x1"] + 50,
                "y1": first_det["y1"] + 10,
                "x2": first_det["x1"] + 80,
                "y2": first_det["y1"] + 40,
                "class": "locked_icon",
                "confidence": 0.90
            }
            detections.append(lock_detection)
        
        return detections
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image using YOLO model or mock detection
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        if self.model is not None:
            # Real YOLO detection
            results = self.model(image, conf=self.confidence_threshold)
            detections = []
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confidences, classes):
                    class_name = results[0].names[int(cls_id)]
                    
                    detection = {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "class": class_name,
                        "confidence": float(conf)
                    }
                    detections.append(detection)
            
            return detections
        else:
            # Use mock detection
            return self.mock_detect_objects(image)
    
    def extract_agent_name(self, detection_class: str) -> Optional[str]:
        """Extract agent name from detection class"""
        detection_lower = detection_class.lower()
        
        for agent in self.agent_names:
            if agent in detection_lower:
                return agent.capitalize()
        
        return None
    
    def find_associated_lock_icon(self, highlighted_box: BoundingBox, lock_icons: List[BoundingBox], 
                                 proximity_threshold: float = 0.1) -> Optional[BoundingBox]:
        """Find lock icon associated with highlighted agent"""
        best_match = None
        best_distance = float('inf')
        
        highlight_center = highlighted_box.center()
        
        for lock_icon in lock_icons:
            lock_center = lock_icon.center()
            distance = ((highlight_center[0] - lock_center[0]) ** 2 + 
                       (highlight_center[1] - lock_center[1]) ** 2) ** 0.5
            
            # Check if lock icon is within reasonable distance
            max_distance = min(highlighted_box.area() ** 0.5 * 0.5, 100)
            
            if distance < max_distance and distance < best_distance:
                best_match = lock_icon
                best_distance = distance
        
        return best_match
    
    def assign_detection_to_player(self, box: BoundingBox, existing_assignments: Dict) -> str:
        """Assign detection to player based on position"""
        box_center = box.center()
        
        # Simple assignment based on vertical position
        # Assuming players are arranged vertically in selection screen
        player_index = min(max(1, int(box_center[1] // 120) + 1), 5)
        return f"player{player_index}"
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze a single image for Valorant agent selection events
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing analysis results
        """
        # Load image
        if not os.path.exists(image_path):
            return {"error": f"Image file {image_path} not found"}
        
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not load image {image_path}"}
        
        # Detect objects
        detections = self.detect_objects(image)
        
        # Parse detections into BoundingBox objects
        boxes = []
        for detection in detections:
            box = BoundingBox(
                x1=detection['x1'],
                y1=detection['y1'],
                x2=detection['x2'],
                y2=detection['y2'],
                class_name=detection['class'],
                confidence=detection['confidence']
            )
            boxes.append(box)
        
        # Separate detection types
        highlighted_agents = [box for box in boxes if "highlighted_agent" in box.class_name or "agent_" in box.class_name]
        lock_icons = [box for box in boxes if "locked_icon" in box.class_name]
        agent_cards = [box for box in boxes if "agent_card" in box.class_name]
        
        # Initialize player statuses
        player_statuses = {}
        team_composition = []
        events = []
        
        # Process highlighted agents
        for highlighted_box in highlighted_agents:
            agent_name = self.extract_agent_name(highlighted_box.class_name)
            if not agent_name:
                continue
            
            player_id = self.assign_detection_to_player(highlighted_box, player_statuses)
            
            # Check for associated lock icon
            associated_lock = self.find_associated_lock_icon(highlighted_box, lock_icons)
            
            if associated_lock:
                # Player has locked in
                player_statuses[player_id] = {
                    "state": "locked",
                    "agent": agent_name
                }
                team_composition.append(agent_name)
                events.append(f"{player_id}_locked_{agent_name}")
            else:
                # Player is hovering
                player_statuses[player_id] = {
                    "state": "hovering", 
                    "agent": agent_name
                }
                events.append(f"{player_id}_hovering_{agent_name}")
        
        # Fill in remaining players
        for i in range(1, 6):
            player_id = f"player{i}"
            if player_id not in player_statuses:
                player_statuses[player_id] = {
                    "state": "not_selected",
                    "agent": None
                }
        
        # Build result
        result = {
            "image_path": image_path,
            "phase": "agent_selection",
            "player_status": player_statuses,
            "team_composition": team_composition,
            "events": events,
            "detection_summary": {
                "total_detections": len(detections),
                "highlighted_agents": len(highlighted_agents),
                "lock_icons": len(lock_icons),
                "agent_cards": len(agent_cards)
            },
            "raw_detections": detections
        }
        
        return result
    
    def visualize_detections(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Visualize detections on the image
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            
        Returns:
            Annotated image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image {image_path}")
            return None
        
        detections = self.detect_objects(image)
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])
            conf = detection['confidence']
            class_name = detection['class']
            
            # Color based on detection type
            if "highlighted_agent" in class_name:
                color = (0, 255, 0)  # Green
            elif "locked_icon" in class_name:
                color = (0, 0, 255)  # Red
            elif "agent_card" in class_name:
                color = (255, 0, 0)  # Blue
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to {output_path}")
        
        return image

def main():
    parser = argparse.ArgumentParser(description="Analyze Valorant agent selection from image")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model", help="Path to YOLO model file (.pt)")
    parser.add_argument("--output", help="Path to save JSON results")
    parser.add_argument("--visualize", help="Path to save annotated image")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ValorantImageAnalyzer(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Analyze image
    print(f"Analyzing image: {args.image_path}")
    result = analyzer.analyze_image(args.image_path)
    
    # Print results
    print("\n" + "="*50)
    print("VALORANT AGENT SELECTION ANALYSIS")
    print("="*50)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Phase: {result['phase']}")
    print(f"Events detected: {result['events']}")
    print(f"Team composition: {result['team_composition']}")
    
    print("\nPlayer Status:")
    for player_id, status in result['player_status'].items():
        agent = status['agent'] or 'None'
        print(f"  {player_id}: {status['state']} - {agent}")
    
    print(f"\nDetection Summary:")
    summary = result['detection_summary']
    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Highlighted agents: {summary['highlighted_agents']}")
    print(f"  Lock icons: {summary['lock_icons']}")
    print(f"  Agent cards: {summary['agent_cards']}")
    
    # Save JSON output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Create visualization
    if args.visualize:
        analyzer.visualize_detections(args.image_path, args.visualize)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()