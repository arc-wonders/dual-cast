import cv2
import numpy as np
import re
import os
import json

# Note: pytesseract import is commented out to avoid dependency issues
# Uncomment when you have it properly installed
# import pytesseract

class ValorantOwnedBoxDetector:
    def __init__(self, image_path):
        """
        Initialize the detector with an image path
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        self.height, self.width = self.image.shape[:2]
        
    def detect_owned_regions(self):
        """
        Detect regions with the green/teal color used for OWNED items in Valorant
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Define green/teal color range for Valorant OWNED boxes
        # Based on the image, OWNED boxes have a green/teal color
        # Hue: 75-95 (green-cyan range)
        # Saturation: 40-255 (avoid very pale colors)
        # Value: 60-255 (avoid very dark regions)
        lower_owned = np.array([75, 40, 60])
        upper_owned = np.array([95, 255, 255])
        
        # Create mask for owned regions
        owned_mask = cv2.inRange(hsv, lower_owned, upper_owned)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        owned_mask = cv2.morphologyEx(owned_mask, cv2.MORPH_CLOSE, kernel)
        owned_mask = cv2.morphologyEx(owned_mask, cv2.MORPH_OPEN, kernel)
        
        return owned_mask
    
    def find_owned_boxes(self, mask):
        """
        Find bounding boxes around owned regions
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area - Valorant weapon/item boxes are typically medium-sized
            if 800 < area < 25000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by dimensions - Valorant item boxes have specific proportions
                if w > 50 and h > 30 and w < 500 and h < 200:
                    # Expand the box slightly to capture full content
                    padding = 3
                    x_expanded = max(0, x - padding)
                    y_expanded = max(0, y - padding)
                    w_expanded = min(self.width - x_expanded, w + 2 * padding)
                    h_expanded = min(self.height - y_expanded, h + 2 * padding)
                    
                    boxes.append([x_expanded, y_expanded, w_expanded, h_expanded])
        
        return boxes
    
    def preprocess_roi_for_ocr(self, roi):
        """
        Preprocess region of interest for better OCR results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        preprocessed = []
        
        # Method 1: Original grayscale
        preprocessed.append(gray)
        
        # Method 2: Binary threshold (for white text on green/teal background)
        _, thresh1 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        preprocessed.append(thresh1)
        
        # Method 3: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        preprocessed.append(adaptive)
        
        # Method 4: OTSU threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(otsu)
        
        return preprocessed
    
    def extract_text_with_ocr(self, roi):
        """
        Extract text from ROI using OCR with multiple preprocessing attempts
        """
        try:
            # Uncomment when pytesseract is available
            """
            preprocessed_images = self.preprocess_roi_for_ocr(roi)
            
            all_texts = []
            
            for i, processed_img in enumerate(preprocessed_images):
                try:
                    # Configure pytesseract for better text recognition
                    config = '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
                    
                    text = pytesseract.image_to_string(processed_img, config=config).strip()
                    
                    # Clean up the text
                    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
                    text = ' '.join(text.split())  # Remove extra whitespace
                    text = text.upper()
                    
                    if text and len(text) > 0:
                        all_texts.append(text)
                        
                except Exception as e:
                    continue
            
            # Return the text that contains "OWNED" or the longest text found
            for text in all_texts:
                if "OWNED" in text.upper():
                    return text
            
            if all_texts:
                return max(all_texts, key=len)
            """
            
            # Fallback method when pytesseract is not available
            return self.fallback_text_detection(roi)
            
        except Exception as e:
            return self.fallback_text_detection(roi)
    
    def fallback_text_detection(self, roi):
        """
        Fallback text detection based on visual analysis for owned boxes
        Returns generic text when OCR is not available
        """
        # Convert to grayscale and HSV
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check for green/teal background (Valorant OWNED color)
        owned_pixels = cv2.countNonZero(cv2.inRange(hsv_roi, 
                                                  np.array([75, 40, 60]), 
                                                  np.array([95, 255, 255])))
        total_pixels = roi.shape[0] * roi.shape[1]
        owned_ratio = owned_pixels / total_pixels
        
        # Check for bright text (white text on green/teal background)
        _, bright_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        bright_pixels = cv2.countNonZero(bright_mask)
        bright_ratio = bright_pixels / total_pixels
        
        # Return generic text detection for owned boxes
        if owned_ratio > 0.3 and bright_ratio > 0.03:
            return "TEXT_DETECTED"
        elif owned_ratio > 0.5:  # Strong green/teal presence
            return "TEXT_DETECTED"
        
        return "OWNED_BOX"
    
    def check_contains_owned(self, text):
        """
        Check if the extracted text contains "OWNED"
        """
        if not text:
            return False
        
        # Check for "OWNED" in various forms
        text_upper = text.upper()
        owned_variations = ["OWNED", "0WNED", "OWNE0", "0WNE0", "DWNED"]  # Common OCR mistakes
        
        for variation in owned_variations:
            if variation in text_upper:
                return True
        
        return False
    
    def detect_owned_boxes(self, debug=False):
        """
        Main function to detect owned boxes and extract all text from them
        """
        results = []
        
        # Step 1: Find owned regions
        owned_mask = self.detect_owned_regions()
        boxes = self.find_owned_boxes(owned_mask)
        
        print(f"Found {len(boxes)} owned-colored boxes to analyze")
        
        # Step 2: Extract text from all owned boxes
        for i, box in enumerate(boxes):
            x, y, w, h = box
            
            # Extract ROI
            roi = self.image[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
            
            if debug:
                cv2.imwrite(f"debug_owned_box_{i}.jpg", roi)
            
            # Perform OCR extraction first
            extracted_text = self.extract_text_with_ocr(roi)
            
            if debug:
                print(f"Owned Box {i}: OCR result: '{extracted_text}'")
            
            # Add all text found in owned boxes (not just OWNED)
            if extracted_text and extracted_text.strip():
                results.append({
                    "label": extracted_text,
                    "box": [x, y, w, h]
                })
                
                if debug:
                    print(f"✅ Owned Box {i} text extracted: '{extracted_text}'")
            elif debug:
                print(f"❌ Owned Box {i} no text extracted")
        
        return results
    
    def visualize_all_owned_boxes(self, output_path="all_owned_boxes.jpg"):
        """
        Visualize all detected owned-colored boxes for debugging
        """
        owned_mask = self.detect_owned_regions()
        boxes = self.find_owned_boxes(owned_mask)
        
        result_image = self.image.copy()
        
        for i, box in enumerate(boxes):
            x, y, w, h = box
            
            # Draw bounding box in cyan for all owned-colored boxes
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            
            # Label each box
            cv2.putText(result_image, f"Owned {i}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imwrite(output_path, result_image)
        print(f"All owned-colored boxes visualization saved to: {output_path}")
        
        return result_image

    def get_owned_boxes_json(self, debug=False):
        """
        Get all owned boxes text in JSON format
        """
        results = self.detect_owned_boxes(debug=debug)
        
        json_output = {
            "total_owned_boxes": len(results),
            "owned_boxes": []
        }
        
        for i, result in enumerate(results):
            json_output["owned_boxes"].append({
                "box_id": i + 1,
                "extracted_text": result['label'],
                "bounding_box": {
                    "x": result['box'][0],
                    "y": result['box'][1],
                    "width": result['box'][2],
                    "height": result['box'][3]
                }
            })
        
        return json_output
    
    def get_owned_boxes_json(self, debug=False):
        """
        Get all owned boxes text in JSON format
        """
        results = self.detect_owned_boxes(debug=debug)
        
        json_output = {
            "total_owned_boxes": len(results),
            "owned_boxes": []
        }
        
        for i, result in enumerate(results):
            json_output["owned_boxes"].append({
                "box_id": i + 1,
                "extracted_text": result['label'],
                "bounding_box": {
                    "x": result['box'][0],
                    "y": result['box'][1],
                    "width": result['box'][2],
                    "height": result['box'][3]
                }
            })
        
        return json_output
        """
        Visualize all owned boxes with their extracted text
        """
        result_image = self.image.copy()
        
        for i, result in enumerate(results):
            x, y, w, h = result["box"]
            label = result["label"]
            
            # Draw bounding box in green for owned boxes
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Draw label above the box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Draw background rectangle for text
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(result_image, (x, y - text_height - 10), 
                         (x + text_width, y), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(result_image, label, (x, y - 5), 
                       font, font_scale, (0, 0, 0), thickness)
        
        cv2.imwrite(output_path, result_image)
        print(f"Owned boxes with text visualization saved to: {output_path}")
        
        return result_image

def main():
    """
    Example usage
    """
    image_path = "D:/dual cast/buy/buy6.jpg"  # Update this path
    
    try:
        # Initialize detector - FIXED: Use correct class name
        detector = ValorantOwnedBoxDetector(image_path)
        
        # Optional: Visualize all owned boxes for debugging
        detector.visualize_all_owned_boxes()
        
        # Detect owned boxes containing OWNED text - FIXED: Use correct method name
        results = detector.detect_owned_boxes(debug=True)
        
        # Print results in both text and JSON format
        print("\n=== ALL TEXT IN OWNED BOXES ===")
        if results:
            for i, result in enumerate(results):
                print(f"Owned Box {i+1}:")
                print(f"  Text: {result['label']}")
                print(f"  Box: {result['box']} (x, y, width, height)")
                print()
        else:
            print("No text found in owned boxes.")
        
        # Output results as JSON
        print("\n=== JSON OUTPUT ===")
        json_output = {
            "total_owned_boxes": len(results),
            "owned_boxes": []
        }
        
        for i, result in enumerate(results):
            json_output["owned_boxes"].append({
                "box_id": i + 1,
                "extracted_text": result['label'],
                "bounding_box": {
                    "x": result['box'][0],
                    "y": result['box'][1],
                    "width": result['box'][2],
                    "height": result['box'][3]
                }
            })
        
        # Pretty print JSON
        print(json.dumps(json_output, indent=2))
        
        # Save JSON to file
        with open("owned_boxes_text.json", "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\nJSON results saved to: owned_boxes_text.json")
        
        # Create visualization for all owned boxes with text
        if results:
            detector.visualize_owned_detections(results)
        
        # Return results in the requested format
        return json_output
        
    except Exception as e:
        print(f"Error: {e}")
        error_output = {
            "total_owned_boxes": 0,
            "owned_boxes": [],
            "error": str(e)
        }
        return error_output

if __name__ == "__main__":
    main()

# Example usage:
# detector = ValorantOwnedBoxDetector("your_screenshot.jpg")
# results = detector.detect_owned_boxes()
# json_results = detector.get_owned_boxes_json()
# print(json.dumps(json_results, indent=2))