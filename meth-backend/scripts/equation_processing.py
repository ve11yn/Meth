import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import re

class HandwrittenEquationProcessor:
    def __init__(self, model_path=None, label_encoder_path=None):
        """
        Initialize the equation processor
        """
        self.model = None
        self.label_encoder = None
        
        if model_path:
            self.model = load_model(model_path)
        if label_encoder_path:
            self.label_encoder = joblib.load(label_encoder_path)
        
        # Label to symbol mapping - exactly matching your trained model
        self.label_to_symbol = {
            # Numbers (labels 0-9)
            '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
            '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
            # Operators (labels 10-13)
            '10': '+',   # Addition
            '11': '/',   # Division  
            '12': '*',   # Multiplication
            '13': '-',   # Subtraction
            'add': '+',  # Addition (if label encoder uses 'add')
            'sub': '-',  # Subtraction (if label encoder uses 'sub')
            'mul': '*',  # Multiplication (if label encoder uses 'mul')
            'div': '/',  # Division (if label encoder uses 'div')
        }

    # =============================================================================
    # IMAGE PREPROCESSING SECTION - ENHANCED FOR OPERATORS
    # =============================================================================
    
    def preprocess_equation_image(self, image_path):
        """Load and preprocess the handwritten equation image - less aggressive noise removal"""
        print("1. Loading and preprocessing image...")
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize if too large
        height, width = img.shape
        max_dimension = 1200
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Simple preprocessing - less aggressive
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Use Otsu's thresholding
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Minimal morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Very minimal noise removal - only remove tiny specs
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        min_area = 2  # Reduced from 3 - don't remove minus signs!
        
        removed_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                img[labels == i] = 0
                removed_count += 1
        
        print(f"   Removed {removed_count} tiny noise components (area < {min_area})")
        
        print(f"   Preprocessed image shape: {img.shape}")
        return img
    
    # =============================================================================
    # LINE SEGMENTATION SECTION - IMPROVED
    # =============================================================================
    
    def segment_equation_lines(self, img):
        """Segment the equation into individual lines with better handling"""
        print("2. Segmenting equation lines...")
        
        horizontal_projection = np.sum(img, axis=1)
        
        # Use a more adaptive threshold
        threshold = max(5, np.max(horizontal_projection) * 0.03)  # Lowered threshold
        
        line_boundaries = []
        in_line = False
        start = 0
        
        for i, proj in enumerate(horizontal_projection):
            if proj > threshold and not in_line:
                start = i
                in_line = True
            elif proj <= threshold and in_line:
                if i - start > 3:  # Minimum line height reduced
                    line_boundaries.append((start, i))
                in_line = False
        
        if in_line and len(horizontal_projection) - start > 3:
            line_boundaries.append((start, len(horizontal_projection)))
        
        # If no lines found, treat entire image as one line
        if not line_boundaries:
            line_boundaries = [(0, img.shape[0])]
        
        line_images = []
        for start, end in line_boundaries:
            line_img = img[start:end, :]
            if line_img.shape[0] > 2 and line_img.shape[1] > 2:
                line_images.append(line_img)
        
        print(f"   Found {len(line_images)} lines")
        return line_images, line_boundaries
    
    # =============================================================================
    # CHARACTER SEGMENTATION SECTION - ENHANCED FOR OPERATORS
    # =============================================================================
    
    def segment_characters_in_line(self, line_img):
        """Segment individual characters/symbols with enhanced operator detection"""
        print("   Segmenting characters in line...")
        
        # Find contours
        contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        line_height = line_img.shape[0]
        line_width = line_img.shape[1]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # More lenient filtering - especially for operators like minus signs
            min_width = 2      # Reduced from 3 for thin operators
            min_height = 2     # Reduced from 5 for short operators like minus
            min_area = 4       # Reduced from 10 for small operators
            max_area = line_width * line_height * 0.95
            
            if (w >= min_width and h >= min_height and 
                area >= min_area and area <= max_area):
                
                # Much more lenient aspect ratio - especially for minus signs
                aspect_ratio = w / h
                if 0.02 <= aspect_ratio <= 50.0:  # Increased upper limit for wide operators
                    bounding_boxes.append((x, y, w, h, area))
        # Enhanced detection for horizontal lines (minus signs, equals signs)
        # Method 1: Use horizontal projection
        horizontal_projection = np.sum(line_img, axis=0)
        vertical_projection = np.sum(line_img, axis=1)
        
        # Method 2: Morphological detection of horizontal lines
        if line_img.shape[1] > 10:  # Only if line is wide enough
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, line_img.shape[1]//20), 1))
            horizontal_lines = cv2.morphologyEx(line_img, cv2.MORPH_OPEN, h_kernel)
            
            # Find contours in the horizontal lines image
            h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in h_contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Very lenient criteria for horizontal operators
                if w >= 3 and h >= 1 and area >= 3:
                    aspect_ratio = w / h
                    if aspect_ratio >= 2.0:  # Must be wider than tall
                        # Check if this overlaps with existing boxes
                        overlaps_existing = False
                        for existing_box in bounding_boxes:
                            ex_x, ex_y, ex_w, ex_h, ex_area = existing_box
                            overlap_x = max(0, min(x + w, ex_x + ex_w) - max(x, ex_x))
                            overlap_y = max(0, min(y + h, ex_y + ex_h) - max(y, ex_y))
                            if overlap_x > w * 0.3 and overlap_y > 0:  # 30% horizontal overlap
                                overlaps_existing = True
                                break
                        
                        if not overlaps_existing:
                            bounding_boxes.append((x, y, w, h, area))
                            print(f"     Detected horizontal operator at x={x}, y={y}, w={w}, h={h}")
        
        # Method 3: Row-by-row horizontal line detection for very thin lines
        line_threshold = max(3, np.max(vertical_projection) * 0.05) if np.max(vertical_projection) > 0 else 3
        
        for row in range(line_img.shape[0]):
            row_pixels = line_img[row, :]
            if np.sum(row_pixels) >= line_threshold:  # This row has enough pixels
                # Find continuous segments in this row
                segments = []
                in_segment = False
                start_col = 0
                
                for col in range(len(row_pixels)):
                    if row_pixels[col] > 0 and not in_segment:
                        start_col = col
                        in_segment = True
                    elif row_pixels[col] == 0 and in_segment:
                        segment_width = col - start_col
                        if segment_width >= 4:  # Minimum width for operator
                            segments.append((start_col, row, segment_width, 1))
                        in_segment = False
                
                # Handle case where segment goes to end of row
                if in_segment:
                    segment_width = len(row_pixels) - start_col
                    if segment_width >= 4:
                        segments.append((start_col, row, segment_width, 1))
                
                # Add segments that don't overlap with existing boxes
                for seg_x, seg_y, seg_w, seg_h in segments:
                    overlaps_existing = False
                    for existing_box in bounding_boxes:
                        ex_x, ex_y, ex_w, ex_h, ex_area = existing_box
                        overlap_x = max(0, min(seg_x + seg_w, ex_x + ex_w) - max(seg_x, ex_x))
                        overlap_y = max(0, min(seg_y + seg_h, ex_y + ex_h) - max(seg_y, ex_y))
                        if overlap_x > seg_w * 0.5:  # 50% horizontal overlap
                            overlaps_existing = True
                            break
                    
                    if not overlaps_existing:
                        bounding_boxes.append((seg_x, seg_y, seg_w, seg_h, seg_w * seg_h))
                        print(f"     Detected thin horizontal line at row {seg_y}, x={seg_x}, width={seg_w}")
        
        # Sort by x-coordinate (left to right)
        bounding_boxes.sort(key=lambda box: box[0])
        
        print(f"     Found {len(bounding_boxes)} potential characters before overlap removal")
        for i, (x, y, w, h, area) in enumerate(bounding_boxes):
            aspect_ratio = w / h
            print(f"       Box {i}: x={x}, y={y}, w={w}, h={h}, area={area:.1f}, aspect={aspect_ratio:.2f}")
        
        # Simple overlap removal - less aggressive
        filtered_boxes = []
        for box in bounding_boxes:
            x1, y1, w1, h1, area1 = box
            overlaps = False
            
            for other_box in filtered_boxes:
                x2, y2, w2, h2, area2 = other_box
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                box1_area = w1 * h1
                box2_area = w2 * h2
                
                # Only remove if significant overlap (50%+)
                if overlap_area > 0.5 * min(box1_area, box2_area):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_boxes.append(box)
        
        character_images = []
        character_positions = []
        
        for x, y, w, h, area in filtered_boxes:
            # Simple padding
            padding = 3
            
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(line_img.shape[1], x + w + padding)
            y_end = min(line_img.shape[0], y + h + padding)
            
            char_img = line_img[y_start:y_end, x_start:x_end]
            
            # Simple white pixel check
            white_pixels = np.sum(char_img > 0)
            if white_pixels >= 5:  # Very low threshold
                char_img = self.pad_and_resize_character(char_img)
                character_images.append(char_img)
                character_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))
        
        print(f"     Found {len(character_images)} characters")
        return character_images, character_positions
    
    def remove_overlapping_boxes(self, bounding_boxes):
        """Improved overlap removal"""
        filtered_boxes = []
        
        for i, box in enumerate(bounding_boxes):
            x1, y1, w1, h1, area1 = box
            overlaps = False
            
            for j, other_box in enumerate(filtered_boxes):
                x2, y2, w2, h2, area2 = other_box
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                box1_area = w1 * h1
                box2_area = w2 * h2
                
                # More conservative overlap threshold
                overlap_threshold = 0.3  # Reduced from 0.5
                
                if overlap_area > overlap_threshold * min(box1_area, box2_area):
                    if area1 > area2:
                        filtered_boxes[j] = box
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_boxes.append(box)
        
        return filtered_boxes
    
    def enhance_character_image(self, char_img):
        """Enhance individual character image for better recognition"""
        # Apply slight dilation to make thin strokes more visible
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        char_img = cv2.dilate(char_img, kernel, iterations=1)
        
        # Ensure the image has good contrast
        char_img = cv2.normalize(char_img, None, 0, 255, cv2.NORM_MINMAX)
        
        return char_img
    
    def pad_and_resize_character(self, char_img, target_size=(32, 32), preserve_aspect=True):
        """Enhanced padding and resizing for better recognition"""
        h, w = char_img.shape
        target_h, target_w = target_size
        
        # More generous border for operators
        border_size = max(6, max(h, w) // 8)
        char_img = np.pad(char_img, border_size, 'constant', constant_values=0)
        h, w = char_img.shape
        
        if preserve_aspect:
            scale_h = target_h / h
            scale_w = target_w / w
            scale = min(scale_h, scale_w)
            
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            if new_h > 0 and new_w > 0:
                char_img = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                pad_h = target_h - new_h
                pad_w = target_w - new_w
                
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                char_img = np.pad(char_img, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                                 'constant', constant_values=0)
            else:
                char_img = np.zeros(target_size, dtype=np.uint8)
        else:
            char_img = cv2.resize(char_img, target_size, interpolation=cv2.INTER_AREA)
        
        return char_img
    
    # =============================================================================
    # CHARACTER RECOGNITION SECTION - ENHANCED
    # =============================================================================
    
    def recognize_characters(self, character_images):
        """Enhanced character recognition with confidence filtering"""
        print("3. Recognizing characters...")
        
        if self.model is None or self.label_encoder is None:
            print("   Warning: No model or label encoder loaded!")
            return ["?" for _ in character_images], [0.0 for _ in character_images]
        
        predictions = []
        confidences = []
        
        # Confidence threshold - ignore characters below this
        confidence_threshold = 0.4  # Increased from 0.2 to filter out more noise
        
        for i, char_img in enumerate(character_images):
            # Normalize the image
            char_img_norm = char_img.astype(np.float32) / 255.0
            
            # Add batch and channel dimensions
            if len(char_img_norm.shape) == 2:
                char_img_norm = np.expand_dims(char_img_norm, axis=-1)
            char_img_norm = np.expand_dims(char_img_norm, axis=0)
            
            try:
                # Get prediction
                pred = self.model.predict(char_img_norm, verbose=0)
                predicted_class = np.argmax(pred)
                confidence = np.max(pred)
                
                if confidence > confidence_threshold:
                    # Get the label from label encoder
                    predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
                    predicted_label = str(predicted_label)
                    
                    # Convert label to symbol using our mapping
                    if predicted_label in self.label_to_symbol:
                        predicted_symbol = self.label_to_symbol[predicted_label]
                    else:
                        predicted_symbol = predicted_label
                    
                    print(f"     Character {i+1}: '{predicted_symbol}' (confidence: {confidence:.3f})")
                    predictions.append(predicted_symbol)
                    confidences.append(float(confidence))
                else:
                    print(f"     Character {i+1}: IGNORED (low confidence: {confidence:.3f})")
                    # Skip this character entirely - don't add to predictions
                    continue
                
            except Exception as e:
                print(f"     Error recognizing character {i+1}: {e}")
                # Skip this character entirely
                continue
        
        print(f"   Recognized {len(predictions)} characters (filtered out low confidence)")
        return predictions, confidences
    
    # =============================================================================
    # POST-PROCESSING SECTION
    # =============================================================================
    
    def post_process_equation(self, equation_string):
        """Post-process the recognized equation to fix common errors"""
        print(f"   Post-processing equation: '{equation_string}'")
        
        original_equation = equation_string
        
        # Remove spaces and newlines
        equation_string = equation_string.replace(' ', '').replace('\n', '')
        print(f"   After removing spaces: '{equation_string}'")
        
        # Common corrections (only if needed)
        corrections = {
            'x': '*',  # x often recognized instead of *
            '×': '*',  # multiplication symbol
            '÷': '/',  # division symbol
            '−': '-',  # minus symbol variants
            '–': '-',
            '—': '-',
        }
        
        for old, new in corrections.items():
            if old in equation_string:
                equation_string = equation_string.replace(old, new)
                print(f"   Corrected '{old}' to '{new}': '{equation_string}'")
        
        # Only remove consecutive question marks (keep single ones for now)
        equation_string = re.sub(r'\?{2,}', '?', equation_string)
        print(f"   After removing consecutive '?': '{equation_string}'")
        
        print(f"   Final processed equation: '{equation_string}'")
        return equation_string
    
    # =============================================================================
    # VISUALIZATION SECTION - ENHANCED
    # =============================================================================
    
    def visualize_process(self, image_path, results, img, line_boundaries):
        """Create visualization of the processing steps"""
        print("4. Creating visualization...")

        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Handwritten Equation Processing Steps', fontsize=16)

        # 1. Original Image
        axes[0, 0].imshow(original_img, cmap='gray')
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')

        # 2. Preprocessed Image
        axes[0, 1].imshow(img, cmap='gray')
        axes[0, 1].set_title('2. Preprocessed Image (Binary)')
        axes[0, 1].axis('off')

        # 3. Line Segmentation
        line_img = img.copy()
        for i, (start, end) in enumerate(line_boundaries):
            cv2.line(line_img, (0, start), (line_img.shape[1], start), 128, 2)
            cv2.line(line_img, (0, end), (line_img.shape[1], end), 128, 2)
        axes[1, 0].imshow(line_img, cmap='gray')
        axes[1, 0].set_title('3. Line Segmentation')
        axes[1, 0].axis('off')

        # 4. Character Bounding Boxes with labels
        bbox_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        y_offset = 0
        
        for line_idx, line_result in enumerate(results):
            if line_idx < len(line_boundaries):
                y_offset = line_boundaries[line_idx][0]
            
            for pos, char, conf in zip(line_result['positions'], 
                                     line_result['characters'], 
                                     line_result['confidences']):
                x, y, w, h = pos
                y_adjusted = y + y_offset
                
                # Color code by confidence
                color = (0, 255, 0) if conf > 0.5 else (255, 255, 0) if conf > 0.2 else (255, 0, 0)
                cv2.rectangle(bbox_img, (x, y_adjusted), (x + w, y_adjusted + h), color, 2)
                
                # Add text label
                cv2.putText(bbox_img, f"{char}", (x, y_adjusted - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        axes[1, 1].imshow(bbox_img)
        axes[1, 1].set_title('4. Character Detection & Recognition')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig("processing_visualization.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Show individual characters
        self.show_recognized_characters(results)

    def show_recognized_characters(self, results):
        """Display individual recognized characters with enhanced info"""
        print("5. Displaying recognized characters...")
        
        for line_idx, line_result in enumerate(results):
            if len(line_result['character_images']) == 0:
                continue
                
            n_chars = len(line_result['character_images'])
            fig, axes = plt.subplots(1, n_chars, figsize=(2 * n_chars, 3))
            fig.suptitle(f'Line {line_idx + 1} - Recognized Characters', fontsize=14)
            
            if n_chars == 1:
                axes = [axes]
            
            for i, (char_img, char, conf) in enumerate(zip(
                line_result['character_images'], 
                line_result['characters'], 
                line_result['confidences']
            )):
                axes[i].imshow(char_img, cmap='gray')
                
                # Color code title by confidence
                title_color = 'green' if conf > 0.5 else 'orange' if conf > 0.2 else 'red'
                axes[i].set_title(f"'{char}'\n{conf:.3f}", color=title_color)
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"characters_line_{line_idx + 1}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    # =============================================================================
    # CALCULATION SECTION
    # =============================================================================
    
    def calculate_simple(self, equation_string):
        """Enhanced calculator with better error handling"""
        print("6. Calculating result...")
        
        print(f"   Original equation: '{equation_string}'")
        
        # Post-process the equation
        processed_equation = self.post_process_equation(equation_string)
        
        print(f"   Processed equation: '{processed_equation}'")
        
        if not processed_equation or processed_equation.strip() == '':
            return "No valid equation found - equation is empty after processing"
        
        # Check if equation is just question marks
        if processed_equation.replace('?', '').strip() == '':
            return f"No valid equation found - only unrecognized characters: '{equation_string}'"
        
        try:
            # Handle equations with =
            if '=' in processed_equation:
                print("   This is an equation, showing both sides:")
                parts = processed_equation.split('=')
                if len(parts) == 2:
                    left, right = parts
                    left = left.strip()
                    right = right.strip()
                    
                    left_result = "Invalid"
                    right_result = "Invalid"
                    
                    if left and '?' not in left:
                        try:
                            left_result = eval(left)
                        except Exception as e:
                            print(f"   Error evaluating left side '{left}': {e}")
                    
                    if right and '?' not in right:
                        try:
                            right_result = eval(right)
                        except Exception as e:
                            print(f"   Error evaluating right side '{right}': {e}")
                    
                    return f"{left} = {left_result}, {right} = {right_result}"
                else:
                    return f"Complex equation: {processed_equation}"
            else:
                # Simple expression evaluation
                if '?' in processed_equation:
                    return f"Incomplete equation: {processed_equation} (contains unrecognized characters)"
                
                print(f"   Evaluating: {processed_equation}")
                result = eval(processed_equation)
                return f"{processed_equation} = {result}"
                
        except Exception as e:
            return f"Cannot calculate: {processed_equation} (Error: {str(e)})"
    
    # =============================================================================
    # MAIN PROCESSING PIPELINE
    # =============================================================================
    
    def process_and_calculate(self, image_path, show_visualization=True):
        """Complete pipeline with enhanced processing"""
        print("=" * 60)
        print("ENHANCED HANDWRITTEN EQUATION PROCESSOR")
        print("=" * 60)
        
        try:
            # Process the equation
            results, img = self.process_equation(image_path)
            
            # Convert to string
            equation_string = self.equation_to_string(results)
            print(f"\nRecognized: '{equation_string}'")
            
            # Calculate result
            result = self.calculate_simple(equation_string)
            print(f"Result: {result}")
            
            # Show visualization if requested
            if show_visualization:
                line_images, line_boundaries = self.segment_equation_lines(img)
                self.visualize_process(image_path, results, img, line_boundaries)
            
            return {
                'recognized': equation_string,
                'result': result,
                'details': results,
                'success': True
            }
            
        except Exception as e:
            print(f"Error processing equation: {e}")
            return {
                'recognized': '',
                'result': f'Error: {str(e)}',
                'details': [],
                'success': False
            }
    
    def process_equation(self, image_path):
        """Process handwritten equation with enhanced steps"""
        img = self.preprocess_equation_image(image_path)
        line_images, line_boundaries = self.segment_equation_lines(img)
        
        equation_results = []
        
        for i, line_img in enumerate(line_images):
            print(f"   Processing line {i+1}/{len(line_images)}")
            
            char_images, char_positions = self.segment_characters_in_line(line_img)
            
            if len(char_images) > 0:
                predictions, confidences = self.recognize_characters(char_images)
            else:
                predictions = []
                confidences = []
                print(f"     No characters found in line {i+1}")
            
            line_result = {
                'line_number': i,
                'characters': predictions,
                'confidences': confidences,
                'positions': char_positions,
                'character_images': char_images
            }
            
            equation_results.append(line_result)
        
        return equation_results, img
    
    def equation_to_string(self, results):
        """Convert results to string with better formatting and debugging"""
        print("   Converting recognition results to equation string...")
        
        equation_lines = []
        
        for line_idx, line_result in enumerate(results):
            print(f"   Line {line_idx + 1}: Found {len(line_result['characters'])} characters")
            
            if len(line_result['characters']) > 0:
                char_strings = [str(char) for char in line_result['characters']]
                line_string = ''.join(char_strings)
                print(f"     Line {line_idx + 1} string: '{line_string}'")
                equation_lines.append(line_string)
            else:
                print(f"     Line {line_idx + 1}: No characters recognized")
        
        final_equation = ' '.join(equation_lines) if equation_lines else ''
        print(f"   Final equation string: '{final_equation}'")
        
        return final_equation


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

