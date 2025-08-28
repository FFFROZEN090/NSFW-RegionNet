"""
Generate SAM2 prompts from detected keypoints with intelligent validation.
"""

from typing import Tuple, List, Optional
import numpy as np
import cv2

from ..data_models import DetectionResult, Keypoint


class PromptGenerator:
    """Generate SAM2 prompts from detected keypoints for skin segmentation."""
    
    
    @staticmethod
    def generate_prompts(
        detection: DetectionResult, 
        image_shape: Optional[Tuple[int, int]] = None,
        all_detections: Optional[List[DetectionResult]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate positive and negative prompts from keypoints.
        Strategy: 1 positive point (eyes+nose triangle center) + 1 negative point (clothing center).
        
        Args:
            detection: Current detection result containing keypoints
            image_shape: Shape of the image (height, width) - optional
            all_detections: List of all detections (for context)
            
        Returns:
            Tuple of (points, labels) where:
            - points: np.ndarray of shape (N, 2) with (x, y) coordinates
            - labels: np.ndarray of shape (N,) with 1 for positive, 0 for negative
        """
        points = []
        labels = []
        
        # 1. Get positive point (eyes + nose triangle center)
        face_center = PromptGenerator._get_face_triangle_center(detection)
        if face_center:
            points.append(face_center)
            labels.append(1)
        
        # 2. Get negative point (clothing center)
        clothing_point = PromptGenerator._get_clothing_center(detection)
        if clothing_point:
            points.append(clothing_point)
            labels.append(0)
        
        # 3. Fallback if no valid points generated
        if not points:
            # Emergency fallback: use center of bounding box
            x1, y1, x2, y2 = detection.bbox
            fallback_point = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            points.append(fallback_point)
            labels.append(1)
        
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)
    
    @staticmethod
    def _get_face_triangle_center(detection: DetectionResult) -> Optional[List[int]]:
        """
        Calculate the center of triangle formed by TWO EYES and NOSE.
        
        Args:
            detection: Detection result containing keypoints
            
        Returns:
            [x, y] coordinates of face triangle center or None
        """
        nose = detection.get_keypoint_by_label('nose')
        left_eye = detection.get_keypoint_by_label('left_eye')
        right_eye = detection.get_keypoint_by_label('right_eye')
        
        # MUST have ALL THREE points: nose + left_eye + right_eye
        if not (nose and left_eye and right_eye):
            return None
        
        # Check confidence scores for all three points
        if not (nose.score > 0.3 and left_eye.score > 0.3 and right_eye.score > 0.3):
            return None
        
        # Calculate the centroid of the triangle formed by nose + left_eye + right_eye
        triangle_vertices = [
            [nose.x, nose.y],
            [left_eye.x, left_eye.y], 
            [right_eye.x, right_eye.y]
        ]
        
        # Triangle centroid = (P1 + P2 + P3) / 3
        center_x = int((nose.x + left_eye.x + right_eye.x) / 3)
        center_y = int((nose.y + left_eye.y + right_eye.y) / 3)
        
        return [center_x, center_y]
    
    @staticmethod
    def _get_clothing_center(detection: DetectionResult) -> Optional[List[int]]:
        """
        Get center point of clothing area (chest/torso).
        
        Args:
            detection: Detection result containing keypoints
            
        Returns:
            [x, y] coordinates of clothing center or None
        """
        x1, y1, x2, y2 = detection.bbox
        
        # Get key body keypoints
        left_shoulder = detection.get_keypoint_by_label('left_shoulder')
        right_shoulder = detection.get_keypoint_by_label('right_shoulder')
        left_hip = detection.get_keypoint_by_label('left_hip')
        right_hip = detection.get_keypoint_by_label('right_hip')
        nose = detection.get_keypoint_by_label('nose')
        
        # Strategy 1: Use shoulders and hips if available
        if (left_shoulder and right_shoulder and 
            left_shoulder.score > 0.3 and right_shoulder.score > 0.3):
            
            chest_x = int((left_shoulder.x + right_shoulder.x) / 2)
            
            # If we have hips, position between shoulders and hips
            if (left_hip and right_hip and 
                left_hip.score > 0.3 and right_hip.score > 0.3):
                shoulder_y = int((left_shoulder.y + right_shoulder.y) / 2)
                hip_y = int((left_hip.y + right_hip.y) / 2)
                chest_y = int((shoulder_y + hip_y) / 2)
            else:
                # Position below shoulders
                shoulder_y = int((left_shoulder.y + right_shoulder.y) / 2)
                chest_y = shoulder_y + int((y2 - shoulder_y) * 0.4)
            
            return [chest_x, chest_y]
        
        # Strategy 2: Use nose as reference if available
        elif nose and nose.score > 0.3:
            chest_x = int((x1 + x2) / 2)  # Center of person
            chest_y = nose.y + int((y2 - nose.y) * 0.5)  # Below face
            return [chest_x, chest_y]
        
        # Strategy 3: Fallback to geometric center of person
        else:
            chest_x = int((x1 + x2) / 2)
            chest_y = int(y1 + (y2 - y1) * 0.6)  # Lower center of bbox
            return [chest_x, chest_y]
    
    @staticmethod
    def _get_strategic_negative_points(
        all_detections: List[DetectionResult], 
        image_shape: Tuple[int, int], 
        max_points: int = 3
    ) -> List[List[int]]:
        """
        Get strategic negative points for intelligent prompt generation.
        
        Args:
            all_detections: List of all detection results
            image_shape: Image dimensions (height, width)
            max_points: Maximum number of negative points to return
            
        Returns:
            List of strategic negative points (clothing + background)
        """
        negative_points = []
        
        # 1. Get one clothing point from primary person (if available)
        if all_detections and max_points > 0:
            primary_detection = all_detections[0]
            clothing_points = PromptGenerator._get_clothing_negative_points(primary_detection)
            if clothing_points:
                negative_points.append(clothing_points[0])
        
        # 2. Add strategic background points to fill remaining slots
        remaining_slots = max_points - len(negative_points)
        if remaining_slots > 0:
            background_point = PromptGenerator._get_strategic_background_point(all_detections, image_shape)
            if background_point:
                negative_points.append(background_point)
                remaining_slots -= 1
        
        # 3. Add corner background if we still have slots
        if remaining_slots > 0:
            corner_point = PromptGenerator._get_corner_background_point(all_detections, image_shape)
            if corner_point:
                negative_points.append(corner_point)
        
        return negative_points
    
    @staticmethod
    def _get_positive_points(detection: DetectionResult) -> List[List[int]]:
        """
        Get balanced positive prompt points - nose and forehead.
        
        Args:
            detection: Detection result containing keypoints
            
        Returns:
            List of [x, y] coordinates for positive prompts (nose + forehead)
        """
        positive_points = []
        
        # 1. Nose - primary facial keypoint (most reliable skin area)
        nose = detection.get_keypoint_by_label('nose')
        if nose and nose.score > 0.3:
            positive_points.append([nose.x, nose.y])
        
        # 2. Forehead - calculated position above facial features
        forehead_point = PromptGenerator._calculate_forehead_position(detection)
        if forehead_point:
            positive_points.append(forehead_point)
        
        # Fallback: if we don't have both nose and forehead, add eye as backup
        if len(positive_points) < 2:
            for eye_label in ['left_eye', 'right_eye']:
                if len(positive_points) >= 2:
                    break
                eye = detection.get_keypoint_by_label(eye_label)
                if eye and eye.score > 0.4:
                    positive_points.append([eye.x, eye.y])
        
        return positive_points
    
    @staticmethod
    def _calculate_forehead_position(detection: DetectionResult) -> Optional[List[int]]:
        """
        Calculate forehead position based on facial keypoints.
        
        Args:
            detection: Detection result containing keypoints
            
        Returns:
            [x, y] coordinates for forehead point, or None if can't calculate
        """
        nose = detection.get_keypoint_by_label('nose')
        left_eye = detection.get_keypoint_by_label('left_eye')
        right_eye = detection.get_keypoint_by_label('right_eye')
        
        # Need at least nose and one eye
        if not nose or nose.score < 0.3:
            return None
            
        if left_eye and right_eye and left_eye.score > 0.3 and right_eye.score > 0.3:
            # Use both eyes to calculate forehead
            eyes_center_x = int((left_eye.x + right_eye.x) / 2)
            eyes_center_y = int((left_eye.y + right_eye.y) / 2)
            
            # Forehead is above the line between eyes and nose
            forehead_x = int((eyes_center_x + nose.x) / 2)
            forehead_y = eyes_center_y - int(abs(eyes_center_y - nose.y) * 0.8)  # 80% up from eyes toward forehead
            
        elif left_eye and left_eye.score > 0.3:
            # Use left eye and nose
            forehead_x = int((left_eye.x + nose.x) / 2)
            forehead_y = left_eye.y - int(abs(left_eye.y - nose.y) * 0.8)
            
        elif right_eye and right_eye.score > 0.3:
            # Use right eye and nose
            forehead_x = int((right_eye.x + nose.x) / 2)
            forehead_y = right_eye.y - int(abs(right_eye.y - nose.y) * 0.8)
            
        else:
            # Only nose available, estimate forehead
            forehead_x = nose.x
            forehead_y = nose.y - 40  # Simple offset upward
            
        return [forehead_x, forehead_y]
    
    @staticmethod
    def _calculate_cheek_positions(detection: DetectionResult) -> List[List[int]]:
        """
        Calculate cheek positions for additional facial skin points.
        
        Args:
            detection: Detection result containing keypoints
            
        Returns:
            List of [x, y] coordinates for cheek points
        """
        cheek_points = []
        
        nose = detection.get_keypoint_by_label('nose')
        left_eye = detection.get_keypoint_by_label('left_eye')
        right_eye = detection.get_keypoint_by_label('right_eye')
        left_ear = detection.get_keypoint_by_label('left_ear')
        right_ear = detection.get_keypoint_by_label('right_ear')
        
        # Left cheek (between nose and left ear/eye)
        if nose and left_eye and nose.score > 0.3 and left_eye.score > 0.3:
            if left_ear and left_ear.score > 0.3:
                # Use ear for more accurate cheek position
                left_cheek_x = int((nose.x + left_ear.x) / 2)
                left_cheek_y = int((nose.y + left_eye.y) / 2)
            else:
                # Use eye position
                left_cheek_x = int((nose.x + left_eye.x) / 2 + 20)  # Slightly toward ear
                left_cheek_y = int((nose.y + left_eye.y) / 2)
            cheek_points.append([left_cheek_x, left_cheek_y])
        
        # Right cheek (between nose and right ear/eye)
        if nose and right_eye and nose.score > 0.3 and right_eye.score > 0.3:
            if right_ear and right_ear.score > 0.3:
                # Use ear for more accurate cheek position
                right_cheek_x = int((nose.x + right_ear.x) / 2)
                right_cheek_y = int((nose.y + right_eye.y) / 2)
            else:
                # Use eye position
                right_cheek_x = int((nose.x + right_eye.x) / 2 - 20)  # Slightly toward ear
                right_cheek_y = int((nose.y + right_eye.y) / 2)
            cheek_points.append([right_cheek_x, right_cheek_y])
        
        return cheek_points
    
    @staticmethod
    def _get_negative_points(all_detections: List[DetectionResult], image_shape: Tuple[int, int]) -> List[List[int]]:
        """
        Get balanced negative prompt points - clothing areas + strategic background.
        Total points should not exceed 3 negative points.
        
        Args:
            all_detections: List of all detection results in the image
            image_shape: Shape of the image (height, width)
            
        Returns:
            List of [x, y] coordinates for negative prompts (max 3 points)
        """
        negative_points = []
        height, width = image_shape
        
        # 1. Get one clothing point from the primary detection (usually first person)
        if all_detections:
            primary_detection = all_detections[0]  # Focus on primary person
            clothing_points = PromptGenerator._get_clothing_negative_points(primary_detection)
            if clothing_points:
                # Take only the first (most reliable) clothing point
                negative_points.append(clothing_points[0])
        
        # 2. Add one strategic background point
        background_point = PromptGenerator._get_strategic_background_point(all_detections, image_shape)
        if background_point:
            negative_points.append(background_point)
        
        # 3. If we have space for a third negative point, add another clothing area or background
        if len(negative_points) < 3:
            if len(all_detections) > 1:
                # If multiple people, get clothing from second person
                secondary_detection = all_detections[1]
                secondary_clothing = PromptGenerator._get_clothing_negative_points(secondary_detection)
                if secondary_clothing:
                    negative_points.append(secondary_clothing[0])
            else:
                # Add another strategic background point
                corner_point = PromptGenerator._get_corner_background_point(all_detections, image_shape)
                if corner_point:
                    negative_points.append(corner_point)
        
        # Ensure we have at least 1 negative point
        if not negative_points:
            # Fallback to safe background position
            fallback_point = [width // 2, int(0.9 * height)]
            negative_points.append(fallback_point)
        
        return negative_points
    
    @staticmethod
    def _get_clothing_negative_points(detection: DetectionResult) -> List[List[int]]:
        """
        Get negative points from clothing areas (chest and lower body).
        
        Args:
            detection: Detection result containing keypoints
            
        Returns:
            List of [x, y] coordinates for clothing area negative prompts
        """
        clothing_points = []
        x1, y1, x2, y2 = detection.bbox
        
        # Get key body keypoints for clothing area calculation
        left_shoulder = detection.get_keypoint_by_label('left_shoulder')
        right_shoulder = detection.get_keypoint_by_label('right_shoulder')
        left_hip = detection.get_keypoint_by_label('left_hip')
        right_hip = detection.get_keypoint_by_label('right_hip')
        nose = detection.get_keypoint_by_label('nose')
        
        # 1. Chest/torso clothing area (between shoulders and hips)
        if left_shoulder and right_shoulder and left_shoulder.score > 0.3 and right_shoulder.score > 0.3:
            chest_center_x = int((left_shoulder.x + right_shoulder.x) / 2)
            
            # If we have hips, position between shoulders and hips
            if left_hip and right_hip and left_hip.score > 0.3 and right_hip.score > 0.3:
                hip_center_y = int((left_hip.y + right_hip.y) / 2)
                shoulder_center_y = int((left_shoulder.y + right_shoulder.y) / 2)
                chest_y = int((shoulder_center_y + hip_center_y) / 2)
            else:
                # Fall back to area below shoulders
                shoulder_center_y = int((left_shoulder.y + right_shoulder.y) / 2)
                chest_y = shoulder_center_y + int((y2 - shoulder_center_y) * 0.3)
            
            clothing_points.append([chest_center_x, chest_y])
        
        # 2. Lower body clothing area (if hips are available)
        if left_hip and right_hip and left_hip.score > 0.3 and right_hip.score > 0.3:
            hip_center_x = int((left_hip.x + right_hip.x) / 2)
            hip_center_y = int((left_hip.y + right_hip.y) / 2)
            
            # Position below hips (pants/skirt area)
            lower_clothing_y = hip_center_y + int((y2 - hip_center_y) * 0.4)
            clothing_points.append([hip_center_x, lower_clothing_y])
        
        # 3. If we don't have enough keypoints, use geometric estimation
        if not clothing_points:
            # Use center of person's bounding box as clothing area
            person_center_x = int((x1 + x2) / 2)
            
            # If we have nose, position clothing area below face
            if nose and nose.score > 0.3:
                clothing_y = nose.y + int((y2 - nose.y) * 0.4)
            else:
                # Use middle-lower area of bounding box
                clothing_y = int(y1 + (y2 - y1) * 0.6)
            
            clothing_points.append([person_center_x, clothing_y])
        
        # Ensure points are within person's bounding box
        valid_clothing_points = []
        for point in clothing_points:
            x, y = point
            if x1 <= x <= x2 and y1 <= y <= y2:
                valid_clothing_points.append(point)
        
        return valid_clothing_points
    
    @staticmethod
    def _get_strategic_background_point(all_detections: List[DetectionResult], image_shape: Tuple[int, int]) -> Optional[List[int]]:
        """
        Get one strategic background point away from all people.
        
        Args:
            all_detections: List of all detection results
            image_shape: Image dimensions (height, width)
            
        Returns:
            Single background point coordinates or None
        """
        height, width = image_shape
        
        # Create exclusion zones around all people
        exclusion_zones = []
        for detection in all_detections:
            x1, y1, x2, y2 = detection.bbox
            margin = 100
            exclusion_zones.append([
                max(0, x1 - margin), max(0, y1 - margin),
                min(width, x2 + margin), min(height, y2 + margin)
            ])
        
        # Try strategic positions (top-center, bottom-center, sides)
        candidate_positions = [
            [width // 2, height // 8],      # Top center
            [width // 2, 7 * height // 8],  # Bottom center
            [width // 8, height // 2],      # Left center
            [7 * width // 8, height // 2],  # Right center
        ]
        
        for position in candidate_positions:
            x, y = position
            # Check if position is outside all exclusion zones
            is_safe = True
            for ex_x1, ex_y1, ex_x2, ex_y2 in exclusion_zones:
                if ex_x1 <= x <= ex_x2 and ex_y1 <= y <= ex_y2:
                    is_safe = False
                    break
            
            if is_safe:
                return position
        
        return None
    
    @staticmethod
    def _get_corner_background_point(all_detections: List[DetectionResult], image_shape: Tuple[int, int]) -> Optional[List[int]]:
        """
        Get a corner background point away from all people.
        
        Args:
            all_detections: List of all detection results
            image_shape: Image dimensions (height, width)
            
        Returns:
            Corner background point coordinates or None
        """
        height, width = image_shape
        
        # Try corner positions
        corner_positions = [
            [width // 6, height // 6],        # Top-left
            [5 * width // 6, height // 6],    # Top-right
            [width // 6, 5 * height // 6],    # Bottom-left
            [5 * width // 6, 5 * height // 6], # Bottom-right
        ]
        
        for position in corner_positions:
            if PromptGenerator._is_safe_background_point(position, all_detections, min_distance=120):
                return position
        
        return None
    
    @staticmethod
    def _generate_safe_background_points(
        width: int, 
        height: int, 
        exclusion_zones: List[List[int]]
    ) -> List[List[int]]:
        """
        Generate candidate points in safe background areas.
        
        Args:
            width: Image width
            height: Image height
            exclusion_zones: List of [x1, y1, x2, y2] zones to avoid
            
        Returns:
            List of candidate background points
        """
        candidate_points = []
        grid_size = 100  # Grid spacing for candidate points
        
        # Generate grid points across the image
        for y in range(grid_size, height - grid_size, grid_size):
            for x in range(grid_size, width - grid_size, grid_size):
                point = [x, y]
                
                # Check if point is outside all exclusion zones
                is_safe = True
                for ex_x1, ex_y1, ex_x2, ex_y2 in exclusion_zones:
                    if ex_x1 <= x <= ex_x2 and ex_y1 <= y <= ex_y2:
                        is_safe = False
                        break
                
                if is_safe:
                    candidate_points.append(point)
        
        # Add image border points (safe background areas)
        border_margin = 50
        border_points = [
            # Top border
            [width // 4, border_margin],
            [width // 2, border_margin],
            [3 * width // 4, border_margin],
            # Bottom border
            [width // 4, height - border_margin],
            [width // 2, height - border_margin],
            [3 * width // 4, height - border_margin],
            # Left border
            [border_margin, height // 4],
            [border_margin, height // 2],
            [border_margin, 3 * height // 4],
            # Right border
            [width - border_margin, height // 4],
            [width - border_margin, height // 2],
            [width - border_margin, 3 * height // 4],
        ]
        
        # Filter border points to ensure they're outside exclusion zones
        for point in border_points:
            is_safe = True
            for ex_x1, ex_y1, ex_x2, ex_y2 in exclusion_zones:
                if ex_x1 <= point[0] <= ex_x2 and ex_y1 <= point[1] <= ex_y2:
                    is_safe = False
                    break
            if is_safe:
                candidate_points.append(point)
        
        return candidate_points
    
    @staticmethod
    def _is_safe_background_point(
        point: List[int], 
        all_detections: List[DetectionResult], 
        min_distance: int = 100
    ) -> bool:
        """
        Check if a point is safely away from all detected persons.
        
        Args:
            point: [x, y] coordinate to check
            all_detections: List of all detection results
            min_distance: Minimum distance from any keypoint
            
        Returns:
            True if point is safe background area
        """
        px, py = point
        
        for detection in all_detections:
            # Check distance from bounding box center
            x1, y1, x2, y2 = detection.bbox
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            bbox_distance = np.sqrt((px - bbox_center_x)**2 + (py - bbox_center_y)**2)
            
            if bbox_distance < min_distance:
                return False
            
            # Check distance from all keypoints
            for kp in detection.keypoints:
                if kp.score > 0.3:  # Only consider confident keypoints
                    distance = np.sqrt((px - kp.x)**2 + (py - kp.y)**2)
                    if distance < min_distance:
                        return False
        
        return True
    
    @staticmethod
    def _is_far_from_keypoints(
        point: List[int], 
        detection: DetectionResult, 
        min_distance: int = 30
    ) -> bool:
        """
        Check if a point is far enough from all keypoints.
        
        Args:
            point: [x, y] coordinate to check
            detection: Detection result containing keypoints
            min_distance: Minimum distance in pixels
            
        Returns:
            True if point is far from all keypoints
        """
        px, py = point
        
        for kp in detection.keypoints:
            if kp.score > 0.3:  # Only consider confident keypoints
                distance = np.sqrt((px - kp.x)**2 + (py - kp.y)**2)
                if distance < min_distance:
                    return False
        
        return True
    
    @staticmethod
    def generate_chest_triangle_mask(
        detection: DetectionResult, 
        image_shape: Tuple[int, int], 
        expansion_factor: float = 1.2
    ) -> np.ndarray:
        """
        Generate a triangular mask for the chest area based on keypoints.
        
        Args:
            detection: Detection result containing keypoints
            image_shape: Shape of the image (height, width)
            expansion_factor: Factor to expand the triangle
            
        Returns:
            Binary mask for chest triangle area
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get key points for chest triangle
        nose = detection.get_keypoint_by_label('nose')
        left_shoulder = detection.get_keypoint_by_label('left_shoulder')
        right_shoulder = detection.get_keypoint_by_label('right_shoulder')
        
        if not all([nose, left_shoulder, right_shoulder]):
            return mask
            
        if not all([kp.score > 0.3 for kp in [nose, left_shoulder, right_shoulder]]):
            return mask
        
        # Define triangle vertices
        triangle_points = np.array([
            [nose.x, nose.y],
            [left_shoulder.x, left_shoulder.y],
            [right_shoulder.x, right_shoulder.y]
        ], dtype=np.int32)
        
        # Expand triangle if needed
        if expansion_factor != 1.0:
            center = np.mean(triangle_points, axis=0)
            triangle_points = center + (triangle_points - center) * expansion_factor
            triangle_points = triangle_points.astype(np.int32)
        
        # Fill triangle
        cv2.fillPoly(mask, [triangle_points], 255)
        
        return mask