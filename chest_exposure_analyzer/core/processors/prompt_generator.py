"""
Generate SAM2 prompts from detected keypoints.
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
        image_shape: Tuple[int, int],
        all_detections: List[DetectionResult] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate positive and negative prompts for SAM2 based on keypoints.
        
        Args:
            detection: Current detection result containing keypoints
            image_shape: Shape of the image (height, width)
            all_detections: List of all detections in the image (for negative point placement)
            
        Returns:
            Tuple of (points, labels) where:
            - points: np.ndarray of shape (N, 2) with (x, y) coordinates
            - labels: np.ndarray of shape (N,) with 1 for positive, 0 for negative
        """
        points = []
        labels = []
        
        # Generate positive points (only facial regions)
        positive_points = PromptGenerator._get_positive_points(detection)
        points.extend(positive_points)
        labels.extend([1] * len(positive_points))
        
        # Generate negative points (outside all person bounding boxes)
        if all_detections is None:
            all_detections = [detection]
        negative_points = PromptGenerator._get_negative_points(all_detections, image_shape)
        points.extend(negative_points)
        labels.extend([0] * len(negative_points))
        
        # Ensure we have valid points
        if not points:
            # Fallback: use nose as positive if available
            nose = detection.get_keypoint_by_label('nose')
            if nose and nose.score > 0.3:
                points.append([nose.x, nose.y])
                labels.append(1)
        
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)
    
    @staticmethod
    def _get_positive_points(detection: DetectionResult) -> List[List[int]]:
        """
        Get positive prompt points from facial region only (conservative approach).
        
        Args:
            detection: Detection result containing keypoints
            
        Returns:
            List of [x, y] coordinates for positive prompts (facial area only)
        """
        positive_points = []
        
        # Primary facial keypoints (high confidence skin areas)
        nose = detection.get_keypoint_by_label('nose')
        if nose and nose.score > 0.5:
            positive_points.append([nose.x, nose.y])
        
        # Eyes (additional facial confirmation points)
        for eye_label in ['left_eye', 'right_eye']:
            eye = detection.get_keypoint_by_label(eye_label)
            if eye and eye.score > 0.5:
                positive_points.append([eye.x, eye.y])
        
        # Calculate and add forehead position
        forehead_point = PromptGenerator._calculate_forehead_position(detection)
        if forehead_point:
            positive_points.append(forehead_point)
        
        # Add cheek area points if we have enough facial landmarks
        cheek_points = PromptGenerator._calculate_cheek_positions(detection)
        positive_points.extend(cheek_points)
        
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
        Get negative prompt points outside all person bounding boxes (safe background areas).
        
        Args:
            all_detections: List of all detection results in the image
            image_shape: Shape of the image (height, width)
            
        Returns:
            List of [x, y] coordinates for negative prompts (background only)
        """
        height, width = image_shape
        negative_points = []
        
        # Create a combined exclusion zone from all person bounding boxes
        exclusion_zones = []
        for detection in all_detections:
            x1, y1, x2, y2 = detection.bbox
            # Add margin around each person to avoid clothing/nearby objects
            margin = 80
            exclusion_zones.append([
                max(0, x1 - margin), 
                max(0, y1 - margin), 
                min(width, x2 + margin), 
                min(height, y2 + margin)
            ])
        
        # Generate candidate points in safe background areas
        candidate_points = PromptGenerator._generate_safe_background_points(
            width, height, exclusion_zones
        )
        
        # Filter points to ensure they don't conflict with any detection
        for point in candidate_points:
            if PromptGenerator._is_safe_background_point(point, all_detections, min_distance=100):
                negative_points.append(point)
        
        return negative_points
    
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