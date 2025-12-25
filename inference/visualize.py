import cv2
import numpy as np

class AdvancedVisualizer:
    def __init__(self, class_names):
        self.class_names = class_names
        # Professional Color Palette (BGR)
        self.colors = {
            'with_mask': (0, 255, 127),      # Soft Green
            'without_mask': (0, 0, 255),     # Bright Red
            'mask_weared_incorrect': (0, 165, 255) # Orange
        }

    def _get_color(self, label):
        return self.colors.get(label, (255, 255, 255))

    def draw_styled_boxes(self, img, results):
        """
        Advanced Drawing with Semi-Transparent Overlays
        """
        overlay = img.copy()
        
        # Dynamic scaling based on image size
        thickness = max(2, int(img.shape[1] / 500))
        font_scale = img.shape[1] / 1000

        for box in results.boxes:
            # Extract coordinates and info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label_name = self.class_names[cls_id]
            color = self._get_color(label_name)

            # 1. Draw Glass-morphism Effect (Semi-transparent Fill)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # 2. Draw Main Bounding Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            # 3. Design Label Tag
            label_text = f"{label_name} {conf:.2%}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Label Background
            cv2.rectangle(img, (x1, y1 - int(h*1.5)), (x1 + w, y1), color, -1)
            
            # Label Text
            cv2.putText(img, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 
                        max(1, int(thickness/2)), cv2.LINE_AA)

        # Apply transparency to the fill (Alpha blending)
        alpha = 0.15
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        return img

    def add_info_panel(self, img, fps, counts):
        """Adds a top-bar summary panel"""
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 50), (20, 20, 20), -1)
        
        info_str = f"FPS: {fps} | Safe: {counts.get('with_mask', 0)} | Risk: {counts.get('without_mask', 0)}"
        cv2.putText(img, info_str, (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

