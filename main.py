import cv2
import numpy as np
import os

class BodySegmentAnalyzer:
    def __init__(self, image_path, actual_height):
        self.image_path = image_path
        self.actual_height = actual_height  # in meters
        self.keypoints = []
        self.segments = {
            'thorax': ['CV7', 'L3/L4'],
            'upper_arm_left': ['left_shoulder', 'left_elbow'],
            'upper_arm_right': ['right_shoulder', 'right_elbow'],
            'forearm_left': ['left_elbow', 'left_wrist'],
            'forearm_right': ['right_elbow', 'right_wrist'],
            'thigh_left': ['left_hip', 'left_knee'],
            'thigh_right': ['right_hip', 'right_knee'],
            'shank_left': ['left_knee', 'left_ankle'],
            'shank_right': ['right_knee', 'right_ankle']
        }
        self.required_points = [
            'CV7', 'L3/L4',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'head_top', 'heel'
        ]
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display (possibly scaled) coordinates back to original image coordinates
            if hasattr(self, 'display_scale') and self.display_scale > 0:
                x_orig = int(x / self.display_scale)
                y_orig = int(y / self.display_scale)
            else:
                x_orig, y_orig = x, y

            if len(self.keypoints) < len(self.required_points):
                point_name = self.required_points[len(self.keypoints)]
                self.keypoints.append((x_orig, y_orig, point_name))
                print(f"Point {point_name} selected at ({x_orig}, {y_orig})")
            else:
                print("All required points have already been selected.")
                
    def select_keypoints(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print("Error: Could not load image")
            return False
        window_name = "Body Segment Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("Select the following points in order:")
        for i, point in enumerate(self.required_points):
            print(f"{i+1}. {point}")
        print("\nClick on the image to select each point. Close window or press ESC when done.")

        # compute a display size that preserves aspect ratio and fits the screen
        # try to get actual screen size on Windows, otherwise use a reasonable max
        max_w, max_h = 1280, 800
        try:
            import ctypes
            user32 = ctypes.windll.user32
            max_w = user32.GetSystemMetrics(0) - 100
            max_h = user32.GetSystemMetrics(1) - 150
        except Exception:
            pass

        # Keep display at a mobile-friendly resolution to avoid fullscreen distortion
        max_w = min(max_w, 600)
        max_h = min(max_h, 1000)

        img_h, img_w = self.image.shape[:2]
        scale = min(max_w / img_w, max_h / img_h, 1.0)
        self.display_scale = scale

        while True:
            display = self.image.copy()

            # Draw already-selected points
            # Draw already-selected points (scale to display coordinates)
            for (px, py, pname) in self.keypoints:
                dpx = int(px * self.display_scale)
                dpy = int(py * self.display_scale)
                cv2.circle(display, (px, py), 5, (0, 255, 0), -1)
                # Put labels on the resized display; compute position in display coords
                cv2.putText(display, pname, (px+10, py-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # If there are still points to select, show hint for the next one
            if len(self.keypoints) < len(self.required_points):
                next_idx = len(self.keypoints)
                next_name = self.required_points[next_idx]
                hint_text = f"Next: {next_name} ({next_idx+1}/{len(self.required_points)})"
            else:
                hint_text = "All points selected. Close window or press ESC."

            # resize display for showing (preserve aspect ratio)
            disp_w = max(1, int(img_w * self.display_scale))
            disp_h = max(1, int(img_h * self.display_scale))
            display_resized = cv2.resize(display, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

            # Draw hint background for readability on the resized image
            cv2.rectangle(display_resized, (5, 5), (min(400, disp_w-10), 35), (0, 0, 0), -1)
            cv2.putText(display_resized, hint_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Ensure window is not fullscreen and set it to the resized dimensions
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 0)
            except Exception:
                pass
            try:
                cv2.resizeWindow(window_name, disp_w, disp_h)
            except Exception:
                pass
            cv2.imshow(window_name, display_resized)

            key = cv2.waitKey(20) & 0xFF
            # ESC key to exit early
            if key == 27:
                break

            # If window was closed by user, stop
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

            # If finished selecting all points, allow user to close or press ESC
            if len(self.keypoints) >= len(self.required_points):
                # small sleep handled by waitKey; continue to show final message until user closes
                pass

        cv2.destroyAllWindows()
        return True

    def create_annotated_image(self, out_path="annotated_output.jpg"):
        """Draw segments, points and labels on a copy of the original image and save it."""
        if len(self.keypoints) == 0:
            print("No keypoints to annotate.")
            return None

        # Map keypoints by name
        kp_dict = {name: (x, y) for (x, y, name) in self.keypoints}

        annotated = self.image.copy()

        # Color palette for segments
        colors = [
            (0, 0, 255),    # red
            (0, 165, 255),  # orange
            (0, 255, 255),  # yellow
            (0, 255, 0),    # green
            (255, 0, 0),    # blue
            (255, 0, 255),  # magenta
            (255, 255, 0),  # cyan (BGR order)
            (128, 0, 128),  # purple-ish
            (0, 128, 255),
        ]

        # Draw segments lines
        i = 0
        for segment, points in self.segments.items():
            p1_name, p2_name = points[0], points[1]
            p1 = kp_dict.get(p1_name)
            p2 = kp_dict.get(p2_name)
            if p1 and p2:
                color = colors[i % len(colors)]
                cv2.line(annotated, p1, p2, color, thickness=max(2, int(min(self.image.shape[:2]) / 200)))
                # draw endpoints
                cv2.circle(annotated, p1, 6, color, -1)
                cv2.circle(annotated, p2, 6, color, -1)
                # put segment name at midpoint
                mx, my = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
                cv2.putText(annotated, segment.replace('_', ' ').title(), (mx+5, my+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            i += 1

        # Also annotate any remaining standalone keypoints (like head_top, heel)
        for (x, y, name) in self.keypoints:
            cv2.putText(annotated, name, (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save annotated image
        cv2.imwrite(out_path, annotated)
        print(f"Annotated image saved to '{out_path}'")

        # Show annotated image resized to fit screen
        try:
            import ctypes
            user32 = ctypes.windll.user32
            max_w = user32.GetSystemMetrics(0) - 100
            max_h = user32.GetSystemMetrics(1) - 150
        except Exception:
            max_w, max_h = 1280, 800

        h, w = annotated.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        disp = cv2.resize(annotated, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow("Annotated Output", disp)
        cv2.waitKey(0)
        cv2.destroyWindow("Annotated Output")
        return out_path
        
    def calculate_pixel_height(self):
        """Calculate pixel height from head top to heel"""
        head_top = next((p for p in self.keypoints if p[2] == 'head_top'), None)
        heel = next((p for p in self.keypoints if p[2] == 'heel'), None)
        
        if head_top and heel:
            return np.sqrt((head_top[0]-heel[0])**2 + (head_top[1]-heel[1])**2)
        return None
        
    def calculate_segment_lengths(self):
        if len(self.keypoints) != len(self.required_points):
            print(f"Error: Selected {len(self.keypoints)} points, but {len(self.required_points)} are required")
            return None
            
        pixel_height = self.calculate_pixel_height()
        if pixel_height is None:
            print("Error: Could not calculate pixel height")
            return None
            
        scale_factor = self.actual_height / pixel_height
        print(f"Scale factor: {scale_factor:.6f} m/px")
        
        segment_lengths = {}
        
        for segment, points in self.segments.items():
            point1 = next((p for p in self.keypoints if p[2] == points[0]), None)
            point2 = next((p for p in self.keypoints if p[2] == points[1]), None)
            
            if point1 and point2:
                pixel_distance = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
                real_length = pixel_distance * scale_factor
                segment_lengths[segment] = {
                    'pixel_length': pixel_distance,
                    'real_length_m': real_length
                }
                
        return segment_lengths, scale_factor
        
    def save_results(self, segment_lengths, scale_factor):
        # Save keypoints
        with open("keypoints_coordinates.txt", "w") as f:
            f.write("Keypoint Coordinates (x, y, name):\n")
            for x, y, name in self.keypoints:
                f.write(f"{name}: ({x}, {y})\n")
                
        # Save segment lengths
        with open("segment_lengths.txt", "w") as f:
            f.write("Body Segment Lengths Analysis\n")
            f.write("=" * 40 + "\n")
            f.write(f"Student Name: Sameer Wanjari\n")
            f.write(f"Roll No: 21174028\n")
            f.write(f"Branch: Engineering Physics\n")
            f.write(f"Actual Height: {self.actual_height} m\n")
            f.write(f"Pixel Height: {self.calculate_pixel_height()} px\n")
            f.write(f"Scale Factor: {scale_factor:.6f} m/px\n\n")
            
            f.write("Segment Lengths:\n")
            f.write("-" * 40 + "\n")
            for segment, data in segment_lengths.items():
                f.write(f"{segment.replace('_', ' ').title()}: ")
                f.write(f"{data['real_length_m']:.3f} m ")
                f.write(f"({data['pixel_length']:.1f} px)\n")
                
        print("Results saved to 'keypoints_coordinates.txt' and 'segment_lengths.txt'")
        
    def run_analysis(self):
        print("Starting Body Segment Analysis...")
        print(f"Student: Sameer Wanjari (21174028)")
        print(f"Image: {self.image_path}")
        print(f"Actual Height: {self.actual_height} m")
        
        if not self.select_keypoints():
            return
            
        results = self.calculate_segment_lengths()
        if results:
            segment_lengths, scale_factor = results
            self.save_results(segment_lengths, scale_factor)
            # Create and show annotated image with segments and labels
            try:
                self.create_annotated_image()
            except Exception as e:
                print(f"Could not create annotated image: {e}")
            
            # Print summary
            print("\n" + "="*50)
            print("ANALYSIS SUMMARY")
            print("="*50)
            print("Student: Sameer Wanjari")
            print("Roll No: 21174028")
            print("Branch: Engineering Physics")
            print("-" * 50)
            for segment, data in segment_lengths.items():
                print(f"{segment.replace('_', ' ').title():<20}: {data['real_length_m']:.3f} m")
                
        else:
            print("Analysis failed. Please check your point selections.")

# Main execution
if __name__ == "__main__":
    # STUDENT: Replace with your actual height in meters
    ACTUAL_HEIGHT = 1.78  # meters
    
    # STUDENT: Replace with your image filename
    IMAGE_FILE = "body_photo.jpg"
    
    # Check if image exists
    if not os.path.exists(IMAGE_FILE):
        print(f"Error: Image file '{IMAGE_FILE}' not found.")
        print("Please ensure the image is in the same directory as this script.")
    else:
        analyzer = BodySegmentAnalyzer(IMAGE_FILE, ACTUAL_HEIGHT)
        analyzer.run_analysis()