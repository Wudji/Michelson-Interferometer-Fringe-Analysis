import cv2
import numpy as np
from collections import deque
from scipy.signal import find_peaks
from pathlib import Path

class MichelsonFringeCounter:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Read video metadata and first frame.
        ret, first_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read the first frame from the video.")
        
        self.frame_height, self.frame_width = first_frame.shape[:2]
        self.first_frame = first_frame
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps is None or self.fps <= 1:
            self.fps = 30.0
        # UI display scale factor.
        self.display_scale = self.calculate_display_scale()
        # Region of interest (ROI) selection.
        self.roi = None
        self.select_roi()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Counting state.
        self.fringe_count = 0
        self.intensity_history = []
        self.frame_num = 0
        
        self.display_history = deque(maxlen=100)
        # Auto-tune detection using video FPS.
        self.max_fringe_rate = 12.0  # fringes per second
        self.min_cycle_frames = max(1, int(round(self.fps / self.max_fringe_rate)))
        self.peak_distance = max(1, self.min_cycle_frames)
        self.min_prominence = 1.0
        self.prominence_factor = 3.0
        self.noise_window = max(20, int(self.fps))
        self.flat_window = max(10, int(self.fps * 0.5))
        self.flat_std_threshold = 0.8

    def calculate_display_scale(self):
        if self.frame_width > 2000:
            return 0.25
        elif self.frame_width > 1500:
            return 0.5
        else:
            return 1.0
    
    def select_roi(self):
        temp_show = cv2.resize(self.first_frame, None, 
                              fx=self.display_scale, 
                              fy=self.display_scale)
        
        print("\nSelect the ROI at the interference center")
        r = cv2.selectROI("Center Choose Window", temp_show, fromCenter=False)
        cv2.destroyWindow("Center Choose Window")
        
        if r[2] == 0 or r[3] == 0:
            raise ValueError("Invalid ROI.")
        
        self.roi = [int(x / self.display_scale) for x in r]
        self.roi_display = r
        
        print(f"Selected ROI: x={self.roi[0]}, y={self.roi[1]}, w={self.roi[2]}, h={self.roi[3]}")
    
    def get_center_intensity(self, frame):
        red_channel = frame[:, :, 2]
        
        roi = red_channel[self.roi[1]:self.roi[1]+self.roi[3], 
                         self.roi[0]:self.roi[0]+self.roi[2]]
        avg_intensity = np.mean(roi)
        
        return avg_intensity, red_channel
    
    def analyze_fringes(self):
        if len(self.intensity_history) < 20:
            return 0, np.array([]), np.array([])
        
        data = np.array(self.intensity_history)
        
        # Reduce smoothing when fringe change is fast.
        window_size = max(1, min(5, self.min_cycle_frames))
        if window_size % 2 == 0:
            window_size -= 1
        if len(data) > window_size:
            smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed_data = data

        tail = smoothed_data[-self.noise_window:]
        if len(tail) > 2:
            noise_level = np.median(np.abs(np.diff(tail)))
        else:
            noise_level = 0.0
        adaptive_prominence = max(self.min_prominence, self.prominence_factor * noise_level)

        # Flat-segment gate to suppress false counts when OPD is nearly unchanged.
        kernel = np.ones(self.flat_window) / self.flat_window
        mean = np.convolve(smoothed_data, kernel, mode='same')
        sq_mean = np.convolve(smoothed_data**2, kernel, mode='same')
        local_std = np.sqrt(np.maximum(sq_mean - mean**2, 0))
        
        # Peak detection.
        peaks, _ = find_peaks(smoothed_data, 
                              distance=self.peak_distance, 
                              prominence=adaptive_prominence)
        valid_peaks = peaks[local_std[peaks] >= self.flat_std_threshold]
        
        return len(valid_peaks), smoothed_data, valid_peaks
    
    def draw_detection_info(self, frame, red_channel, current_intensity, current_count):
        """Draw detection overlays and intensity history."""
        display_frame = cv2.resize(frame, None, 
                                  fx=self.display_scale, 
                                  fy=self.display_scale)
        
        cv2.rectangle(display_frame, 
                     (self.roi_display[0], self.roi_display[1]), 
                     (self.roi_display[0]+self.roi_display[2], 
                      self.roi_display[1]+self.roi_display[3]), 
                     (0, 255, 0), 3)
        
        info_height = 170
        cv2.rectangle(display_frame, (5, 5), (480, info_height), (0, 0, 0), -1)
        
        y_offset = 30
        cv2.putText(display_frame, f"Frame: {self.frame_num}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(display_frame, f"Intensity: {current_intensity:.2f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        if len(self.intensity_history) > 10:
            intensity_range = np.max(self.intensity_history[-50:]) - np.min(self.intensity_history[-50:])
            cv2.putText(display_frame, f"Range (50fr): {intensity_range:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        y_offset += 30
        cv2.putText(display_frame, f"Current Count: {current_count}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(display_frame, f"Params: dist={self.peak_distance}, prom>= {self.min_prominence:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Update rolling intensity history for display.
        self.display_history.append(current_intensity)
        
        if len(self.display_history) > 1:
            graph_height = 150
            graph_width = 450
            graph_x = display_frame.shape[1] - graph_width - 10
            graph_y = 10
            # Draw graph background.
            cv2.rectangle(display_frame, 
                         (graph_x, graph_y), 
                         (graph_x + graph_width, graph_y + graph_height),
                         (0, 0, 0), -1)
            cv2.rectangle(display_frame, 
                         (graph_x, graph_y), 
                         (graph_x + graph_width, graph_y + graph_height),
                         (255, 255, 255), 2)
            
            # Draw intensity curve.
            history_array = np.array(self.display_history)
            min_val = np.min(history_array)
            max_val = np.max(history_array)
            
            # Draw min/max labels.
            cv2.putText(display_frame, f"Max: {max_val:.1f}", 
                       (graph_x + 5, graph_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Min: {min_val:.1f}", 
                       (graph_x + 5, graph_y + graph_height - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if max_val > min_val:
                normalized = (history_array - min_val) / (max_val - min_val)
                points = []
                for i, val in enumerate(normalized):
                    x = graph_x + 5 + int((i / len(normalized)) * (graph_width - 10))
                    y = graph_y + graph_height - 10 - int(val * (graph_height - 20))
                    points.append((x, y))
                
                for i in range(len(points) - 1):
                    cv2.line(display_frame, points[i], points[i+1], (0, 255, 255), 2)
                
                # Draw mean line.
                mean_val = np.mean(history_array)
                mean_normalized = (mean_val - min_val) / (max_val - min_val)
                mean_y = graph_y + graph_height - 10 - int(mean_normalized * (graph_height - 20))
                cv2.line(display_frame, (graph_x + 5, mean_y), 
                        (graph_x + graph_width - 5, mean_y), (255, 0, 0), 1)
            
            cv2.putText(display_frame, "Intensity History (last 100 frames)", 
                       (graph_x + 5, graph_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        red_display = cv2.resize(red_channel, None, 
                                fx=self.display_scale, 
                                fy=self.display_scale)
        cv2.rectangle(red_display, 
                     (self.roi_display[0], self.roi_display[1]), 
                     (self.roi_display[0]+self.roi_display[2], 
                      self.roi_display[1]+self.roi_display[3]), 
                     255, 3)
        
        return display_frame, red_display
    
    def process_video(self):
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_num += 1
                
                # Measure center intensity.
                current_intensity, red_channel = self.get_center_intensity(frame)
                self.intensity_history.append(current_intensity)
                current_count, _, _ = self.analyze_fringes()
                
                # Draw overlays.
                display_frame, red_display = self.draw_detection_info(
                    frame, red_channel, current_intensity, current_count)
            
            # Show windows.
            cv2.imshow('Michelson Interferometer - Tracking', display_frame)
            cv2.imshow('Red Channel (ROI marked)', red_display)
            
            # Keyboard handling.
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
            
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n" + "="*50)
        final_count, smoothed_data, peaks = self.analyze_fringes()
        
        print(f"Total frames: {self.frame_num}")
        print(f"Detected fringes: {final_count}")
        print(f"Intensity range: {np.min(self.intensity_history):.2f} - {np.max(self.intensity_history):.2f}")
        print(f"Intensity variation: {np.max(self.intensity_history) - np.min(self.intensity_history):.2f}")
        print("="*50)
        
        self.plot_analysis(smoothed_data, peaks)
        
        return final_count
    
    def plot_analysis(self, smoothed_data, peaks):
        """Plot raw signal, smoothed signal, and detected peaks."""
        
        import matplotlib.pyplot as plt
        
        data = np.array(self.intensity_history)
        
        plt.figure(figsize=(14, 7))
        plt.plot(data, color='gray', alpha=0.4, linewidth=1, label='Raw Data')
        plt.plot(smoothed_data, color='red', linewidth=2, label='Smoothed Signal')
        plt.plot(peaks, smoothed_data[peaks], "x", color='blue', 
                markersize=10, markeredgewidth=2, label=f'Fringes Detected ({len(peaks)})')
        
        plt.title(f"Michelson Interferometer Fringe Analysis\nTotal Fringes: {len(peaks)}", 
                 fontsize=14, fontweight='bold')
        plt.xlabel("Frame Number", fontsize=12)
        plt.ylabel("Intensity (Red Channel)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    

def main():
    print("="*50)
    print("Fringe Counting")
    print("="*50)
    
    project_dir = Path(__file__).resolve().parent
    video_path = str(project_dir / "proceed_video.mp4")
    
    try:
        analyzer = MichelsonFringeCounter(video_path)
        total_fringes = analyzer.process_video()
        
        print(f"\nFinal result: detected {total_fringes} fringes.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
