import sys
import cv2
import numpy as np
import logging
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QWidget, QCheckBox, QSlider, QLabel as QtLabel, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QThread

# Import qdarkstyle for dark theme
import qdarkstyle  # Added import for qdarkstyle

# Setup logging for debugging and error handling
logging.basicConfig(
    filename='app_log.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Define Kalman Filter class for predicting weigh-points
class KalmanFilter:
    def __init__(self):
        """
        Initialize the Kalman filter parameters.
        """
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            np.float32
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05

    def predict(self, pt):
        """
        Predict the next position of a weigh-point using the Kalman filter.

        Args:
            pt (tuple): Current position (x, y) of the weigh-point.

        Returns:
            tuple: Predicted position (x, y) of the weigh-point.
        """
        measurement = np.array([[np.float32(pt[0])], [np.float32(pt[1])]])
        prediction = self.kalman.predict()
        self.kalman.correct(measurement)
        return (prediction[0][0], prediction[1][0])

# Worker Thread Class for video processing
class VideoProcessor(QThread):
    frame_processed = pyqtSignal(np.ndarray)  # Signal to send processed frames to GUI

    def __init__(self, video_path, output_directory, track_single_point, save_frames, parameters):
        """
        Initialize the video processor thread.

        Args:
            video_path (str): Path to the input video file.
            output_directory (str): Directory to save processed frames.
            track_single_point (bool): Whether to track a single weigh-point or two.
            save_frames (bool): Whether to save processed frames to disk.
            parameters (dict): Dictionary of adjustable parameters.
        """
        super().__init__()
        self.video_path = video_path
        self.output_directory = output_directory
        self.track_single_point = track_single_point
        self.save_frames = save_frames
        self.parameters = parameters  # Store adjustable parameters
        # Initialize Kalman filters for each weigh-point
        if self.track_single_point:
            self.kalman_filters = [KalmanFilter()]
        else:
            self.kalman_filters = [KalmanFilter(), KalmanFilter()]
        self.is_running = True  # Flag to control thread execution

    def run(self):
        """
        Main method to process the video frame by frame.
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video file: {self.video_path}")
                return

            frame_counter = 0

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale and apply pre-processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Enhancement: Contrast Limited Adaptive Histogram Equalization (CLAHE)
                if self.parameters['use_clahe']:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)

                # Apply Gaussian Blur with adjustable kernel size
                blurred_frame = cv2.GaussianBlur(
                    gray,
                    (self.parameters['gaussian_blur_kernel'], self.parameters['gaussian_blur_kernel']),
                    0
                )

                # Enhancement: Adaptive Thresholding
                if self.parameters['use_adaptive_threshold']:
                    adaptive_thresh = cv2.adaptiveThreshold(
                        blurred_frame,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        self.parameters['adaptive_thresh_block_size'],
                        self.parameters['adaptive_thresh_C']
                    )
                    preprocessed_frame = adaptive_thresh
                else:
                    preprocessed_frame = blurred_frame

                # Enhancement: Dynamic Canny Edge Detection thresholds
                if self.parameters['use_dynamic_canny']:
                    v = np.median(preprocessed_frame)
                    sigma = self.parameters['canny_sigma'] / 100.0
                    lower = int(max(0, (1.0 - sigma) * v))
                    upper = int(min(255, (1.0 + sigma) * v))
                    edges = cv2.Canny(preprocessed_frame, lower, upper)
                else:
                    edges = cv2.Canny(
                        preprocessed_frame,
                        self.parameters['canny_threshold1'],
                        self.parameters['canny_threshold2']
                    )

                # Apply morphological closing to close gaps in the edges
                kernel_size = self.parameters['morph_kernel_size']
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                # Enhancement: Skeletonization (if enabled)
                if self.parameters['use_skeletonization']:
                    binary_image = closed_edges // 255  # Convert to binary (0 or 1)
                    skeleton = self.skeletonize(binary_image)
                    skeleton = (skeleton * 255).astype(np.uint8)
                    processed_edges = skeleton
                else:
                    processed_edges = closed_edges

                # Find contours on processed edges
                contours = self.find_contours(processed_edges)
                if not contours:
                    logging.warning(f"No contours found in frame {frame_counter}")
                    continue

                # Filter and keep the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                contours = [largest_contour]

                # Enhancement: Contour Approximation
                if self.parameters['use_contour_approx']:
                    epsilon = self.parameters['contour_approx_epsilon'] * cv2.arcLength(largest_contour, True)
                    largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                    contours = [largest_contour]

                # Draw the worm's outline (contours)
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

                # Initialize weigh-points
                weigh_points = self.get_weigh_points(largest_contour)

                # Predict positions using Kalman filters
                predicted_weigh_points = []
                for i, (kf, pt) in enumerate(zip(self.kalman_filters, weigh_points)):
                    predicted_pt = kf.predict(pt)
                    # Snap the predicted point to the nearest contour point
                    closest_point = min(
                        largest_contour,
                        key=lambda c: np.linalg.norm(c[0] - np.array([predicted_pt[0], predicted_pt[1]]))
                    )
                    predicted_weigh_points.append((closest_point[0][0], closest_point[0][1]))

                # Draw the predicted weigh-points on the frame
                for point in predicted_weigh_points:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

                # Save the processed frame with weigh-points if saving is enabled
                if self.save_frames:
                    save_path = os.path.join(
                        self.output_directory,
                        f"frame_{frame_counter:04d}.png"
                    )
                    cv2.imwrite(save_path, frame)
                    logging.info(f"Saved frame {frame_counter} to {save_path}")

                # Emit the processed frame to update the GUI
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_processed.emit(rgb_frame)

                frame_counter += 1

            cap.release()
            logging.info("Video processing completed.")
        except Exception as e:
            logging.error(f"Error processing video: {e}")
        finally:
            if cap.isOpened():
                cap.release()

    def stop(self):
        """
        Stop the video processing thread.
        """
        self.is_running = False

    def apply_canny_edge_detection(self, gray_img):
        """
        Apply Canny edge detection to a grayscale image.

        Args:
            gray_img (np.ndarray): Grayscale image.

        Returns:
            np.ndarray: Image after edge detection.
        """
        try:
            edges = cv2.Canny(gray_img, 40, 120)
            return edges
        except Exception as e:
            logging.error(f"Failed to apply Canny edge detection: {e}")
            return np.array([])

    def find_contours(self, edges):
        """
        Find contours in an edge-detected image.

        Args:
            edges (np.ndarray): Edge-detected image.

        Returns:
            list: List of contours found.
        """
        try:
            # Adjusted to handle OpenCV version differences
            contours_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_info) == 2:
                contours, _ = contours_info
            else:
                _, contours, _ = contours_info
            return contours
        except Exception as e:
            logging.error(f"Failed to find contours: {e}")
            return []

    def get_weigh_points(self, contour):
        """
        Determine the weigh-points based on the selected tracking mode.

        Args:
            contour (np.ndarray): The largest contour of the worm.

        Returns:
            list: List of weigh-points to track.
        """
        try:
            if self.track_single_point:
                # Track a single weigh-point (e.g., centroid)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                    weigh_points = [(centroid_x, centroid_y)]
                else:
                    weigh_points = [(0, 0)]
            else:
                # Track two weigh-points (e.g., head and tail)
                leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
                rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
                weigh_points = [leftmost, rightmost]
            return weigh_points
        except Exception as e:
            logging.error(f"Failed to get weigh-points: {e}")
            return [(0, 0)]

    def skeletonize(self, binary_image):
        """
        Perform skeletonization on a binary image.

        Args:
            binary_image (np.ndarray): Binary image (0 or 1).

        Returns:
            np.ndarray: Skeletonized image.
        """
        try:
            from skimage.morphology import skeletonize
            skeleton = skeletonize(binary_image)
            return skeleton
        except ImportError:
            logging.error("scikit-image is not installed. Install it to use skeletonization.")
            return binary_image

# Main GUI class for PyQt5 application
class WormApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("C. Elegans Outline and Weigh-point Detection")
        self.setGeometry(100, 100, 800, 600)

        # Setup GUI components
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)

        self.save_button = QPushButton("Select Output Directory", self)
        self.save_button.clicked.connect(self.select_output_directory)

        self.process_button = QPushButton("Process Video", self)
        self.process_button.clicked.connect(self.process_video_multithread)

        # Enhancement: Add checkbox to allow user to select tracking mode
        self.track_single_point_checkbox = QCheckBox("Track Single Point", self)
        self.track_single_point_checkbox.setChecked(True)
        self.track_single_point_checkbox.stateChanged.connect(self.set_tracking_mode)

        # Enhancement: Add checkbox to allow user to enable or disable saving frames
        self.save_frames_checkbox = QCheckBox("Save Processed Frames", self)
        self.save_frames_checkbox.setChecked(True)
        self.save_frames_checkbox.stateChanged.connect(self.set_save_frames)

        # Enhancement: Add controls for adjustable parameters
        self.parameters = {
            'use_clahe': False,
            'gaussian_blur_kernel': 7,
            'use_adaptive_threshold': False,
            'adaptive_thresh_block_size': 11,
            'adaptive_thresh_C': 2,
            'use_dynamic_canny': False,
            'canny_threshold1': 40,
            'canny_threshold2': 120,
            'canny_sigma': 33,  # Percentage for dynamic Canny
            'morph_kernel_size': 3,
            'use_skeletonization': False,
            'use_contour_approx': False,
            'contour_approx_epsilon': 0.01,
        }

        self.clahe_checkbox = QCheckBox("Use CLAHE (Contrast Enhancement)", self)
        self.clahe_checkbox.setChecked(self.parameters['use_clahe'])
        self.clahe_checkbox.stateChanged.connect(self.set_clahe)

        self.adaptive_thresh_checkbox = QCheckBox("Use Adaptive Thresholding", self)
        self.adaptive_thresh_checkbox.setChecked(self.parameters['use_adaptive_threshold'])
        self.adaptive_thresh_checkbox.stateChanged.connect(self.set_adaptive_threshold)

        self.dynamic_canny_checkbox = QCheckBox("Use Dynamic Canny Thresholds", self)
        self.dynamic_canny_checkbox.setChecked(self.parameters['use_dynamic_canny'])
        self.dynamic_canny_checkbox.stateChanged.connect(self.set_dynamic_canny)

        self.skeletonization_checkbox = QCheckBox("Use Skeletonization", self)
        self.skeletonization_checkbox.setChecked(self.parameters['use_skeletonization'])
        self.skeletonization_checkbox.stateChanged.connect(self.set_skeletonization)

        self.contour_approx_checkbox = QCheckBox("Use Contour Approximation", self)
        self.contour_approx_checkbox.setChecked(self.parameters['use_contour_approx'])
        self.contour_approx_checkbox.stateChanged.connect(self.set_contour_approx)

        # Sliders for adjustable parameters
        self.gaussian_blur_slider = self.create_slider(
            label_text="Gaussian Blur Kernel Size",
            min_val=1,
            max_val=31,
            init_val=self.parameters['gaussian_blur_kernel'],
            callback=self.set_gaussian_blur_kernel,
            step=2  # Kernel size must be odd
        )

        self.morph_kernel_slider = self.create_slider(
            label_text="Morphological Kernel Size",
            min_val=1,
            max_val=31,
            init_val=self.parameters['morph_kernel_size'],
            callback=self.set_morph_kernel_size,
            step=2  # Kernel size must be odd
        )

        self.canny_threshold_slider = self.create_slider(
            label_text="Canny Sigma (%)",
            min_val=1,
            max_val=100,
            init_val=self.parameters['canny_sigma'],
            callback=self.set_canny_sigma
        )

        self.contour_approx_slider = self.create_slider(
            label_text="Contour Approximation Epsilon (%)",
            min_val=1,
            max_val=10,
            init_val=int(self.parameters['contour_approx_epsilon'] * 100),
            callback=self.set_contour_approx_epsilon
        )

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.save_button)

        # Parameter controls layout
        param_layout = QVBoxLayout()
        param_layout.addWidget(self.track_single_point_checkbox)
        param_layout.addWidget(self.save_frames_checkbox)
        param_layout.addWidget(self.clahe_checkbox)
        param_layout.addWidget(self.adaptive_thresh_checkbox)
        param_layout.addWidget(self.dynamic_canny_checkbox)
        param_layout.addWidget(self.skeletonization_checkbox)
        param_layout.addWidget(self.contour_approx_checkbox)
        param_layout.addLayout(self.gaussian_blur_slider['layout'])
        param_layout.addLayout(self.morph_kernel_slider['layout'])
        param_layout.addLayout(self.canny_threshold_slider['layout'])
        param_layout.addLayout(self.contour_approx_slider['layout'])

        layout.addLayout(param_layout)
        layout.addWidget(self.process_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.video_path = None
        self.output_directory = None

        # User preferences for single or two weigh-points
        self.track_single_point = True  # Default tracking mode
        self.save_frames = True  # Default to saving frames

        self.video_thread = None  # Reference to the video processing thread

    def create_slider(self, label_text, min_val, max_val, init_val, callback, step=1):
        """
        Create a slider with a label.

        Args:
            label_text (str): Label text for the slider.
            min_val (int): Minimum value of the slider.
            max_val (int): Maximum value of the slider.
            init_val (int): Initial value of the slider.
            callback (function): Function to call when the slider value changes.
            step (int): Step size of the slider.

        Returns:
            dict: Contains the layout and the slider widget.
        """
        label = QtLabel(f"{label_text}: {init_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.setSingleStep(step)
        slider.valueChanged.connect(lambda val: [label.setText(f"{label_text}: {val}"), callback(val)])
        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(slider)
        return {'layout': layout, 'slider': slider}

    @pyqtSlot(int)
    def set_tracking_mode(self, state):
        """
        Set the tracking mode based on the checkbox state.

        Args:
            state (int): State of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.track_single_point = state == Qt.Checked
        logging.info(f"Tracking mode set to {'single' if self.track_single_point else 'two'} weigh-point(s).")

    @pyqtSlot(int)
    def set_save_frames(self, state):
        """
        Set whether to save processed frames based on the checkbox state.

        Args:
            state (int): State of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.save_frames = state == Qt.Checked
        logging.info(f"Save frames set to {self.save_frames}.")

    @pyqtSlot(int)
    def set_clahe(self, state):
        """
        Set the use of CLAHE based on the checkbox state.

        Args:
            state (int): State of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.parameters['use_clahe'] = state == Qt.Checked
        logging.info(f"CLAHE set to {self.parameters['use_clahe']}.")

    @pyqtSlot(int)
    def set_adaptive_threshold(self, state):
        """
        Set the use of adaptive thresholding based on the checkbox state.

        Args:
            state (int): State of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.parameters['use_adaptive_threshold'] = state == Qt.Checked
        logging.info(f"Adaptive Thresholding set to {self.parameters['use_adaptive_threshold']}.")

    @pyqtSlot(int)
    def set_dynamic_canny(self, state):
        """
        Set the use of dynamic Canny thresholds based on the checkbox state.

        Args:
            state (int): State of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.parameters['use_dynamic_canny'] = state == Qt.Checked
        logging.info(f"Dynamic Canny Thresholds set to {self.parameters['use_dynamic_canny']}.")

    @pyqtSlot(int)
    def set_skeletonization(self, state):
        """
        Set the use of skeletonization based on the checkbox state.

        Args:
            state (int): State of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.parameters['use_skeletonization'] = state == Qt.Checked
        logging.info(f"Skeletonization set to {self.parameters['use_skeletonization']}.")

    @pyqtSlot(int)
    def set_contour_approx(self, state):
        """
        Set the use of contour approximation based on the checkbox state.

        Args:
            state (int): State of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.parameters['use_contour_approx'] = state == Qt.Checked
        logging.info(f"Contour Approximation set to {self.parameters['use_contour_approx']}.")

    @pyqtSlot(int)
    def set_gaussian_blur_kernel(self, value):
        """
        Set the Gaussian blur kernel size.

        Args:
            value (int): New kernel size (must be odd).
        """
        # Ensure the kernel size is odd
        if value % 2 == 0:
            value += 1
        self.parameters['gaussian_blur_kernel'] = value
        logging.info(f"Gaussian Blur Kernel Size set to {value}.")

    @pyqtSlot(int)
    def set_morph_kernel_size(self, value):
        """
        Set the morphological kernel size.

        Args:
            value (int): New kernel size (must be odd).
        """
        # Ensure the kernel size is odd
        if value % 2 == 0:
            value += 1
        self.parameters['morph_kernel_size'] = value
        logging.info(f"Morphological Kernel Size set to {value}.")

    @pyqtSlot(int)
    def set_canny_sigma(self, value):
        """
        Set the sigma percentage for dynamic Canny thresholds.

        Args:
            value (int): New sigma percentage.
        """
        self.parameters['canny_sigma'] = value
        logging.info(f"Canny Sigma set to {value}%.")

    @pyqtSlot(int)
    def set_contour_approx_epsilon(self, value):
        """
        Set the epsilon percentage for contour approximation.

        Args:
            value (int): New epsilon percentage.
        """
        self.parameters['contour_approx_epsilon'] = value / 100.0
        logging.info(f"Contour Approximation Epsilon set to {self.parameters['contour_approx_epsilon']}.")

    @pyqtSlot()
    def load_video(self):
        """
        Load video file using a file dialog.
        """
        try:
            options = QFileDialog.Options()
            self.video_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Video File",
                "",
                "Video Files (*.mp4 *.avi)",
                options=options
            )
            if self.video_path:
                logging.info(f"Loaded video: {self.video_path}")
        except Exception as e:
            logging.error(f"Failed to load video: {e}")

    @pyqtSlot()
    def select_output_directory(self):
        """
        Select output directory for saving results using a directory dialog.
        """
        try:
            options = QFileDialog.Options()
            self.output_directory = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory",
                options=options
            )
            if self.output_directory:
                logging.info(f"Selected output directory: {self.output_directory}")
        except Exception as e:
            logging.error(f"Failed to select output directory: {e}")

    def process_video_multithread(self):
        """
        Start the video processing in a separate thread.
        """
        if not self.video_path or not self.output_directory:
            logging.error("No video file or output directory selected.")
            return

        # Stop any existing video processing thread
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()

        # Start a new video processing thread
        self.video_thread = VideoProcessor(
            self.video_path,
            self.output_directory,
            self.track_single_point,
            self.save_frames,
            self.parameters
        )
        self.video_thread.frame_processed.connect(self.display_frame)
        self.video_thread.start()
        logging.info("Video processing started.")

    @pyqtSlot(np.ndarray)
    def display_frame(self, img):
        """
        Display a frame on the PyQt GUI.

        Args:
            img (np.ndarray): Image to display.
        """
        try:
            # Enhancement: Ensure correct image format and bytes per line
            if len(img.shape) == 2:
                # Grayscale image
                q_img = QImage(
                    img.data,
                    img.shape[1],
                    img.shape[0],
                    img.shape[1],
                    QImage.Format_Grayscale8
                )
            else:
                # Color image
                h, w, ch = img.shape
                bytes_per_line = ch * w
                q_img = QImage(
                    img.data,
                    w,
                    h,
                    bytes_per_line,
                    QImage.Format_RGB888
                )
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"Failed to display frame: {e}")

    def closeEvent(self, event):
        """
        Override the close event to ensure the video thread is stopped.

        Args:
            event (QCloseEvent): The close event.
        """
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()

# Main function to run the PyQt5 application
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)

        # Apply QDarkStyleSheet for dark theme
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())  # Added line to set dark theme

        window = WormApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
