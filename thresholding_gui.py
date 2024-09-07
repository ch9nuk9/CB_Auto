import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, 
                             QSlider, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt
import cv2
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Helper function to load image files
def load_image_file(filepath):
    _, ext = os.path.splitext(filepath.lower())
    
    if ext in ['.tif', '.tiff', '.bigtiff', '.ndtiff', '.omitiff']:  # TIFF and similar formats
        img = tifffile.imread(filepath)
        if img.ndim == 3:  # If it's a stack, we treat it as a multi-frame image
            return img
        return np.expand_dims(img, axis=0)  # Ensure single-frame image is a stack with 1 frame
    
    elif ext in ['.png', '.jpeg', '.jpg']:  # Common image formats
        img = Image.open(filepath)
        img = np.array(img.convert('L'))  # Convert to grayscale
        return np.expand_dims(img, axis=0)  # Ensure single-frame image is a stack with 1 frame
    
    else:
        raise ValueError(f"Unsupported image format: {ext}")

# Processing function (updated with Gaussian and Median blur fix)
def process_images(image_stack, threshold, blur_gaussian, blur_median, iterations):
    processed_stack = []
    
    for img in image_stack:
        # Ensure the blur_gaussian value is a positive odd number or skip blur if 0
        if blur_gaussian > 0:
            if blur_gaussian % 2 == 0:
                blur_gaussian += 1  # Make it odd by adding 1
            img = cv2.GaussianBlur(img, (blur_gaussian, blur_gaussian), 0)
        
        # Ensure the blur_median value is a positive odd number greater than 1 or skip if 0
        if blur_median > 0:
            if blur_median < 3:
                blur_median = 3  # Median blur requires ksize >= 3
            if blur_median % 2 == 0:
                blur_median += 1  # Make it odd by adding 1
            img = cv2.medianBlur(img, blur_median)
        
        # Apply binary threshold
        _, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply binary erosion
        img_eroded = cv2.erode(img_bin, None, iterations=iterations)
        
        # Append the processed frame to the stack
        processed_stack.append(img_eroded)
    
    return np.array(processed_stack)

# PyQt5 Main Window class
class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_stack = None  # Image stack to be processed
        self.processed_stack = None  # Processed image stack
        self.current_frame = 0  # Current frame to display
        
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Image Processing GUI')
        self.setGeometry(200, 100, 800, 600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")  # Dark grey theme

        # Layouts
        main_layout = QVBoxLayout()

        # Load Button
        load_button = QPushButton('Load Images', self)
        load_button.setStyleSheet("background-color: #404040; color: white;")
        load_button.clicked.connect(self.load_images)
        main_layout.addWidget(load_button)

        # Threshold slider with label
        self.threshold_slider, self.threshold_label = self.create_slider("Threshold", 0, 255, 128)
        main_layout.addLayout(self.threshold_slider[1])

        # Gaussian Blur slider with label
        self.gaussian_slider, self.gaussian_label = self.create_slider("Gaussian Blur", 0, 10, 1)
        main_layout.addLayout(self.gaussian_slider[1])

        # Median Blur slider with label
        self.median_slider, self.median_label = self.create_slider("Median Blur", 0, 10, 1)
        main_layout.addLayout(self.median_slider[1])

        # Erosion Iteration slider with label
        self.erosion_slider, self.erosion_label = self.create_slider("Erosion Iterations", 0, 10, 1)
        main_layout.addLayout(self.erosion_slider[1])

        # Frame Slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.update_frame)
        main_layout.addWidget(self.frame_slider)

        # Image Display Canvas
        self.canvas = FigureCanvas(plt.Figure())
        main_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()

        # Save settings button
        save_button = QPushButton('Save Threshold Settings', self)
        save_button.setStyleSheet("background-color: #404040; color: white;")
        save_button.clicked.connect(self.save_threshold_settings)
        main_layout.addWidget(save_button)

        # Main Widget
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Connect sliders to real-time updates
        self.threshold_slider[0].valueChanged.connect(self.update_image)
        self.gaussian_slider[0].valueChanged.connect(self.update_image)
        self.median_slider[0].valueChanged.connect(self.update_image)
        self.erosion_slider[0].valueChanged.connect(self.update_image)
    
    def create_slider(self, label, min_val, max_val, default_val):
        layout = QHBoxLayout()
        slider_label = QLabel(label)
        slider_label.setStyleSheet("color: white;")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        
        # Value label
        value_label = QLabel(str(default_val))
        value_label.setStyleSheet("color: white;")
        slider.valueChanged.connect(lambda value: value_label.setText(str(value)))  # Update label dynamically

        layout.addWidget(slider_label)
        layout.addWidget(slider)
        layout.addWidget(value_label)
        return (slider, layout), value_label  # Return slider and value label for easy updates

    def load_images(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilters([
            "TIFF files (*.tif *.tiff *.bigtiff *.ndtiff *.omitiff)",
            "Image files (*.png *.jpeg *.jpg)"
        ])
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.image_stack = load_image_file(file_path)
            self.frame_slider.setMaximum(len(self.image_stack) - 1)
            self.update_image()  # Process and update the image immediately
    
    def update_image(self):
        if self.image_stack is None:
            return
        
        # Access the value directly from the slider widgets
        threshold = self.threshold_slider[0].value()  # Access the slider from tuple
        blur_gaussian = self.gaussian_slider[0].value()
        blur_median = self.median_slider[0].value()
        iterations = self.erosion_slider[0].value()
        
        # Process the image stack using the slider values
        self.processed_stack = process_images(
            self.image_stack,
            threshold=threshold,
            blur_gaussian=blur_gaussian,
            blur_median=blur_median,
            iterations=iterations
        )
        
        # Update the frame display
        self.update_frame(self.frame_slider.value())
    
    def update_frame(self, frame_index):
        if self.processed_stack is not None:
            frame = self.processed_stack[frame_index]
            self.ax.clear()
            # Normalize the image for better visibility
            self.ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
            self.ax.axis('off')
            self.canvas.draw()

    def save_threshold_settings(self):
        # Get current slider values
        threshold = self.threshold_slider[0].value()
        blur_gaussian = self.gaussian_slider[0].value()
        blur_median = self.median_slider[0].value()
        iterations = self.erosion_slider[0].value()

        # Create a DataFrame to store the settings
        settings_df = pd.DataFrame({
            "Threshold": [threshold],
            "Gaussian_Blur": [blur_gaussian],
            "Median_Blur": [blur_median],
            "Erosion_Iterations": [iterations]
        })

        # Save to CSV
        file_dialog = QFileDialog(self)
        save_path, _ = file_dialog.getSaveFileName(self, "Save Threshold Settings", "", "CSV Files (*.csv)")
        if save_path:
            settings_df.to_csv(save_path, index=False)
            QMessageBox.information(self, "Settings Saved", f"Threshold settings saved to {save_path}")

# Main entry point
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessingApp()
    ex.show()
    sys.exit(app.exec_())
