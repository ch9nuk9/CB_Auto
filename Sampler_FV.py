import sys
import os
import shutil
import cv2
import random
import uuid
import csv
import logging
import math
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QTextEdit, QProgressBar, QSpinBox, QComboBox, QMessageBox, QGroupBox,
    QGridLayout, QHBoxLayout, QSizePolicy, QMainWindow, QAction, QMenuBar,
    QMenu, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap

# Set up logging to log messages to a file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_log.log'),
        logging.StreamHandler()
    ]
)

try:
    import qdarkstyle
except ImportError:
    logging.warning("qdarkstyle not installed. Install it for a better dark theme experience.")

# Helper function to format exceptions
def format_exception(e):
    return f"{type(e).__name__}: {str(e)}"

class MetadataLogger(QThread):
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, source_dir, log_file_path):
        super().__init__()
        self.source_dir = source_dir
        self.log_file_path = log_file_path
        self.is_running = True

    def extract_metadata(self, file_path):
        """Extracts metadata such as duration, frame count, and frame rate from the video file."""
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                self.log_message.emit(f"Failed to open video file: {file_path}")
                return {'duration': 0, 'frames': 0, 'frame_rate': 0}

            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            duration = frames / frame_rate if frame_rate > 0 else 0
            cap.release()
            return {'duration': duration, 'frames': frames, 'frame_rate': frame_rate}
        except Exception as e:
            logging.error(f"Error extracting metadata: {format_exception(e)}")
            self.error_occurred.emit(f"Error extracting metadata from {file_path}: {format_exception(e)}")
            return {'duration': 0, 'frames': 0, 'frame_rate': 0}

    def run(self):
        """Logs metadata of each video file in the directory structure."""
        try:
            with open(self.log_file_path, mode='w', newline='') as log_file:
                writer = csv.writer(log_file)
                writer.writerow(['Date', 'Time', 'Experimental Condition', 'Session Subfolder',
                                 'Track ID', 'File Path', 'Duration', 'Frame Count', 'Frame Rate'])

                # Traverse the directory structure to find video files in output folders
                for root, dirs, files in os.walk(self.source_dir):
                    if os.path.basename(root) == "output":  # Look for the 'output' folder
                        folder_info = self.get_folder_info(root)
                        if folder_info is None:
                            continue  # Skip if folder info couldn't be parsed
                        date, time, condition, session_subfolder, track_id = folder_info
                        for file in files:
                            if file.lower().endswith(('.avi', '.mp4', '.mov')):
                                file_path = os.path.join(root, file)
                                metadata = self.extract_metadata(file_path)
                                writer.writerow([date, time, condition, session_subfolder, track_id,
                                                 file_path, metadata['duration'], metadata['frames'], metadata['frame_rate']])
                                self.log_message.emit(f"Logged: {file_path}")
                                if not self.is_running:
                                    break
                    if not self.is_running:
                        break
            self.log_message.emit("Metadata logging completed.")
        except Exception as e:
            logging.error(f"Error during metadata logging: {format_exception(e)}")
            self.error_occurred.emit(f"Error during metadata logging: {format_exception(e)}")

    def stop(self):
        """Stops the thread gracefully."""
        self.is_running = False

    def get_folder_info(self, root):
        """Extracts the date, time, experimental condition, session subfolder, and track ID from the directory path."""
        try:
            parts = root.split(os.sep)

            # Find the index of 'output' in the path
            output_index = parts.index('output')

            # Extract track ID
            track_id = parts[output_index - 1]

            # Extract session subfolder
            session_subfolder = parts[output_index - 2]

            # Extract session folder name
            session_folder = parts[output_index - 3]

            # Extract date, time, and experimental condition from session folder name
            # Assuming the format is 'YYYY-MM-DD_HH-MM-SS_condition'
            session_info = session_folder.split('_')
            date = session_info[0]  # 'YYYY-MM-DD'
            time = session_info[1]  # 'HH-MM-SS'
            condition = '_'.join(session_info[2:])  # 'condition'

            return date, time, condition, session_subfolder, track_id
        except Exception as e:
            logging.error(f"Error parsing folder info: {format_exception(e)}")
            self.log_message.emit(f"Error parsing folder info in path: {root}")
            return None

class Randomizer(QThread):
    update_progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, log_file_path, total_files_to_select, output_dir):
        super().__init__()
        self.log_file_path = log_file_path
        self.total_files_to_select = total_files_to_select  # Total number of tracks to sample
        self.output_dir = output_dir
        self.is_running = True

    def run(self):
        """Performs proportional allocation and sampling, then copies selected files."""
        try:
            sessions = self.group_files_by_session()
            selected_files = []
            total_tracks = sum([len(files) for files in sessions.values()])

            if total_tracks == 0:
                self.log_message.emit("No tracks found in the metadata log.")
                return

            # Ensure total_files_to_select does not exceed total_tracks
            if self.total_files_to_select > total_tracks:
                self.total_files_to_select = total_tracks
                self.log_message.emit(f"Adjusted total files to select to {total_tracks} due to limited tracks.")

            # Proportional allocation: Sample files proportionally from each session
            for session_id, files in sessions.items():
                proportion = len(files) / total_tracks
                num_files_to_sample = max(1, int(self.total_files_to_select * proportion))
                num_files_to_sample = min(num_files_to_sample, len(files))  # Edge case handling

                # Uniform random sampling within the session
                selected = random.sample(files, k=num_files_to_sample)
                selected_files.extend(selected)
                if not self.is_running:
                    break

            # Copy selected files to the output directory
            self.copy_selected_files(selected_files)
            if self.is_running:
                self.log_message.emit("Randomization and copying completed.")
        except Exception as e:
            logging.error(f"Error during randomization: {format_exception(e)}")
            self.error_occurred.emit(f"Error during randomization: {format_exception(e)}")

    def stop(self):
        """Stops the thread gracefully."""
        self.is_running = False

    def group_files_by_session(self):
        """Groups files by session from the log file."""
        sessions = {}
        try:
            with open(self.log_file_path, mode='r') as log_file:
                reader = csv.DictReader(log_file)
                for row in reader:
                    session_id = f"{row['Date']}_{row['Time']}_{row['Experimental Condition']}"
                    if session_id not in sessions:
                        sessions[session_id] = []
                    sessions[session_id].append(row)
            return sessions
        except Exception as e:
            logging.error(f"Error grouping files by session: {format_exception(e)}")
            self.error_occurred.emit(f"Error reading metadata log: {format_exception(e)}")
            return {}

    def copy_selected_files(self, selected_files):
        """Copies selected files to the output directory."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)  # Ensure that the output directory exists

            total_files = len(selected_files)
            for idx, file in enumerate(selected_files, start=1):
                if not self.is_running:
                    break
                date = file['Date']
                time = file['Time']
                condition = file['Experimental Condition']
                session_subfolder = file['Session Subfolder']
                track_id = file['Track ID']
                file_path = file['File Path']

                unique_id = str(uuid.uuid4())
                new_file_name = f"{date}_{time}_{condition}_{session_subfolder}_{track_id}_{unique_id}.avi"
                new_file_path = os.path.join(self.output_dir, new_file_name)

                try:
                    # Use buffered copying for large files
                    with open(file_path, 'rb') as src_file, open(new_file_path, 'wb') as dest_file:
                        shutil.copyfileobj(src_file, dest_file, length=1024*1024)  # 1MB buffer
                    self.log_message.emit(f"Copied: {new_file_name}")
                except Exception as e:
                    logging.error(f"Error copying {file_path}: {format_exception(e)}")
                    self.log_message.emit(f"Error copying {file_path}: {format_exception(e)}")

                progress = int((idx / total_files) * 100)
                self.update_progress.emit(progress)
            if not self.is_running:
                self.log_message.emit("Copying process was stopped by the user.")
        except Exception as e:
            logging.error(f"Error during file copying: {format_exception(e)}")
            self.error_occurred.emit(f"Error during file copying: {format_exception(e)}")

class AVIFileRandomizer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the UI when the object is created
        self.initUI()

    def initUI(self):
        """Sets up the GUI layout and elements."""
        self.setWindowTitle('AVI File Randomizer')
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon('icon.png'))

        # Apply dark theme if qdarkstyle is available
        try:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        except Exception:
            # Fallback styling
            self.setStyleSheet("""
                QWidget {
                    background-color: #2b2b2b;
                    color: #f1f1f1;
                }
                QPushButton {
                    background-color: #444444;
                    color: #ffffff;
                    font-size: 14px;
                }
            """)

        # Create a central widget and set layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Menu bar
        self.create_menu_bar()

        # Directory selection group
        dir_group = QGroupBox("Directories")
        dir_layout = QGridLayout()
        dir_group.setLayout(dir_layout)

        # Source directory selection
        self.dir_label = QLabel('Select a Source Directory:')
        self.dir_button = QPushButton('Browse...')
        self.dir_button.clicked.connect(self.browse_directory)
        self.selected_dir_label = QLabel('No directory selected')

        dir_layout.addWidget(self.dir_label, 0, 0)
        dir_layout.addWidget(self.dir_button, 0, 1)
        dir_layout.addWidget(self.selected_dir_label, 1, 0, 1, 2)

        # Output directory selection
        self.output_label = QLabel('Select Output Directory:')
        self.output_button = QPushButton('Browse...')
        self.output_button.clicked.connect(self.browse_output_directory)
        self.output_dir_label = QLabel('No output directory selected')

        dir_layout.addWidget(self.output_label, 2, 0)
        dir_layout.addWidget(self.output_button, 2, 1)
        dir_layout.addWidget(self.output_dir_label, 3, 0, 1, 2)

        main_layout.addWidget(dir_group)

        # Sampling parameters group
        param_group = QGroupBox("Sampling Parameters")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)

        # Total number of tracks to sample
        self.num_total_files_spinbox = QSpinBox(self)
        self.num_total_files_spinbox.setRange(1, 1000000)
        self.num_total_files_spinbox.setValue(100)
        self.num_total_files_spinbox.setSingleStep(10)
        self.num_total_files_spinbox.setToolTip("Set the total number of tracks to sample.")

        param_layout.addWidget(QLabel('Total number of tracks to sample:'), 0, 0)
        param_layout.addWidget(self.num_total_files_spinbox, 0, 1)

        # Sample size calculator inputs
        self.confidence_level_combo = QComboBox(self)
        self.confidence_level_combo.addItems(['90%', '95%', '99%'])
        self.confidence_level_combo.setCurrentIndex(1)  # Default to 95%
        self.margin_of_error_spinbox = QDoubleSpinBox(self)
        self.margin_of_error_spinbox.setRange(0.1, 20.0)
        self.margin_of_error_spinbox.setValue(5.0)
        self.margin_of_error_spinbox.setSuffix('%')
        self.margin_of_error_spinbox.setDecimals(1)
        self.margin_of_error_spinbox.setSingleStep(0.5)
        self.variability_spinbox = QDoubleSpinBox(self)
        self.variability_spinbox.setRange(0.0, 1.0)
        self.variability_spinbox.setValue(0.5)
        self.variability_spinbox.setDecimals(2)
        self.variability_spinbox.setSingleStep(0.05)

        self.calculate_sample_size_button = QPushButton('Calculate Sample Size')
        self.calculate_sample_size_button.clicked.connect(self.calculate_sample_size)

        param_layout.addWidget(QLabel('Confidence Level:'), 1, 0)
        param_layout.addWidget(self.confidence_level_combo, 1, 1)
        param_layout.addWidget(QLabel('Margin of Error:'), 2, 0)
        param_layout.addWidget(self.margin_of_error_spinbox, 2, 1)
        param_layout.addWidget(QLabel('Estimated Variability (P):'), 3, 0)
        param_layout.addWidget(self.variability_spinbox, 3, 1)
        param_layout.addWidget(self.calculate_sample_size_button, 4, 0, 1, 2)

        main_layout.addWidget(param_group)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_bar)

        # Log output area
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier", 10))
        self.log_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.log_output)

        # Control buttons
        button_layout = QHBoxLayout()
        self.log_button = QPushButton('Log Metadata')
        self.log_button.clicked.connect(self.log_metadata)
        self.run_button = QPushButton('Run Randomizer')
        self.run_button.clicked.connect(self.run_randomizer)
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_process)

        button_layout.addWidget(self.log_button)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)

        main_layout.addLayout(button_layout)

        # Initialize variables
        self.source_dir = None
        self.output_dir = None
        self.logger_thread = None
        self.randomizer_thread = None
        self.total_population_size = 0  # Total number of tracks

    def create_menu_bar(self):
        """Creates the menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menu_bar.addMenu('Help')

        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_about_dialog(self):
        """Displays the about dialog."""
        QMessageBox.information(self, "About AVI File Randomizer",
                                "AVI File Randomizer\nVersion 1.1\nDeveloped by [Your Name]")

    @pyqtSlot()
    def browse_directory(self):
        """Opens a dialog to allow the user to select a source directory."""
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Source Directory')
        if dir_path and os.path.exists(dir_path):
            self.selected_dir_label.setText(f'Selected Directory: {dir_path}')
            self.source_dir = dir_path
            logging.info(f"Source directory selected: {dir_path}")
        else:
            QMessageBox.warning(self, "Invalid Directory", "The selected directory does not exist.")

    @pyqtSlot()
    def browse_output_directory(self):
        """Opens a dialog to allow the user to select an output directory."""
        output_dir = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if output_dir and os.path.exists(output_dir):
            self.output_dir_label.setText(f'Selected Output Directory: {output_dir}')
            self.output_dir = output_dir
            logging.info(f"Output directory selected: {output_dir}")
        else:
            QMessageBox.warning(self, "Invalid Directory", "The selected directory does not exist.")

    @pyqtSlot()
    def log_metadata(self):
        """Starts the metadata logging process in a separate thread."""
        if self.source_dir:
            self.log_output.append("Starting metadata logging...")
            self.progress_bar.setValue(0)
            self.log_button.setEnabled(False)
            self.run_button.setEnabled(False)

            log_file_path = os.path.join(self.source_dir, 'metadata_log.csv')
            self.logger_thread = MetadataLogger(self.source_dir, log_file_path)
            self.logger_thread.log_message.connect(self.log_output.append)
            self.logger_thread.error_occurred.connect(self.display_error)
            self.logger_thread.finished.connect(self.on_logging_finished)
            self.logger_thread.start()
        else:
            QMessageBox.warning(self, "No Source Directory", "Please select a source directory first.")

    @pyqtSlot()
    def run_randomizer(self):
        """Starts the randomizer process in a separate thread."""
        if self.source_dir and self.output_dir:
            log_file_path = os.path.join(self.source_dir, 'metadata_log.csv')

            # Check if metadata log exists
            if not os.path.exists(log_file_path):
                QMessageBox.warning(self, "Metadata Log Missing", "Please run metadata logging first.")
                return

            total_files = self.num_total_files_spinbox.value()

            # Get total tracks available
            total_tracks_available = self.get_total_tracks_available(log_file_path)
            if total_tracks_available == 0:
                QMessageBox.warning(self, "No Tracks Found", "No tracks found in the metadata log.")
                return

            if total_files > total_tracks_available:
                QMessageBox.warning(self, "Sampling Error",
                                    f"Requested {total_files} tracks but only {total_tracks_available} are available.")
                total_files = total_tracks_available
                self.num_total_files_spinbox.setValue(total_files)

            self.progress_bar.setValue(0)
            self.run_button.setEnabled(False)
            self.log_button.setEnabled(False)

            self.randomizer_thread = Randomizer(log_file_path, total_files, self.output_dir)
            self.randomizer_thread.update_progress.connect(self.progress_bar.setValue)
            self.randomizer_thread.log_message.connect(self.log_output.append)
            self.randomizer_thread.error_occurred.connect(self.display_error)
            self.randomizer_thread.finished.connect(self.on_randomizer_finished)
            self.randomizer_thread.start()
        else:
            QMessageBox.warning(self, "Directories Not Selected", "Please select both source and output directories.")

    def get_total_tracks_available(self, log_file_path):
        """Returns the total number of tracks available in the metadata log."""
        try:
            with open(log_file_path, mode='r') as log_file:
                reader = csv.DictReader(log_file)
                total_tracks = sum(1 for _ in reader)
            self.total_population_size = total_tracks  # Update total population size
            return total_tracks
        except Exception as e:
            logging.error(f"Error reading metadata log: {format_exception(e)}")
            return 0

    @pyqtSlot()
    def calculate_sample_size(self):
        """Calculates the recommended sample size based on statistical parameters."""
        if self.total_population_size == 0:
            # Attempt to get total population size
            log_file_path = os.path.join(self.source_dir, 'metadata_log.csv')
            total_tracks = self.get_total_tracks_available(log_file_path)
            if total_tracks == 0:
                QMessageBox.warning(self, "No Data Available", "Please run metadata logging first to calculate total tracks.")
                return

        N = self.total_population_size  # Total population size

        # Get user inputs
        confidence_level = self.confidence_level_combo.currentText()
        margin_of_error = self.margin_of_error_spinbox.value() / 100  # Convert percentage to proportion
        P = self.variability_spinbox.value()  # Estimated variability

        # Map confidence level to Z-score
        z_scores = {'90%': 1.645, '95%': 1.96, '99%': 2.576}
        Z = z_scores.get(confidence_level, 1.96)  # Default to 95% if not found

        # Calculate initial sample size (n0)
        numerator = (Z ** 2) * P * (1 - P)
        denominator = margin_of_error ** 2
        n0 = numerator / denominator

        # Adjust for finite population
        n = n0 / (1 + ((n0 - 1) / N))

        # Round up to the nearest whole number
        recommended_sample_size = math.ceil(n)

        # Ensure sample size is at least 1 and does not exceed population size
        recommended_sample_size = max(1, min(recommended_sample_size, N))

        # Update the spin box with the calculated sample size
        self.num_total_files_spinbox.setValue(recommended_sample_size)

        # Display the calculation result
        self.log_output.append(f"Calculated Sample Size: {recommended_sample_size} tracks (Confidence Level: {confidence_level}, Margin of Error: {margin_of_error*100}%, Variability: {P})")

    @pyqtSlot()
    def stop_process(self):
        """Stops the running threads."""
        if self.logger_thread and self.logger_thread.isRunning():
            self.logger_thread.stop()
            self.logger_thread.wait()
            self.log_output.append("Metadata logging stopped.")
            self.log_button.setEnabled(True)
            self.run_button.setEnabled(True)

        if self.randomizer_thread and self.randomizer_thread.isRunning():
            self.randomizer_thread.stop()
            self.randomizer_thread.wait()
            self.log_output.append("Randomization stopped.")
            self.run_button.setEnabled(True)
            self.log_button.setEnabled(True)

    @pyqtSlot()
    def on_logging_finished(self):
        """Enables buttons after logging is finished."""
        self.log_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.log_output.append("Metadata logging process completed.")

    @pyqtSlot()
    def on_randomizer_finished(self):
        """Enables buttons after randomization is finished."""
        self.run_button.setEnabled(True)
        self.log_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.log_output.append("Randomization process completed.")

    @pyqtSlot(str)
    def display_error(self, message):
        """Displays error messages to the user."""
        QMessageBox.critical(self, "Error", message)
        self.log_output.append(f"Error: {message}")
        self.log_button.setEnabled(True)
        self.run_button.setEnabled(True)

    def closeEvent(self, event):
        """Ensures threads are stopped when the application is closed."""
        if self.logger_thread and self.logger_thread.isRunning():
            self.logger_thread.stop()
            self.logger_thread.wait()
        if self.randomizer_thread and self.randomizer_thread.isRunning():
            self.randomizer_thread.stop()
            self.randomizer_thread.wait()
        event.accept()

# Main entry point
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)

        # Apply dark theme if qdarkstyle is available
        try:
            app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        except Exception:
            pass  # Use default styling if qdarkstyle is not available

        ex = AVIFileRandomizer()
        ex.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.critical(f"Application failed to start: {format_exception(e)}")
        sys.exit(1)
