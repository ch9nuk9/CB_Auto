import sys
import os
import yaml
import subprocess
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QVBoxLayout,
                             QWidget, QHBoxLayout, QFileDialog, QGroupBox, QMessageBox, QTextEdit, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class ClusterConfigApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cluster Configuration App")
        self.setGeometry(200, 100, 800, 500)  # Increased the default window size
        
        # Initialize paths
        self.source_dir = ""
        self.dest_dir = ""
        self.config_01_path = ""
        self.config_02_path = ""
        self.snakefile_path = ""
        self.runme_cluster_path = ""

        # Configurations content
        self.config_01 = {}
        self.config_02 = {}

        # Main UI setup
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Set dark grey theme with larger font and bigger buttons
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
                font-size: 18px;  /* Increased font size */
            }
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #5a5a5a;
                padding: 10px;  /* Increased padding */
                font-size: 18px;  /* Increased button font size */
                min-height: 40px; /* Increased button height */
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QLineEdit, QTextEdit {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #5a5a5a;
                font-size: 18px;  /* Increased font size */
            }
            QLabel {
                color: white;
                font-size: 18px;  /* Increased font size */
            }
        """)

        # Create Group Box for Main Controls
        main_controls_group = QGroupBox("Main Controls")
        main_layout = QVBoxLayout()

        # Source directory
        source_layout = QHBoxLayout()
        source_label = QLabel("Source Directory:")
        self.source_dir_entry = QLineEdit()
        self.source_dir_entry.setFixedHeight(30)  # Increase entry field height
        source_browse_btn = QPushButton("Browse")
        source_browse_btn.setFixedSize(100, 40)  # Increase button size
        source_browse_btn.clicked.connect(self.browse_source_dir)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_dir_entry)
        source_layout.addWidget(source_browse_btn)
        main_layout.addLayout(source_layout)

        # Destination directory
        dest_layout = QHBoxLayout()
        dest_label = QLabel("Destination Directory:")
        self.dest_dir_entry = QLineEdit()
        self.dest_dir_entry.setFixedHeight(30)  # Increase entry field height
        dest_browse_btn = QPushButton("Browse")
        dest_browse_btn.setFixedSize(100, 40)  # Increase button size
        dest_browse_btn.clicked.connect(self.browse_dest_dir)
        dest_layout.addWidget(dest_label)
        dest_layout.addWidget(self.dest_dir_entry)
        dest_layout.addWidget(dest_browse_btn)
        main_layout.addLayout(dest_layout)

        # Execute button
        execute_btn = QPushButton("Execute")
        execute_btn.setFixedSize(200, 50)  # Increase button size
        execute_btn.clicked.connect(self.execute_pipeline)
        main_layout.addWidget(execute_btn)

        main_controls_group.setLayout(main_layout)
        layout.addWidget(main_controls_group)

        # Create Group Box for Configurations
        config_group = QGroupBox("Configurations")
        config_layout = QVBoxLayout()

        # Edit Cluster Configuration Button
        cluster_edit_btn = QPushButton("Edit Cluster Configuration")
        cluster_edit_btn.setFixedSize(250, 50)  # Increase button size
        cluster_edit_btn.clicked.connect(lambda: self.edit_config("cluster_config.yaml"))
        config_layout.addWidget(cluster_edit_btn)

        # Edit Platform Configuration Button
        platform_edit_btn = QPushButton("Edit Platform Configuration")
        platform_edit_btn.setFixedSize(250, 50)  # Increase button size
        platform_edit_btn.clicked.connect(lambda: self.edit_config("config.yaml"))
        config_layout.addWidget(platform_edit_btn)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Add button to launch thresholding GUI
        threshold_gui_btn = QPushButton("Open Thresholding GUI")
        threshold_gui_btn.setFixedSize(250, 50)  # Increase button size
        threshold_gui_btn.clicked.connect(self.launch_thresholding_gui)
        layout.addWidget(threshold_gui_btn)

        # Set main widget layout
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    # Browse source directory
    def browse_source_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if dir_name:
            self.source_dir = dir_name
            self.source_dir_entry.setText(dir_name)

    # Browse destination directory
    def browse_dest_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Destination Directory")
        if dir_name:
            self.dest_dir = dir_name
            self.dest_dir_entry.setText(dir_name)
            self.copy_files_to_dest()

    # Automatically copy required files to destination
    def copy_files_to_dest(self):
        if not self.source_dir:
            QMessageBox.critical(self, "Error", "Please select a source directory.")
            return
        required_files = ["cluster_config.yaml", "config.yaml", "Snakefile", "RUNME_cluster.sh"]
        for file_name in required_files:
            src_path = os.path.join(self.source_dir, file_name)
            dest_path = os.path.join(self.dest_dir, file_name)
            try:
                shutil.copy(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")
            except FileNotFoundError:
                QMessageBox.critical(self, "Error", f"File not found: {file_name}")
                return
        QMessageBox.information(self, "Success", "Files copied to the destination directory!")

    # Execute the shell script
    def execute_pipeline(self):
        if not self.dest_dir:
            QMessageBox.critical(self, "Error", "Please select a destination directory.")
            return
        shell_script = os.path.join(self.dest_dir, "RUNME_cluster.sh")
        if not os.path.exists(shell_script):
            QMessageBox.critical(self, "Error", "Shell script not found in the destination directory.")
            return
        try:
            subprocess.Popen(['bash', shell_script])
            QMessageBox.information(self, "Success", "Shell script executed successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to execute shell script: {e}")

    # Edit configuration files
    def edit_config(self, config_file):
        config_path = os.path.join(self.dest_dir, config_file)
        if not os.path.exists(config_path):
            QMessageBox.critical(self, "Error", f"{config_file} not found in the destination directory.")
            return
        with open(config_path, 'r') as file:
            config_data = file.read()

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit {config_file}")
        dialog.setGeometry(300, 150, 600, 400)  # Increased pop-up window size
        layout = QVBoxLayout()

        text_edit = QTextEdit()
        text_edit.setPlainText(config_data)
        layout.addWidget(text_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        # Save changes to file
        def save_changes():
            with open(config_path, 'w') as file:
                file.write(text_edit.toPlainText())
            dialog.accept()
            QMessageBox.information(self, "Saved", f"{config_file} saved successfully!")

        button_box.accepted.connect(save_changes)
        button_box.rejected.connect(dialog.reject)

        dialog.exec_()

    # Launch the external thresholding GUI
    def launch_thresholding_gui(self):
        try:
            subprocess.Popen(['python', 'thresholding_gui.py'])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch thresholding GUI: {e}")


def main():
    app = QApplication(sys.argv)

    # Set larger font for the application
    font = QFont()
    font.setPointSize(14)  # Increased the base font size for the entire application
    app.setFont(font)

    window = ClusterConfigApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
