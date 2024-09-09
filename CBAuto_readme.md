
# Cluster Configuration and Thresholding GUI Applications

This project consists of two Python scripts, each designed to manage different parts of a workflow for image processing and configuration management. Both scripts use **PyQt5** to provide user interfaces and support a variety of functionalities, including configuration file editing, image processing, and running external scripts.

---

## 1. **configurator_gui.py**

### Description:
The `configurator_gui.py` script is a graphical interface tool designed to allow users to:
- **Select source and destination directories**.
- **Copy required configuration files** (such as `cluster_config.yaml`, `config.yaml`, `Snakefile`, and `RUNME_cluster.sh`) from the source to the destination directory.
- **Edit configuration files** (`cluster_config.yaml` and `config.yaml`) directly within the application, with the ability to save changes back to the destination directory.
- **Execute the `RUNME_cluster.sh` script**, which is copied from the source to the destination directory, using a simple button click.
- **Launch the Thresholding GUI**, which provides image processing capabilities (more details below).

The script provides a **dark-themed** interface with large buttons and input fields for easy interaction.

### Features:
- **Source and Destination Directories**: The user can browse and select directories, and files are automatically copied to the destination once selected.
- **Configuration Editing**: Allows the user to open, edit, and save changes to `cluster_config.yaml` and `config.yaml` through a popup window.
- **Shell Script Execution**: Runs the `RUNME_cluster.sh` script located in the destination directory.
- **Launch Thresholding GUI**: Provides a button to open the `thresholding_gui.py` script for image processing.
- **Dark Theme**: Styled with a dark grey theme for comfortable usage.

### How It Works:
1. **Directory Selection**: Once the user selects the source and destination directories, the required files are automatically copied to the destination.
2. **Configuration Editing**: Two buttons, "Edit Cluster Configuration" and "Edit Platform Configuration," allow the user to load the configuration files into a text editor within the app.
3. **Shell/batch Script Execution**: The "Execute" button runs the `login.bat ` (Windows) or `login.sh` (Mac) depending on the users OS, prompting the user to ssh into their user account on HCC, then the user needs to cd into the root (now destination) directory and run the `RUNME_cluster.sh` script. This will automatically push the job to the cluster and the analysis will be performed in accordance with the configurations specified by the user. 
4. **Launch Thresholding GUI**: The "Open Thresholding GUI" button opens the `thresholding_gui.py` script.

### Requirements:
- Python 3.x
- PyQt5
- YAML
- subprocess (built-in)
- shutil (built-in)

### How to Run:
```bash
python configurator_gui.py
```

---

## 2. **thresholding_gui.py**

### Description:
The `thresholding_gui.py` script is a graphical interface tool designed for image processing, specifically for thresholding and contour detection in image stacks. It provides real-time controls to adjust various thresholding parameters and visualize the results on-the-fly.

### Features:
- **Image Stack Loading**: Supports loading `.tif`, `.tiff`, `.png`, and other image formats.
- **Real-Time Image Processing**: As the user adjusts sliders (threshold, Gaussian blur, median blur, erosion iterations), the processed image updates in real-time.
- **Frame Navigation**: Allows the user to navigate through different frames of an image stack.
- **Threshold Settings Saving**: Users can save the current threshold settings (threshold, blur, erosion) to a CSV file.
- **Dark-Themed GUI**: Similar to `configurator_gui.py`, this script uses a dark grey theme.

### How It Works:
1. **Load Image Stack**: The user can load a TIFF or other image formats, which are displayed frame by frame.
2. **Adjust Processing Parameters**: The user can adjust threshold, Gaussian blur, median blur, and erosion iterations. These updates are applied in real-time to the displayed image.
3. **Save Settings**: The user can save the current threshold settings as a CSV file for future use.

### Requirements:
- Python 3.x
- PyQt5
- OpenCV (cv2)
- NumPy
- pandas (for saving threshold settings to CSV)
- matplotlib (for image display)

### How to Run:
```bash
python thresholding_gui.py
```

---

## How These Scripts are Linked:

1. **Launching the Thresholding GUI**:
   - The `configurator_gui.py` script provides a button to open and run the `thresholding_gui.py` script. This allows the user to seamlessly transition from managing configurations to processing image stacks.

2. **Workflow**:
   - The user first runs the `configurator_gui.py` to set up the configuration files and directory structure.
   - After editing the necessary configuration files, the user can process image data using the `thresholding_gui.py` by clicking the "Open Thresholding GUI" button in the `configurator_gui.py` interface.

3. **Integration**:
   - The two scripts are independent but linked via the interface in the `configurator_gui.py`, making it easy to manage both configuration settings and image processing from a unified user interface.

---

### Example Workflow:

1. Run `configurator_gui.py` to select source and destination directories.
2. Edit and save configuration files (`cluster_config.yaml` and `config.yaml`).
3. Execute the shell script (`RUNME_cluster.sh`) in the destination directory.
4. Launch `thresholding_gui.py` via the button in `configurator_gui.py` to perform image processing.

---

### Installation and Setup:

1. Install dependencies using `pip`:
   ```bash
   pip install pyqt5 opencv-python-headless numpy pandas matplotlib pyyaml
   ```

2. Run the main scripts:
   - `configurator_gui.py`: Manages configuration and shell script execution.
   - `thresholding_gui.py`: For image processing.

```bash
python configurator_gui.py
```

This setup will provide a graphical interface for both configuration management and image thresholding.

---

### License

This project is licensed under the MIT License.
