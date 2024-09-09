# Automatically copy required files to destination
def copy_files_to_dest(self):
    if not self.source_dir:
        QMessageBox.critical(self, "Error", "Please select a source directory.")
        return
    required_files = ["cluster_config.yaml", "config.yaml", "Snakefile", "RUNME_cluster.sh", "login.bat", "loginmobax.bat", "login.sh"]  # Added login.sh
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
