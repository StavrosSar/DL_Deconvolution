import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import threading
import os
import sys
import platform
import time
import shutil
import yaml

class ThesisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DL ")
        self.root.geometry("900x600")
        self.root.configure(bg='#f0f0f0')
        
        # Create a main frame with a scrollbar
        self.main_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a scrollbar
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create a canvas for scrolling
        self.canvas = tk.Canvas(self.main_frame, bg='#f0f0f0', yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure the scrollbar
        self.scrollbar.config(command=self.canvas.yview)
        
        # Create a frame inside the canvas
        self.content_frame = tk.Frame(self.canvas, bg='#f0f0f0')
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor=tk.NW)
        
        # Bind events to update scroll region and canvas size
        self.content_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Get the base directory (where the GUI script is located)
        # Get the base directory (handle both script and executable mode)
        if getattr(sys, 'frozen', False):
            # Running as executable
            self.base_dir = os.path.dirname(sys.executable)
        else:
            # Running as script
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
            
      
        self.projects = {
            "DiffUNet": {
                "script": os.path.join(self.base_dir, "scripts", "DiffUNet.py"),
                "config": os.path.join(self.base_dir, "configs", "DiffUNet.yaml"),
                "data_dir": r"C:\Users\user\Desktop\tr",
                "has_train": True
            },
            "SCUNet": {
                "script": os.path.join(self.base_dir, "scripts", "SCUNet.py"),
                "config": os.path.join(self.base_dir, "configs", "SCUNet.yaml"),
                "data_dir": r"C:\Users\user\Desktop\tr",
                "has_train": True
            },
            "PnPDiffUNet": {
                "script": os.path.join(self.base_dir, "scripts", "PnPDiffUNet.py"), 
                "config": os.path.join(self.base_dir, "configs", "PnPDiffUNet.yaml"),
                "data_dir": r"C:\Users\user\Desktop\tr",
                "has_train": True
            },
            "PnPSCUNet": {
                "script": os.path.join(self.base_dir, "scripts", "PnPSCUNet.py"),
                "config": os.path.join(self.base_dir, "configs", "PnPSCUNet.yaml"),
                "data_dir": r"C:\Users\user\Desktop\tr",
                "has_train": True
            }}
    
            #,
            #"SUNet": {
            #   "script": os.path.join(self.base_dir, "scripts", "SUNet.py"),
            #   "config": os.path.join(self.base_dir, "configs", "SUNet.yaml"),
            #   "data_dir": r"C:\Users\user\Desktop\tr",
            #   "has_train": False,
            #   "has_data_path": False,  # SUNet doesn't need data path configuration
            #   "has_options": False     # SUNet doesn't need additional options
            #}}
        
        # Store the expected output directories for each project
        self.output_dirs = {
            "DiffUNet": os.path.join(self.base_dir, "results", "DiffUNet_images_results"),
            "SCUNet": os.path.join(self.base_dir, "results", "SCUNet_images_results"),
            "PnPDiffUNet": os.path.join(self.base_dir, "results", "PnPDiffUNet_images_results"), 
            "PnPSCUNet": os.path.join(self.base_dir, "results", "PnPSCUNet_images_results")
            # "SUNet": os.path.join(self.base_dir, "results", "SUNet_images_results")
        }
        
        # Store the temporary directories where scripts actually save files
        self.temp_output_dirs = {
            "DiffUNet": os.path.join(os.path.dirname(self.base_dir), "results", "DiffUNet_images_results"),
            "SCUNet": os.path.join(os.path.dirname(self.base_dir), "results", "SCUNet_images_results"),
            "PnPDiffUNet": os.path.join(os.path.dirname(self.base_dir), "results", "PnPDiffUNet_images_results"), 
            "PnPSCUNet": os.path.join(os.path.dirname(self.base_dir), "results", "PnPSCUNet_images_results")
            # "SUNet": os.path.join(os.path.dirname(self.base_dir), "results", "SUNet_images_results")
        }
        
        # Store the current process for stopping
        self.current_process = None
        self.is_training = False
        self.setup_ui()
        
        # Bind the project selection change event
        self.project_var.trace('w', self.on_project_change)
        
        # Bind mouse wheel event for scrolling
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        
    def on_frame_configure(self, event):
        """Update the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def on_canvas_configure(self, event):
        """Update the inner frame's width to fill the canvas"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        
    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
     
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.content_frame, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, text="DL Deconvolution Denoisers", font=('Arial', 24, 'bold'), 
                        fg='white', bg='#2c3e50')
        title.pack(expand=True)
        
        # Main content frame
        main_frame = tk.Frame(self.content_frame, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Project selection
        select_frame = tk.LabelFrame(main_frame, text="Denoisers Selection", font=('Arial', 12, 'bold'),
                                    bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        select_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.project_var = tk.StringVar()
        self.project_var.set(list(self.projects.keys())[0])
        
        projects = list(self.projects.keys())
        for i, project in enumerate(projects):
            rb = tk.Radiobutton(select_frame, text=project, variable=self.project_var, 
                               value=project, font=('Arial', 10), bg='#f0f0f0')
            rb.grid(row=i//2, column=i%2, sticky='w', padx=10, pady=5)
        
        # Action buttons
        self.button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        self.button_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.train_btn = tk.Button(self.button_frame, text="Train Model", command=self.train_model,
                                  font=('Arial', 12, 'bold'), bg='#3498db', fg='black', width=12)
        self.train_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.test_btn = tk.Button(self.button_frame, text="Test Model", command=self.test_model,
                                 font=('Arial', 12, 'bold'), bg='#2ecc71', fg='black', width=12)
        self.test_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = tk.Button(self.button_frame, text="Stop Training", command=self.stop_training,
                                 font=('Arial', 12, 'bold'), bg='#e74c3c', fg='black', width=12, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.results_btn = tk.Button(self.button_frame, text="Open Results", command=self.open_results_folder,
                                   font=('Arial', 12, 'bold'), bg='#9b59b6', fg='black', width=12)
        self.results_btn.pack(side=tk.LEFT)
        
        # Update button visibility based on initial selection
        self.update_button_visibility()
        
        # Data path selection
        self.data_frame = tk.Frame(main_frame, bg='#f0f0f0')
        self.data_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(self.data_frame, text="Data Path:", font=('Arial', 10), bg='#f0f0f0').pack(side=tk.LEFT, padx=(0, 10))
        self.data_var = tk.StringVar(value=r"C:\Users\user\Desktop\tr")
        self.data_entry = tk.Entry(self.data_frame, textvariable=self.data_var, width=50)
        self.data_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.data_browse_btn = tk.Button(self.data_frame, text="Browse", command=self.browse_data_dir,
                                   font=('Arial', 10), bg='#95a5a6', fg='white')
        self.data_browse_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Additional options frame
        self.options_frame = tk.LabelFrame(main_frame, text="Additional Options", font=('Arial', 12, 'bold'),
                                     bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        self.options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Epochs override
        epochs_frame = tk.Frame(self.options_frame, bg='#f0f0f0')
        epochs_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(epochs_frame, text="Epochs:", font=('Arial', 10), bg='#f0f0f0').pack(side=tk.LEFT, padx=(0, 10))
        self.epochs_var = tk.StringVar(value="")
        self.epochs_entry = tk.Entry(epochs_frame, textvariable=self.epochs_var, width=10)
        self.epochs_entry.pack(side=tk.LEFT)
        
        # Batch size override
        batch_frame = tk.Frame(self.options_frame, bg='#f0f0f0')
        batch_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(batch_frame, text="Batch Size:", font=('Arial', 10), bg='#f0f0f0').pack(side=tk.LEFT, padx=(0, 10))
        self.batch_var = tk.StringVar(value="")
        self.batch_entry = tk.Entry(batch_frame, textvariable=self.batch_var, width=10)
        self.batch_entry.pack(side=tk.LEFT)
        
        # Learning rate override
        lr_frame = tk.Frame(self.options_frame, bg='#f0f0f0')
        lr_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(lr_frame, text="Learning Rate:", font=('Arial', 10), bg='#f0f0f0').pack(side=tk.LEFT, padx=(0, 10))
        self.lr_var = tk.StringVar(value="")
        self.lr_entry = tk.Entry(lr_frame, textvariable=self.lr_var, width=10)
        self.lr_entry.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 20))
        
        # Output console
        console_frame = tk.LabelFrame(main_frame, text="Output Console", font=('Arial', 12, 'bold'),
                                     bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        console_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.console = tk.Text(console_frame, bg='#2c3e50', fg='#ecf0f1', font=('Consolas', 10))
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Add a scrollbar to the console
        console_scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        console_scrollbar.pack(side="right", fill="y")
        self.console.configure(yscrollcommand=console_scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.content_frame, textvariable=self.status_var, relief=tk.SUNKEN, 
                             anchor=tk.W, bg='#ecf0f1', fg='#2c3e50')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update UI based on initial selection
        self.update_ui_visibility()
    
    def on_project_change(self, *args):
        """Handle project selection change"""
        self.update_button_visibility()
        self.update_ui_visibility()
    
    def update_button_visibility(self):
        """Update button visibility based on selected project"""
        project = self.project_var.get()
        has_train = self.projects[project].get("has_train", True)
        
        if has_train:
            self.train_btn.pack(side=tk.LEFT, padx=(0, 10))
        else:
            self.train_btn.pack_forget()
    
    def update_ui_visibility(self):
        """Update UI visibility based on selected project"""
        project = self.project_var.get()
        has_data_path = self.projects[project].get("has_data_path", True)
        has_options = self.projects[project].get("has_options", True)
        
        # Show/hide data path section
        if has_data_path:
            self.data_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.data_frame.pack_forget()
        
        # Show/hide options section
        if has_options:
            self.options_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.options_frame.pack_forget()
    
    def browse_data_dir(self):
        """Browse for data directory"""
        data_dir = filedialog.askdirectory(
            title="Select Data Directory",
            initialdir=r"C:\Users\user\Desktop"
        )
        if data_dir:
            self.data_var.set(data_dir)
    
    def open_results_folder(self):
        """Open the results folder in file explorer"""
        project = self.project_var.get()
        output_dir = self.output_dirs[project]
        
        if os.path.exists(output_dir):
            try:
                system = platform.system()
                if system == "Windows":
                    os.startfile(output_dir)
                elif system == "Darwin":  # macOS
                    subprocess.call(["open", output_dir])
                else:  # Linux
                    subprocess.call(["xdg-open", output_dir])
                self.log_output(f"Opened results folder: {output_dir}\n")
            except Exception as e:
                self.log_output(f"Error opening results folder: {e}\n")
        else:
            self.log_output(f"Results folder does not exist yet: {output_dir}\n")
            self.log_output("Please run training or testing first.\n")

    def run_model(self, mode):
        project = self.project_var.get()
        
        '''
        # Special handling for SUNet (no training)
        if project == "SUNet" and mode == "train":
            messagebox.showinfo("Info", "SUNet is a pre-trained model and does not support training.")
            return
        '''
            
        self.is_training = (mode == "train")
        script = self.projects[project]["script"]
        config = self.projects[project]["config"]
        output_dir = self.output_dirs[project]
        temp_output_dir = self.temp_output_dirs[project]
        data_dir = self.data_var.get()
    
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(script):
            messagebox.showerror("Error", f"Script {script} not found!\n\nPlease update the project paths in the code.")
            return
        
        if not os.path.exists(config):
            messagebox.showerror("Error", f"Config file {config} not found!\n\nPlease update the config paths in the code.")
            return
        
        # Only check data directory for models that need it
        if not os.path.exists(data_dir):
            messagebox.showerror("Error", f"Data directory {data_dir} not found!\n\nPlease select a valid data directory.")
            return
        
        self.log_output(f"Running {project} in {mode} mode...\n")
        self.log_output(f"Output will be saved to: {output_dir}\n")
        self.log_output(f"Temporary output location: {temp_output_dir}\n")
        self.log_output(f"Using data from: {data_dir}\n")
            
        self.status_var.set(f"Running {project} in {mode} mode...")
        self.progress.start()
        self.train_btn.config(state=tk.DISABLED)
        self.test_btn.config(state=tk.DISABLED)
        self.results_btn.config(state=tk.DISABLED)
        
        # Enable stop button only for training
        if mode == "train":
            self.stop_btn.config(state=tk.NORMAL)
        else:
            self.stop_btn.config(state=tk.DISABLED)
        
        # Run the model in a separate thread to avoid freezing the UI
        thread = threading.Thread(target=self.run_script, args=(script, config, mode, output_dir, temp_output_dir, data_dir))
        thread.daemon = True
        thread.start()

    def train_model(self):
        self.run_model("train")
    
    def test_model(self):
        self.run_model("test")
    
    def stop_training(self):
        """Stop the currently running training process"""
        if self.current_process and self.is_training:
            try:
                # Terminate the process
                self.current_process.terminate()
                self.log_output("\nTraining process stopped by user.\n")
                self.status_var.set("Training stopped by user")
                
                # Update UI
                self.progress.stop()
                self.train_btn.config(state=tk.NORMAL)
                self.test_btn.config(state=tk.NORMAL)
                self.results_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.is_training = False
                
            except Exception as e:
                self.log_output(f"Error stopping process: {e}\n")
        else:
            self.log_output("No active training process to stop.\n")
    
    
    def run_script(self, script, config, mode, output_dir, temp_output_dir, data_dir):
        try:
            # For SUNet, we don't need to create a temporary config with training paths
            if "SUNet" in script:
             # Use the config as-is for SUNet
             temp_config_path = config
            else:
            # Create a temporary config file with the correct data paths for other models
             temp_config_path = self.create_temp_config(config, data_dir)
            
            # Build the command with absolute paths
            cmd = [
                sys.executable, 
                script, 
                "--config", 
                temp_config_path
            ]
          
            
            # Add mode parameter for models that support it
            #  if "SUNet" not in script:
            #     cmd.extend(["--mode", mode])
            
            # Add optional parameters if specified (only for models that support them)
            if "SUNet" not in script:
                if self.epochs_var.get():
                    cmd.extend(["--epochs", self.epochs_var.get()])
                if self.batch_var.get():
                    cmd.extend(["--batch_size", self.batch_var.get()])
                if self.lr_var.get():
                    cmd.extend(["--learning_rate", self.lr_var.get()])
            
            # Add data directory for SUNet
            #if "SUNet" in script:
            #   cmd.extend(["--data_dir", data_dir])
            #  cmd.extend(["--output_dir", output_dir])
            
            
            
            self.log_output(f"Running: {' '.join(cmd)}\n")
            
            # Change to the script directory to ensure relative paths work correctly
            script_dir = os.path.dirname(script)
            original_dir = os.getcwd()
            
            if script_dir:
                os.chdir(script_dir)
                self.log_output(f"Changed working directory to: {script_dir}\n")
            
            # Run the command with Popen to get process handle
            self.current_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Get the output and wait for completion
            stdout, stderr = self.current_process.communicate()
            return_code = self.current_process.returncode
            
            # Copy results from temporary location to desired location if successful
            if return_code == 0 or not self.is_training:
                self.copy_results(temp_output_dir, output_dir)
            
            # Update UI in the main thread
            self.root.after(0, self.script_finished, stdout, stderr, return_code, mode, temp_config_path)
            
        except Exception as e:
            self.root.after(0, self.script_finished, "", str(e), 1, mode, "")
        finally:
            # Restore original working directory
            os.chdir(original_dir)
                  
        
    def create_temp_config(self, original_config_path, data_dir):
        """Create a temporary config file with updated data paths"""
        # Load the original config
        with open(original_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update ALL possible data path keys
        possible_keys = [
            "x_path", "y_path",                    # PnPDiffUNet
            "x_train_path", "y_train_path",        # DiffUNet, SCUNet, PnPSCUNet
        ]
        
        for key in possible_keys:
            if key in config.get("data", {}):
                config["data"][key] = os.path.join(data_dir, "x_train.npy" if "x" in key else "y_train.npy")
        
        # Create a temporary config file
        temp_config_path = os.path.join(os.path.dirname(original_config_path), "temp_config.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.log_output(f"Created temporary config: {temp_config_path}\n")
        self.log_output(f"Updated data paths to: {data_dir}\n")
        
        return temp_config_path
    
    def copy_results(self, source_dir, dest_dir):
        """Copy results from source directory to destination directory, cleaning destination first"""
        try:
            if os.path.exists(source_dir):
                # Create destination directory if it doesn't exist
                os.makedirs(dest_dir, exist_ok=True)
                
                # Clean destination directory (remove all files)
                for file_name in os.listdir(dest_dir):
                    file_path = os.path.join(dest_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        self.log_output(f"Removed old file: {file_name}\n")
                
                # Copy all files from source to destination
                for file_name in os.listdir(source_dir):
                    source_file = os.path.join(source_dir, file_name)
                    dest_file = os.path.join(dest_dir, file_name)
                    
                    if os.path.isfile(source_file):
                        shutil.copy2(source_file, dest_file)
                        self.log_output(f"Copied: {file_name} to {dest_dir}\n")
                
                self.log_output(f"All results copied to: {dest_dir}\n")
            else:
                self.log_output(f"Source directory not found: {source_dir}\n")
        except Exception as e:
            self.log_output(f"Error copying results: {e}\n")
    
    def script_finished(self, stdout, stderr, return_code, mode, temp_config_path):
        # Clean up temporary config file if it exists
        try:
            if temp_config_path and os.path.exists(temp_config_path) and "temp_config" in temp_config_path:
                os.remove(temp_config_path)
                self.log_output(f"Removed temporary config: {temp_config_path}\n")
        except:
            pass
        
        self.progress.stop()
        self.train_btn.config(state=tk.NORMAL)
        self.test_btn.config(state=tk.NORMAL)
        self.results_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED) 
        self.is_training = False 
        
        if stdout:
            self.log_output(f"Output:\n{stdout}\n")
        if stderr:
            self.log_output(f"Errors:\n{stderr}\n")
        
        if return_code != 0:
            self.status_var.set("Error occurred or process was stopped!")
            if return_code == -9 or return_code == 15:  # Common termination signals
                messagebox.showinfo("Process Stopped", "Training process was stopped by user.")
            else:
                messagebox.showerror("Error", f"Script execution failed with return code: {return_code}")
        else:
            self.status_var.set(f"{mode.capitalize()} completed successfully!")
            
            # After completion, just log where to find results
            project = self.project_var.get()
            output_dir = self.output_dirs[project]
            self.log_output(f"Results saved to: {output_dir}\n")
        self.log_output("Use the 'Open Results' button to view the output files.\n")  
     
    
    def log_output(self, message):
        self.console.insert(tk.END, message)
        self.console.see(tk.END)
    
    def clear_console(self):
        self.console.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    
    #ICON
    try:
       root.iconbitmap("deep-learning.ico")   
    except Exception as e:
       print(f"Could not set icon: {e}")
       
    app = ThesisApp(root)
    root.mainloop()