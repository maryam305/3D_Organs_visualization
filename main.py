import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import importlib
import os
import subprocess  # --- NEW: Import subprocess ---
import sys         # --- NEW: Import sys ---

# --- Violet Theme ---
BG_COLOR = "#1a1124"        # Dark violet background
BUTTON_COLOR = "#3a2458"    # Purple buttons
HOVER_COLOR = "#60348a"     # Lighter violet on hover
TEXT_COLOR = "#ffffff"      # White text
ACCENT_COLOR = "#b26bff"    # Bright violet accent

IMAGE_PATH = "assets"          # Look in the current folder

# --- THIS IS THE MODIFIED FUNCTION ---
def run_system(module_name):
    """Run the selected system module in a new, separate process."""
    try:
        # Construct the file name (e.g., "dental_system.py")
        script_file = f"{module_name}.py"
        
        # Check if the file exists
        if not os.path.exists(script_file):
            messagebox.showerror("Error", f"File not found: {script_file}")
            return
            
        # sys.executable is the path to the current Python interpreter (e.g., "python.exe")
        # Popen is non-blocking; it starts the new process and the launcher continues running.
        subprocess.Popen([sys.executable, script_file])
        print(f"Launcher: Started '{script_file}' in a new process.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch {module_name}: {e}")
# --- END OF MODIFIED FUNCTION ---

def load_image(filename, size=None):
    """Load and optionally resize a colored image safely."""
    path = os.path.join(IMAGE_PATH, filename)
    try:
        img = Image.open(path).convert("RGBA")
        if size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}") 
        return None

def create_main_window():
    root = tk.Tk()
    root.title("Human Body Systems Interface")
    root.geometry("1100x700")
    root.configure(bg=BG_COLOR)

    # --- Decorative Banner ---
    banner_img = load_image("banner.png", size=(1000, 200)) 
    if banner_img:
        banner_label = tk.Label(root, image=banner_img, bg=BG_COLOR)
        banner_label.image = banner_img
        banner_label.pack(pady=(0, 25))
    else:
        tk.Label(
            root, text="🧬 Human Systems Interface",
            font=("Helvetica", 26, "bold"), fg=ACCENT_COLOR, bg=BG_COLOR
        ).pack(pady=20)

    # --- Title ---
    title = tk.Label(
        root, text="Select a Body System", 
        font=("Helvetica", 22, "bold"), 
        fg=ACCENT_COLOR, bg=BG_COLOR
    )
    title.pack(pady=10)

    # --- Systems List (with emojis + colorful icons) ---
    systems = [
        ("🫀 Cardiovascular System", "cardiovascular_system", "heart-removebg-preview.png"),
        ("🧠 Nervous System", "nervous_system", "brain-removebg-preview.png"),
        ("💪 Musculoskeletal System", "musculoskeletal_system", "musclu-removebg-preview.png"),
        ("🦷 Dental System", "dental_system", "dental-removebg-preview.png")
    ]

    def on_enter(e): e.widget.config(bg=HOVER_COLOR)
    def on_leave(e): e.widget.config(bg=BUTTON_COLOR)

    # --- Frame to hold all buttons horizontally ---
    systems_frame = tk.Frame(root, bg=BG_COLOR)
    systems_frame.pack(pady=40)

    for name, module, image_file in systems:
        frame = tk.Frame(systems_frame, bg=BG_COLOR)
        frame.pack(side="left", padx=25)

        # Bigger, colorful image
        img = load_image(image_file, size=(100, 100))
        if img:
            icon = tk.Label(frame, image=img, bg=BG_COLOR)
            icon.image = img
            icon.pack(pady=(0, 15))
        else:
            # Fallback text if image fails to load
            icon = tk.Label(frame, text="Image\nNot Found", font=("Helvetica", 10),
                            fg="red", bg=BG_COLOR, width=12, height=5)
            icon.pack(pady=(0, 15))


        # Bigger button
        btn = tk.Button(
            frame, text=name, width=22, height=3,
            bg=BUTTON_COLOR, fg=TEXT_COLOR,
            font=("Helvetica", 15, "bold"),
            activebackground=ACCENT_COLOR,
            activeforeground="#ffffff",
            relief="flat",
            wraplength=180,
            command=lambda m=module: run_system(m)
        )
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.pack()

    footer = tk.Label(
        root, text="© 2025 Human Systems Project", 
        font=("Helvetica", 10), fg="#999", bg=BG_COLOR
    )
    footer.pack(side="bottom", pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_main_window()
