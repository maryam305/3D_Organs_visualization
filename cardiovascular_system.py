import sys
import numpy as np
import vtk
from vtk import vtkMath
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox,
                             QColorDialog, QFileDialog, QGroupBox, QGridLayout,
                             QTabWidget, QCheckBox, QSpinBox, QDoubleSpinBox,
                             QTreeWidget, QTreeWidgetItem, QSplitter, QProgressBar,
                             QMessageBox, QDialog, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QUrl
from PyQt5.QtGui import QColor, QPalette
# --- New Imports for Sound ---
from PyQt5.QtMultimedia import QSoundEffect
import os
# --- New Imports for Video Recording ---
from vtk import vtkWindowToImageFilter, vtkOggTheoraWriter
# -----------------------------
from collections import defaultdict
import time
from scipy import interpolate
import matplotlib
matplotlib.use('Qt5Agg') # Use Qt5Agg for embedding in PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- NIfTI/Volume Import ---
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    print("nibabel not found. Install with 'pip install nibabel' for MPR features.")
    HAS_NIBABEL = False

# --- Text-to-Speech (TTS) Import ---
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    print("pyttsx3 not found. Install with 'pip install pyttsx3' for voice features.")
    HAS_TTS = False

# =============================================================================
# --- Text-to-Speech Thread ---
# From musculoskeletal_system.py
# =============================================================================
class SpeechThread(QThread):
    """Runs text-to-speech in a separate thread to avoid freezing the GUI."""
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.text_to_speak = ""

    def speak(self, text):
        """Sets the text to speak and starts the thread."""
        if self.engine:
            self.text_to_speak = text
            self.start()

    def run(self):
        """The QThread's main execution method."""
        if self.text_to_speak and self.engine:
            try:
                self.engine.say(self.text_to_speak)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
            self.text_to_speak = ""

# =============================================================================
# --- Realistic ECG Conduction System ---
# This object simulates the heart's electrical pathways and events.
# =============================================================================
class ECGConductionSystem:
    def __init__(self, fs=30): # fs should match the animation timer
        self.fs = fs
        self.hr = 70
        self.cycle_time = 60.0 / self.hr
        self.current_time_in_cycle = 0.0
        
        self.contraction_strength = 0.15 # Default contraction strength
        self.glow_strength = 0.6 # How much the segments "light up"

        # --- Define "Pathway" Timings (as fraction of the cycle) ---
        self.P_WAVE_START = 0.1  # P-wave starts at 10% of cycle
        self.P_WAVE_DURATION = 0.08
        
        self.AV_DELAY = 0.16     # QRS starts after P-wave + delay
        self.QRS_DURATION = 0.1
        
        self.T_WAVE_START = 0.4
        self.T_WAVE_DURATION = 0.12

    def _gaussian(self, t, center, duration, amplitude):
        """Helper function to create a wave 'bump'."""
        # A simple gaussian bump
        return amplitude * np.exp(-((t - center)**2) / (2 * (duration/4)**2))

    def update(self, bpm):
        """Advances the simulation by one time step."""
        
        # --- 1. Update HR and Timestep ---
        if bpm != self.hr:
            self.hr = bpm
            self.cycle_time = 60.0 / self.hr
        
        time_step = 1.0 / self.fs
        self.current_time_in_cycle += time_step
        
        # --- Check for Sound Triggers (at the *start* of events) ---
        play_atrial_sound = False
        play_ventricular_sound = False
        
        if self.current_time_in_cycle > self.cycle_time:
            self.current_time_in_cycle = 0.0
            play_atrial_sound = True # SA Node fires at start of cycle
        
        t = self.current_time_in_cycle / self.cycle_time # Normalized time (0.0 to 1.0)
        
        # Check for ventricular sound trigger
        if (t >= self.AV_DELAY) and (t - (time_step / self.cycle_time) < self.AV_DELAY):
            play_ventricular_sound = True # AV Node / Bundle of His fires

        # --- 2. Generate Signal Components ("Pathways") ---
        
        # P-Wave (Atrial Contraction)
        p_center = self.P_WAVE_START + self.P_WAVE_DURATION / 2
        p_wave = self._gaussian(t, p_center, self.P_WAVE_DURATION, 0.2)
        
        # QRS Complex (Ventricular Contraction)
        qrs_center = self.AV_DELAY + self.QRS_DURATION / 2
        q_wave = self._gaussian(t, qrs_center - 0.01, 0.02, -0.2)
        r_wave = self._gaussian(t, qrs_center, 0.04, 1.0)
        s_wave = self._gaussian(t, qrs_center + 0.02, 0.03, -0.15)
        qrs_complex = q_wave + r_wave + s_wave
        
        # T-Wave (Repolarization)
        t_center = self.T_WAVE_START + self.T_WAVE_DURATION / 2
        t_wave = self._gaussian(t, t_center, self.T_WAVE_DURATION, 0.15)
        
        # --- 3. Determine Contraction Scales for 3D Segments ---
        atria_scale = self._gaussian(t, p_center, self.P_WAVE_DURATION, 1.0)
        ventricle_scale = self._gaussian(t, qrs_center, self.QRS_DURATION, 1.0)
        
        # --- 4. Sum graph signal and add noise ---
        total_signal = p_wave + qrs_complex + t_wave
        total_signal += np.random.normal(0, 0.02) # Add noise
        
        return {
            'total': total_signal,          # For the graph
            'atria_scale': atria_scale,       # For Atrium 3D model (scale 0-1)
            'ventricle_scale': ventricle_scale, # For Ventricle 3D model (scale 0-1)
            'play_atrial_sound': play_atrial_sound,
            'play_ventricular_sound': play_ventricular_sound
        }


# =============================================================================
# --- Segment Manager ---
# Modified to store original AMBIENT property for glow effect
# =============================================================================
class SegmentManager:
    """Manages all loaded 3D segments (actors, mappers, etc.)."""
    def __init__(self):
        self.segments = {}
        self.segment_groups = defaultdict(list)
        
    def add_segment(self, name, actor, mapper, reader, system, color=(1, 1, 1), opacity=1.0):
        # Calculate original center *before* any transforms
        try:
            polydata = reader.GetOutput()
            original_center = polydata.GetCenter()
        except:
            # Fallback for procedural sources (like demo heart)
            if hasattr(reader, 'GetCenter'):
                original_center = reader.GetCenter()
            else:
                original_center = actor.GetCenter()
        
        # --- NEW: Store original ambient property for glow ---
        original_ambient = actor.GetProperty().GetAmbient()
                
        self.segments[name] = {
            'actor': actor,
            'mapper': mapper,
            'reader': reader, # VTK reader or source
            'opacity': opacity,
            'color': color,
            'visible': True,
            'system': system,
            'original_center': original_center, # For scaling
            'original_ambient': original_ambient # For glowing
        }
        self.segment_groups[system].append(name)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        
    def set_opacity(self, name, opacity):
        if name in self.segments:
            self.segments[name]['opacity'] = opacity
            self.segments[name]['actor'].GetProperty().SetOpacity(opacity)
            
    def set_visibility(self, name, visible):
        if name in self.segments:
            self.segments[name]['visible'] = visible
            self.segments[name]['actor'].SetVisibility(visible)
    
    def set_color(self, name, color):
        if name in self.segments:
            self.segments[name]['color'] = color
            self.segments[name]['actor'].GetProperty().SetColor(*color)
            
    def get_all_actors(self):
        return [seg['actor'] for seg in self.segments.values()]
    
    def get_segments_by_type(self, system_type):
        return [name for name, seg in self.segments.items() if seg['system'] == system_type]
    
    def clear(self):
        self.segments.clear()
        self.segment_groups.clear()

# =============================================================================
# --- Focus Navigator ---
# From musculoskeletal_system.py
# =============================================================================
class FocusNavigator:
    """Handles focus navigation (isolating segments)."""
    def __init__(self, segment_manager, vtk_widget):
        self.segment_manager = segment_manager
        self.vtk_widget = vtk_widget 
        self.original_properties = {}
        self.is_active = False
        
    def activate(self):
        """Called when Focus Mode is turned ON."""
        self.is_active = True
        self.original_properties.clear()
        for name, segment in self.segment_manager.segments.items():
            prop = segment['actor'].GetProperty()
            self.original_properties[name] = {
                'opacity': prop.GetOpacity(),
                'ambient': prop.GetAmbient(),
            }

    def deactivate(self):
        """Called when Focus Mode is turned OFF."""
        self.is_active = False
        for name, props in self.original_properties.items():
            if name in self.segment_manager.segments:
                try:
                    # Restore opacity AND ambient
                    segment_actor = self.segment_manager.segments[name]['actor']
                    segment_actor.GetProperty().SetOpacity(props['opacity'])
                    # --- FIX: Also restore ambient light ---
                    segment_actor.GetProperty().SetAmbient(props['ambient'])
                except:
                    pass
        self.original_properties.clear()
        if hasattr(self, 'vtk_widget') and self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()

    def focus_on_segment(self, target_segment_name):
        """Called when a segment is CLICKED in focus mode."""
        if not self.is_active:
            return
        
        if not self.original_properties:
            self.activate()

        for name, segment in self.segment_manager.segments.items():
            prop = segment['actor'].GetProperty()
            if name == target_segment_name:
                prop.SetOpacity(1.0)
                prop.SetAmbient(0.8) # Make it glow a bit
            else:
                # Make other parts transparent
                prop.SetOpacity(0.1) 
                # Use the original ambient value from before focus mode
                prop.SetAmbient(self.original_properties.get(name, {}).get('ambient', 0.2))
                
        if hasattr(self, 'vtk_widget') and self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()

# =============================================================================
# --- Clipping Dialog ---
# From musculoskeletal_system.py
# =============================================================================
class ClippingDialog(QDialog):
    """A pop-up dialog for controlling advanced VTK clipping planes."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Clipping Planes")
        self.setGeometry(100, 100, 600, 750)
        self.parent_viewer = parent
        
        # Timer to batch updates and prevent lag
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(100) # 100ms delay
        self.update_timer.timeout.connect(self.apply_clipping_now)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Use parent's color scheme
        if self.parent_viewer:
            colors = self.parent_viewer.colors
            self.setStyleSheet(f"""
                QDialog {{
                    background-color: {colors['bg_dark']};
                }}
                QGroupBox {{
                    border: 2px solid {colors['accent_purple']};
                    border-radius: 8px;
                    margin-top: 10px;
                    padding-top: 15px;
                    font-weight: bold;
                    color: {colors['accent_cyan']};
                }}
                QLabel, QCheckBox {{
                    color: {colors['text_light']};
                    font-size: 11px;
                }}
                QPushButton {{
                    background-color: {colors['accent_purple']};
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 6px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {colors['accent_cyan']};
                }}
            """)
        
        # --- Visual Plane Toggles ---
        plane_group = QGroupBox("Show Anatomical Planes")
        plane_layout = QGridLayout()
        
        self.show_axial = QCheckBox("Show Axial (Blue)")
        self.show_sagittal = QCheckBox("Show Sagittal (Red)")
        self.show_coronal = QCheckBox("Show Coronal (Green)")
        
        plane_layout.addWidget(self.show_axial, 0, 0)
        plane_layout.addWidget(self.show_sagittal, 0, 1)
        plane_layout.addWidget(self.show_coronal, 1, 0)
        
        self.show_axial.stateChanged.connect(self.schedule_update)
        self.show_sagittal.stateChanged.connect(self.schedule_update)
        self.show_coronal.stateChanged.connect(self.schedule_update)
        
        plane_group.setLayout(plane_layout)
        layout.addWidget(plane_group)
        
        # --- Plane Position Sliders ---
        pos_group = QGroupBox("Plane Positions (0-100)")
        pos_layout = QVBoxLayout()
        
        # X-Axis (Sagittal)
        pos_layout.addWidget(QLabel("X-Axis (Sagittal):"))
        x_row = QHBoxLayout()
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setRange(0, 100)
        self.x_slider.setValue(50)
        self.x_value = QLabel("50")
        self.x_value.setFixedWidth(30)
        x_row.addWidget(self.x_slider)
        x_row.addWidget(self.x_value)
        pos_layout.addLayout(x_row)
        
        # Y-Axis (Coronal)
        pos_layout.addWidget(QLabel("Y-Axis (Coronal):"))
        y_row = QHBoxLayout()
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setRange(0, 100)
        self.y_slider.setValue(50)
        self.y_value = QLabel("50")
        self.y_value.setFixedWidth(30)
        y_row.addWidget(self.y_slider)
        y_row.addWidget(self.y_value)
        pos_layout.addLayout(y_row)
        
        # Z-Axis (Axial)
        pos_layout.addWidget(QLabel("Z-Axis (Axial):"))
        z_row = QHBoxLayout()
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, 100)
        self.z_slider.setValue(50)
        self.z_value = QLabel("50")
        self.z_value.setFixedWidth(30)
        z_row.addWidget(self.z_slider)
        z_row.addWidget(self.z_value)
        pos_layout.addLayout(z_row)
        
        # Connect sliders to update
        self.x_slider.valueChanged.connect(lambda v: (self.x_value.setText(str(v)), self.schedule_update()))
        self.y_slider.valueChanged.connect(lambda v: (self.y_value.setText(str(v)), self.schedule_update()))
        self.z_slider.valueChanged.connect(lambda v: (self.z_value.setText(str(v)), self.schedule_update()))
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # --- Clipping Region Toggles ---
        clip_group = QGroupBox("Hide Regions (Octant Clipping)")
        clip_layout = QGridLayout()
        
        self.hide_left = QCheckBox("Hide Left (-X)")
        self.hide_right = QCheckBox("Hide Right (+X)")
        self.hide_front = QCheckBox("Hide Front (-Y)") # Standard neuro/radio convention
        self.hide_back = QCheckBox("Hide Back (+Y)")
        self.hide_bottom = QCheckBox("Hide Bottom (-Z)")
        self.hide_top = QCheckBox("Hide Top (+Z)")
        
        clip_layout.addWidget(self.hide_left, 0, 0)
        clip_layout.addWidget(self.hide_right, 0, 1)
        clip_layout.addWidget(self.hide_front, 1, 0)
        clip_layout.addWidget(self.hide_back, 1, 1)
        clip_layout.addWidget(self.hide_bottom, 2, 0)
        clip_layout.addWidget(self.hide_top, 2, 1)
        
        for cb in [self.hide_left, self.hide_right, self.hide_front, 
                   self.hide_back, self.hide_top, self.hide_bottom]:
            cb.stateChanged.connect(self.schedule_update)
        
        clip_group.setLayout(clip_layout)
        layout.addWidget(clip_group)
        
        # --- Control Buttons ---
        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.reset_all)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def schedule_update(self):
        """Schedules a single update, bundling multiple fast changes."""
        self.update_timer.start()
    
    def reset_all(self):
        """Resets all controls to their default state."""
        self.x_slider.setValue(50)
        self.y_slider.setValue(50)
        self.z_slider.setValue(50)
        self.show_axial.setChecked(False)
        self.show_sagittal.setChecked(False)
        self.show_coronal.setChecked(False)
        for cb in [self.hide_left, self.hide_right, self.hide_front, 
                   self.hide_back, self.hide_top, self.hide_bottom]:
            cb.setChecked(False)
        self.schedule_update() # Apply the reset
    
    def apply_clipping_now(self):
        """Tells the main window to apply the new clipping parameters."""
        if self.parent_viewer:
            self.parent_viewer.apply_advanced_clipping(self.get_params())
    
    def get_params(self):
        """Bundles all UI settings into a dictionary."""
        return {
            'x_pos': self.x_slider.value() / 100.0,
            'y_pos': self.y_slider.value() / 100.0,
            'z_pos': self.z_slider.value() / 100.0,
            'show_axial': self.show_axial.isChecked(),
            'show_sagittal': self.show_sagittal.isChecked(),
            'show_coronal': self.show_coronal.isChecked(),
            'hide_left': self.hide_left.isChecked(),
            'hide_right': self.hide_right.isChecked(),
            'hide_front': self.hide_front.isChecked(),
            'hide_back': self.hide_back.isChecked(),
            'hide_top': self.hide_top.isChecked(),
            'hide_bottom': self.hide_bottom.isChecked()
        }

# =============================================================================
# --- Curved MPR Dialog ---
# From musculoskeletal_system.py
# =============================================================================
class CurvedMPRDialog(QDialog):
    """
    A dialog for loading NIfTI volumes and generating a
    Curved Multi-Planar Reconstruction (MPR) / Reslice.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Curved Multi-Planar Reconstruction (MPR)")
        self.setGeometry(100, 100, 900, 800)
        self.parent_viewer = parent
        self.curve_points = []
        self.volume = None       # The full 3D NIfTI volume
        self.current_slice = None # The 2D slice currently shown
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        info = QLabel("1. Load NIfTI volume\n"
                      "2. Set 'CPR Start/End' slice range\n"
                      "3. Use 'Display Slice' slider to find anatomy\n"
                      "4. Click on 2D view to draw curve, then Generate")
        info.setStyleSheet("padding: 10px; background: #2a2a2a; color: #00d4ff;")
        layout.addWidget(info)
        
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Volume")
        self.load_btn.clicked.connect(self.load_volume)
        self.reset_btn = QPushButton("Reset Curve")
        self.reset_btn.clicked.connect(self.reset_curve)
        self.generate_btn = QPushButton("Generate CPR")
        self.generate_btn.clicked.connect(self.generate_cpr)
        
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.generate_btn)
        layout.addLayout(btn_layout)

        # --- Slice Selection Controls ---
        slice_group = QGroupBox("Slice Selection")
        slice_layout = QGridLayout()
        
        slice_layout.addWidget(QLabel("Display Slice:"), 0, 0)
        self.display_slice_slider = QSlider(Qt.Horizontal)
        self.display_slice_slider.setEnabled(False)
        self.display_slice_slider.valueChanged.connect(self.update_display_slice)
        slice_layout.addWidget(self.display_slice_slider, 0, 1)
        
        self.display_slice_label = QLabel("0")
        self.display_slice_label.setFixedWidth(40)
        slice_layout.addWidget(self.display_slice_label, 0, 2)
        
        slice_layout.addWidget(QLabel("CPR Start Slice:"), 1, 0)
        self.start_slice_spin = QSpinBox()
        self.start_slice_spin.setEnabled(False)
        slice_layout.addWidget(self.start_slice_spin, 1, 1)
        
        slice_layout.addWidget(QLabel("CPR End Slice:"), 2, 0)
        self.end_slice_spin = QSpinBox()
        self.end_slice_spin.setEnabled(False)
        slice_layout.addWidget(self.end_slice_spin, 2, 1)
        
        slice_group.setLayout(slice_layout)
        layout.addWidget(slice_group)
        # --- End Slice Controls ---
        
        # --- Matplotlib Canvas for 2D Slice ---
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        
        self.status = QLabel("Ready")
        self.status.setStyleSheet("padding: 5px; color: #06ffa5;")
        layout.addWidget(self.status)
        
        # Connect mouse click event
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        self.display_placeholder()
    
    def display_placeholder(self):
        """Shows a placeholder message when no volume is loaded."""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Load NIfTI volume to begin', ha='center', va='center', fontsize=14, color='gray')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.canvas.draw()
    
    def load_volume(self):
        """Opens a file dialog to load a NIfTI volume."""
        if not HAS_NIBABEL:
            QMessageBox.warning(self, "Missing Library", "Please install 'nibabel' to use this feature.")
            return
        
        path, _ = QFileDialog.getOpenFileName(self, "Load NIfTI", "", "NIfTI (*.nii *.nii.gz)")
        if not path:
            return
        
        try:
            self.status.setText("Loading, please wait...")
            QApplication.processEvents() # Update UI
            
            nii = nib.load(path)
            self.volume = nii.get_fdata(dtype=np.float32)
            
            if len(self.volume.shape) != 3:
                QMessageBox.critical(self, "Error", f"Invalid shape: {self.volume.shape}. Must be 3D.")
                self.volume = None
                return

            z_dim = self.volume.shape[2]
            middle_slice = z_dim // 2
            
            # Configure and enable UI controls
            self.display_slice_slider.setRange(0, z_dim - 1)
            self.display_slice_slider.setValue(middle_slice)
            self.display_slice_label.setText(str(middle_slice))
            
            self.start_slice_spin.setRange(0, z_dim - 1)
            self.start_slice_spin.setValue(0)
            
            self.end_slice_spin.setRange(0, z_dim - 1)
            self.end_slice_spin.setValue(z_dim - 1)
            
            self.display_slice_slider.setEnabled(True)
            self.start_slice_spin.setEnabled(True)
            self.end_slice_spin.setEnabled(True)
            
            self.current_slice = self.volume[:, :, middle_slice]
            self.reset_curve() # Clear any old curve
            self.display_slice()
            self.status.setText(f"Loaded: {self.volume.shape}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed:\n{e}")
            self.display_placeholder()
            self.display_slice_slider.setEnabled(False)
            self.start_slice_spin.setEnabled(False)
            self.end_slice_spin.setEnabled(False)
            self.status.setText("Load failed")

    def update_display_slice(self, value):
        """Updates the 2D slice view when the slider is moved."""
        if self.volume is None:
            return
        
        if 0 <= value < self.volume.shape[2]:
            self.current_slice = self.volume[:, :, value]
            self.display_slice_label.setText(str(value))
            self.display_slice() # Redraw canvas
            self.status.setText(f"Displaying slice {value}. Curve points are preserved.")

    def display_slice(self):
        """Renders the current 2D slice and curve points to the canvas."""
        if self.current_slice is None:
            self.display_placeholder()
            return
        
        self.ax.clear()
        self.ax.imshow(self.current_slice.T, cmap='gray', aspect='equal', origin='lower')
        self.ax.set_title("Click to draw curve")
        
        # Draw curve points if they exist
        if self.curve_points:
            pts = np.array(self.curve_points)
            self.ax.plot(pts[:, 0], pts[:, 1], 'ro-', linewidth=2, markersize=8)
        
        self.canvas.draw()
    
    def on_click(self, event):
        """Called when the Matplotlib canvas is clicked."""
        if event.inaxes != self.ax or self.current_slice is None:
            return
        
        # Add the (x, y) coordinates of the click
        self.curve_points.append([event.xdata, event.ydata])
        self.display_slice() # Redraw to show the new point
        self.status.setText(f"Points: {len(self.curve_points)}")
    
    def reset_curve(self):
        """Clears all curve points."""
        self.curve_points = []
        if self.volume is not None:
             self.display_slice()
        self.status.setText("Curve reset")
    
    def generate_cpr(self):
        """Generates the final curved reslice image."""
        if self.volume is None:
            QMessageBox.warning(self, "Error", "Load volume first")
            return
            
        if len(self.curve_points) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 points")
            return
        
        start_z = self.start_slice_spin.value()
        end_z = self.end_slice_spin.value()

        if start_z >= end_z:
            QMessageBox.warning(self, "Error", "Start slice must be less than end slice.")
            return
        
        try:
            # Create the sub-volume for CPR
            cpr_volume = self.volume[:, :, start_z:end_z+1]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to slice volume:\n{e}")
            return
        
        try:
            points = np.array(self.curve_points)
            
            # Interpolate points to create a smooth, high-resolution path
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            cumulative = np.concatenate([[0], np.cumsum(distances)])
            
            num_samples = int(cumulative[-1] * 2) # Sample 2x the pixel length
            if num_samples < 2: num_samples = 2
                
            sample_distances = np.linspace(0, cumulative[-1], num_samples)
            
            interp_x = np.interp(sample_distances, cumulative, points[:, 0])
            interp_y = np.interp(sample_distances, cumulative, points[:, 1])
            
            straightened = []
            
            # Resample the volume along the interpolated path
            for x, y in zip(interp_x, interp_y):
                xi, yi = int(round(x)), int(round(y))
                
                # Check bounds against the cpr_volume dimensions
                if 0 <= xi < cpr_volume.shape[0] and 0 <= yi < cpr_volume.shape[1]:
                    # Append the Z-stack (depth) at this (x,y) point
                    straightened.append(cpr_volume[xi, yi, :])
                else:
                    # Point is outside bounds, append a blank stack
                    straightened.append(np.zeros(cpr_volume.shape[2]))
            
            # Transpose to get [Distance, Depth]
            straightened = np.array(straightened).T
            
            # --- Display the result in a new window ---
            result_fig = plt.figure(figsize=(12, 8))
            plt.imshow(straightened, cmap='gray', aspect='auto', origin='lower')
            plt.title(f"Straightened Curved MPR (Slices {start_z} to {end_z})", fontsize=16)
            plt.xlabel("Distance along curve")
            plt.ylabel(f"Depth (Slices {start_z}-{end_z})")
            plt.colorbar(label='Intensity')
            plt.tight_layout()
            plt.show() # Shows the new plot
            
            self.status.setText(f"CPR generated for slices {start_z}-{end_z}!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Generation failed:\n{e}")


# =============================================================================
# --- MAIN GUI WINDOW ---
# =============================================================================
class Medical3DVisualizationGUI(QMainWindow):
    def __init__(self):
        global HAS_TTS
        super().__init__()
        self.setWindowTitle("❤️ Advanced Cardiovascular 3D Visualization System")
        self.setGeometry(50, 50, 1600, 1000)
        
        # --- Color Scheme ---
        self.colors = {
            'bg_dark': '#1a1a2e',
            'bg_medium': '#16213e',
            'accent_purple': '#9d4edd',
            'accent_cyan': '#00d4ff',
            'accent_pink': '#ff006e',
            'accent_green': '#06ffa5',
            'accent_yellow': '#ffbe0b',
            'text_light': '#e0e0e0',
            'panel_bg': '#0f3460'
        }
        
        # --- Heart Colors (from cardiovascular_system.py) ---
        self.heart_colors = {
            "Ventricle": (0.88, 0.3, 0.23), # Red
            "Atrium": (0.91, 0.42, 0.23),    # Lighter Red
            "Artery": (0.78, 0.16, 0.16),    # Darker Red
            "Aorta": (0.78, 0.16, 0.16),
            "Vein": (0.08, 0.39, 0.75),      # Blue
            "Cava": (0.08, 0.39, 0.75),
            "Default": (0.8, 0.8, 0.8)      # Grey
        }
        
        self.apply_stylesheet()
        
        # --- Core Components ---
        self.segment_manager = SegmentManager()
        self.vtk_widget = QVTKRenderWindowInteractor() # Pure VTK Interactor
        self.focus_navigator = FocusNavigator(self.segment_manager, self.vtk_widget)
        
        # --- Dialogs ---
        self.clipping_dialog = None
        self.mpr_dialog = None
        
        # --- TTS (from musculoskeletal_system.py) ---
        self.tts_engine = None
        if HAS_TTS:
            try:
                self.tts_engine = pyttsx3.init()
                self.speech_thread = SpeechThread(self.tts_engine)
            except Exception as e:
                print(f"Failed to initialize TTS engine: {e}")
                HAS_TTS = False
        
        # --- Camera Animation (from musculoskeletal_system.py) ---
        self.cam_anim_timer = QTimer()
        self.cam_anim_timer.timeout.connect(self.update_camera_animation)
        self.cam_anim_duration = 1.0 
        self.cam_anim_start_time = 0
        self.start_cam_pos = np.array([0, 0, 0])
        self.start_cam_fp = np.array([0, 0, 0])
        self.target_cam_pos = np.array([0, 0, 0])
        self.target_cam_fp = np.array([0, 0, 0])
        
        # --- Deep Dive Flight System (from musculoskeletal_system.py) ---
        self.flight_timer = QTimer()
        self.flight_timer.timeout.connect(self.update_flight_animation)
        self.flight_interpolator = vtk.vtkCameraInterpolator()
        self.flight_clip_plane = vtk.vtkPlane()
        self.flight_plane_collection = vtk.vtkPlaneCollection()
        self.flight_plane_collection.AddItem(self.flight_clip_plane)
        self.empty_clip_planes = vtk.vtkPlaneCollection()
        self.flight_step = 0
        self.flight_duration = 300 
        self.is_flight_mode = False # For pre-programmed tour
        self.is_diving = False # For click-to-dive tour
        self.is_picking_dive_point = False # NEW state
        self.original_artery_opacity = 0.3 # Default artery opacity
        self.original_vein_opacity = 0.3  # Default vein opacity
        
        # --- Orbit Camera (from musculoskeletal_system.py) ---
        self.orbit_timer = QTimer()
        self.orbit_timer.timeout.connect(self.update_orbit)
        self.is_orbiting = False
        self.orbit_angle = 0
        self.model_center = np.array([0, 0, 0]) 
        
        # --- ECG/Heart Animation (NEW Realistic Conduction System) ---
        self.heart_animator = ECGConductionSystem(fs=30) # fs=30 to match timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.current_ecg = {'total': 0, 'atria_scale': 0, 'ventricle_scale': 0,
                            'play_atrial_sound': False, 'play_ventricular_sound': False}
        self.animation_frame = 0 # This will be managed by the animator now
        self.ecg_history = np.zeros(90) # For the graph (90 frames = 3s at 30fps)
        
        # --- NEW Animation State ---
        self.run_ecg_graph = False
        self.run_heart_animation = False
        
        self.plane_actors = [] # For visual clipping planes
        
 # --- Heartbeat Sound (Loading user files) ---
        assets_dir = "assets"
        self.atrial_beep_file = os.path.join(assets_dir, "beep.wav")
        self.heartbeat_sound_file = os.path.join(assets_dir, "lub-dub.wav")
        
        self.atrial_beep_sound = QSoundEffect()
        if os.path.exists(self.atrial_beep_file):
            self.atrial_beep_sound.setSource(QUrl.fromLocalFile(self.atrial_beep_file))
            self.atrial_beep_sound.setVolume(0.5)
        else:
            print(f"Warning: {self.atrial_beep_file} not found. Atrial sound will be disabled.")

        self.heartbeat_sound = QSoundEffect()
        if os.path.exists(self.heartbeat_sound_file):
            self.heartbeat_sound.setSource(QUrl.fromLocalFile(self.heartbeat_sound_file))
            self.heartbeat_sound.setVolume(0.8)
        else:
            print(f"Warning: {self.heartbeat_sound_file} not found. Ventricular sound will be disabled.")
            
        # --- NEW: Video Recording ---
        self.video_writer = None
        self.window_to_image_filter = None
        
        self.init_ui() # Must be before setup_vtk to create widgets
        self.setup_vtk()
        
    def apply_stylesheet(self):
        """Applies the dark theme stylesheet to the main window."""
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {self.colors['bg_dark']};
                color: {self.colors['text_light']};
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
            }}
            QPushButton {{
                background-color: {self.colors['accent_purple']};
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_cyan']};
            }}
            QPushButton:checked {{
                background-color: {self.colors['accent_green']};
                color: #1a1a2e; /* Dark text on green button */
            }}
            QGroupBox {{
                border: 2px solid {self.colors['accent_purple']};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: {self.colors['accent_cyan']};
            }}
            QSlider::groove:horizontal {{
                background: {self.colors['panel_bg']};
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {self.colors['accent_purple']};
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }}
            QTreeWidget {{
                background-color: {self.colors['bg_medium']};
                color: {self.colors['text_light']};
                border: 2px solid {self.colors['accent_purple']};
                border-radius: 4px;
            }}
            QTabWidget::pane {{
                border-top: 2px solid {self.colors['accent_purple']};
            }}
            QTabBar::tab {{
                background: {self.colors['bg_medium']};
                padding: 10px;
                border-radius: 5px;
            }}
            QTabBar::tab:selected {{
                background: {self.colors['accent_purple']};
                color: white;
            }}
        """)
        
    def init_ui(self):
        """Initializes the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        splitter = QSplitter(Qt.Horizontal)
        
        splitter.addWidget(self.create_left_panel())
        
        self.vtk_widget.setMinimumSize(800, 600)
        splitter.addWidget(self.vtk_widget)
        
        splitter.addWidget(self.create_right_panel())
        
        splitter.setSizes([350, 900, 350])
        main_layout.addWidget(splitter)
        
        self.statusBar().showMessage("Ready - Load models or demo heart")
        
    def setup_vtk(self):
        """Initializes the VTK renderer, interactor, and lights."""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.18) # Dark blue/purple
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground2(0.2, 0.1, 0.3) # Darker purple
        
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # Add lights
        light1 = vtk.vtkLight()
        light1.SetPosition(100, 100, 100)
        light1.SetIntensity(1.0)
        self.renderer.AddLight(light1)
        
        light2 = vtk.vtkLight()
        light2.SetPosition(-100, -100, 100)
        light2.SetIntensity(0.6)
        self.renderer.AddLight(light2)
        
        # Setup picker for scene clicks
        self.picker = vtk.vtkCellPicker()
        self.interactor.SetPicker(self.picker)
        self.interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.on_scene_click, 1.0)
        
        self.interactor.Initialize()
        self.interactor.Start()
        
    def create_left_panel(self):
        """Creates the left panel with loading controls and segment tree."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        title = QLabel("❤️ 3D Heart Viewer")
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {self.colors['accent_cyan']}; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # --- Data Loading Group ---
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout()
        
        load_file = QPushButton("📁 Load File")
        load_file.clicked.connect(self.load_segment_file)
        data_layout.addWidget(load_file)
        
        load_folder = QPushButton("📂 Load Folder")
        load_folder.clicked.connect(self.load_folder)
        data_layout.addWidget(load_folder)
        
        demo_btn = QPushButton("❤️ Load Demo Heart")
        demo_btn.clicked.connect(self.load_demo_heart)
        data_layout.addWidget(demo_btn)
        
        reset_btn = QPushButton("🔄 Reset Scene")
        reset_btn.clicked.connect(self.reset_model)
        data_layout.addWidget(reset_btn)

        # --- Return to Main Button Removed ---
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # --- Segment Tree ---
        layout.addWidget(QLabel("Segments (Click to Focus, Double-Click to Color):"))
        self.segment_tree = QTreeWidget()
        self.segment_tree.setHeaderLabels(["Name"]) # Simplified
        self.segment_tree.itemChanged.connect(self.on_segment_changed)
        self.segment_tree.itemClicked.connect(self.on_segment_tree_clicked)
        self.segment_tree.itemDoubleClicked.connect(self.on_segment_tree_double_clicked)
        
        layout.addWidget(self.segment_tree)
        
        layout.addStretch()
        return panel
        
    def create_right_panel(self):
        """Creates the right panel with all the control tabs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        tabs = QTabWidget()
        tabs.addTab(self.create_animation_tab(), "⚡ Animation")
        tabs.addTab(self.create_rendering_tab(), "🎨 Render")
        tabs.addTab(self.create_camera_tab(), "📷 Camera")
        tabs.addTab(self.create_clipping_tab(), "✂️ Clipping")
        tabs.addTab(self.create_mpr_tab(), "📐 MPR")
        
        layout.addWidget(tabs)
        return panel
    
    # ==================== TAB CREATION ====================
    
    def create_animation_tab(self):
        """Creates the Animation Control tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # --- Playback controls (NEW: Separate buttons) ---
        playback_group = QGroupBox("Animation Control")
        playback_layout = QGridLayout()

        self.play_ecg_btn = QPushButton("⚡ ECG Only")
        self.play_ecg_btn.setCheckable(True)
        self.play_ecg_btn.clicked.connect(self.toggle_ecg_only)
        playback_layout.addWidget(self.play_ecg_btn, 0, 0)

        self.play_heart_btn = QPushButton("❤️ Heartbeat Only")
        self.play_heart_btn.setCheckable(True)
        self.play_heart_btn.clicked.connect(self.toggle_heart_only)
        playback_layout.addWidget(self.play_heart_btn, 0, 1)

        self.play_sync_btn = QPushButton("▶️ Play Sync")
        self.play_sync_btn.setCheckable(True)
        self.play_sync_btn.setStyleSheet(f"background-color: {self.colors['accent_green']}; color: #1a1a2e;")
        self.play_sync_btn.clicked.connect(self.toggle_sync)
        playback_layout.addWidget(self.play_sync_btn, 1, 0)

        self.stop_all_btn = QPushButton("⏹️ Stop All")
        self.stop_all_btn.clicked.connect(self.stop_all_animation)
        playback_layout.addWidget(self.stop_all_btn, 1, 1)
        
        playback_layout.addWidget(QLabel("Heart Rate (BPM):"), 2, 0, 1, 2)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(40, 180)
        self.speed_slider.setValue(70)
        self.speed_slider.valueChanged.connect(self.update_speed)
        playback_layout.addWidget(self.speed_slider, 3, 0, 1, 2)
        
        self.speed_label = QLabel("70 BPM")
        self.speed_label.setAlignment(Qt.AlignCenter)
        playback_layout.addWidget(self.speed_label, 4, 0, 1, 2)
        
        self.frame_label = QLabel("Frame: 0")
        playback_layout.addWidget(self.frame_label, 5, 0, 1, 2)
        
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        # --- Heart settings ---
        heart_group = QGroupBox("Heart Settings")
        heart_layout = QVBoxLayout()
        
        heart_layout.addWidget(QLabel("Contraction Strength:"))
        self.contraction_slider = QSlider(Qt.Horizontal)
        self.contraction_slider.setRange(0, 100) # Percent
        self.contraction_slider.setValue(15)
        self.contraction_slider.valueChanged.connect(self.update_contraction)
        heart_layout.addWidget(self.contraction_slider)
        
        self.contraction_label = QLabel("15%")
        self.contraction_label.setAlignment(Qt.AlignCenter)
        heart_layout.addWidget(self.contraction_label)
        
        # --- NEW: Glow Strength ---
        heart_layout.addWidget(QLabel("Signal Glow Strength:"))
        self.glow_slider = QSlider(Qt.Horizontal)
        self.glow_slider.setRange(0, 100) # Percent
        self.glow_slider.setValue(60) # Default 60%
        self.glow_slider.valueChanged.connect(self.update_glow)
        heart_layout.addWidget(self.glow_slider)
        
        self.glow_label = QLabel("60%")
        self.glow_label.setAlignment(Qt.AlignCenter)
        heart_layout.addWidget(self.glow_label)
        
        heart_group.setLayout(heart_layout)
        layout.addWidget(heart_group)
        
        # --- ECG display ---
        ecg_group = QGroupBox("📊 ECG Monitor")
        ecg_layout = QVBoxLayout()
        
        # Text labels (Simplified)
        self.ecg_value_label = QLabel("ECG: 0.00")
        ecg_layout.addWidget(self.ecg_value_label)
        
        # ECG graph (Matplotlib)
        self.ecg_figure = Figure(figsize=(4, 2))
        self.ecg_canvas = FigureCanvasQTAgg(self.ecg_figure)
        self.ecg_ax = self.ecg_figure.add_subplot(111)
        self.ecg_ax.set_facecolor('black') # Black background
        self.ecg_figure.patch.set_facecolor(self.colors['panel_bg']) 
        
        # Adjust plot window size to match new ECG fs
        self.plot_window_size = 90 # Show 3 seconds (30fps * 3s)
        self.ecg_history = np.zeros(self.plot_window_size)
        
        self.ecg_line, = self.ecg_ax.plot(self.ecg_history, color='lime', linewidth=2)
        self.ecg_ax.set_ylim(-0.5, 1.5)
        self.ecg_ax.set_xlim(0, self.plot_window_size)
        self.ecg_ax.axis('off') # No axes
        ecg_layout.addWidget(self.ecg_canvas)
        
        ecg_group.setLayout(ecg_layout)
        layout.addWidget(ecg_group)
        
        layout.addStretch()
        return tab

    def create_rendering_tab(self):
        """Creates the Rendering tab with opacity sliders."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # --- Opacity Controls ---
        opacity_group = QGroupBox("Opacity Control")
        opacity_layout = QVBoxLayout()
        
        # Ventricles
        opacity_layout.addWidget(QLabel("Ventricles:"))
        self.ventricle_opacity = QSlider(Qt.Horizontal)
        self.ventricle_opacity.setRange(0, 100)
        self.ventricle_opacity.setValue(100)
        self.ventricle_opacity.valueChanged.connect(lambda v: self.set_type_opacity('Ventricle', v/100))
        opacity_layout.addWidget(self.ventricle_opacity)
        
        # Atria
        opacity_layout.addWidget(QLabel("Atria:"))
        self.atria_opacity = QSlider(Qt.Horizontal)
        self.atria_opacity.setRange(0, 100)
        self.atria_opacity.setValue(100)
        self.atria_opacity.valueChanged.connect(lambda v: self.set_type_opacity('Atrium', v/100))
        opacity_layout.addWidget(self.atria_opacity)
        
        # Arteries
        opacity_layout.addWidget(QLabel("Arteries:"))
        self.artery_opacity = QSlider(Qt.Horizontal)
        self.artery_opacity.setRange(0, 100)
        self.artery_opacity.setValue(30) # Default transparent
        self.artery_opacity.valueChanged.connect(lambda v: self.set_type_opacity('Artery', v/100))
        opacity_layout.addWidget(self.artery_opacity)
        
        # Veins
        opacity_layout.addWidget(QLabel("Veins:"))
        self.vein_opacity = QSlider(Qt.Horizontal)
        self.vein_opacity.setRange(0, 100)
        self.vein_opacity.setValue(30) # Default transparent
        self.vein_opacity.valueChanged.connect(lambda v: self.set_type_opacity('Vein', v/100))
        opacity_layout.addWidget(self.vein_opacity)
        
        opacity_group.setLayout(opacity_layout)
        layout.addWidget(opacity_group)

        # --- Master Opacity ---
        master_group = QGroupBox("Master Opacity")
        master_layout = QVBoxLayout()
        master_row = QHBoxLayout()
        self.master_opacity = QSlider(Qt.Horizontal)
        self.master_opacity.setRange(0, 100)
        self.master_opacity.setValue(100)
        self.master_opacity.valueChanged.connect(self.update_master_opacity)
        self.master_label = QLabel("100%")
        self.master_label.setFixedWidth(40)
        master_row.addWidget(self.master_opacity)
        master_row.addWidget(self.master_label)
        master_layout.addLayout(master_row)
        master_group.setLayout(master_layout)
        layout.addWidget(master_group)
        # -------------------------

        # --- Color Controls ---
        color_group = QGroupBox("Colors")
        color_layout = QVBoxLayout()
        apply_colors_btn = QPushButton("Apply Realistic Colors")
        apply_colors_btn.clicked.connect(self.apply_realistic_colors)
        color_layout.addWidget(apply_colors_btn)
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        layout.addStretch()
        return tab

    def create_camera_tab(self):
        """Creates the Camera tab (from musculoskeletal_system.py)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # --- Deep Dive Tour ---
        tour_group = QGroupBox("Deep Dive 'Blood Flow' Tour")
        tour_layout = QVBoxLayout()
        
        self.flight_btn = QPushButton("🚀 Start Blood Flow Tour")
        self.flight_btn.setCheckable(True) # Make it checkable
        self.flight_btn.clicked.connect(self.toggle_guided_tour)
        tour_layout.addWidget(self.flight_btn)

        # --- NEW: Record Button ---
        self.record_tour_btn = QPushButton("⏺️ Record Tour")
        self.record_tour_btn.setStyleSheet(f"background-color: {self.colors['accent_pink']};")
        self.record_tour_btn.clicked.connect(self.record_guided_tour)
        tour_layout.addWidget(self.record_tour_btn)

        # --- NEW: Click-to-Dive Button ---
        self.dive_btn = QPushButton("🚁 Deep Dive at Click")
        self.dive_btn.setCheckable(True)
        self.dive_btn.clicked.connect(self.toggle_dive_mode)
        tour_layout.addWidget(self.dive_btn)
        
        tour_layout.addWidget(QLabel("Tour Duration (seconds):"))
        self.flight_speed_slider = QSlider(Qt.Horizontal)
        self.flight_speed_slider.setRange(5, 30) # 5 to 30 seconds
        self.flight_speed_slider.setValue(15)
        tour_layout.addWidget(self.flight_speed_slider)
        
        tour_group.setLayout(tour_layout)
        layout.addWidget(tour_group)
        
        # --- Orbit Camera ---
        orbit_group = QGroupBox("Orbit Camera")
        orbit_layout = QVBoxLayout()
        
        self.orbit_btn = QPushButton("🔄 Start Orbit")
        self.orbit_btn.setCheckable(True) # Make it checkable
        self.orbit_btn.clicked.connect(self.toggle_orbit)
        orbit_layout.addWidget(self.orbit_btn)

        orbit_layout.addWidget(QLabel("Orbit Speed:"))
        self.orbit_speed_slider = QSlider(Qt.Horizontal)
        self.orbit_speed_slider.setRange(1, 20)
        self.orbit_speed_slider.setValue(5)
        orbit_layout.addWidget(self.orbit_speed_slider)
        
        orbit_group.setLayout(orbit_layout)
        layout.addWidget(orbit_group)

        # --- Focus Navigation ---
        focus_group = QGroupBox("Focus Navigation")
        focus_layout = QVBoxLayout()
        
        focus_btn = QPushButton("🎯 Fly To Selected")
        focus_btn.clicked.connect(self.on_fly_to_button_pressed)
        focus_layout.addWidget(focus_btn)
        
        self.focus_nav_btn = QPushButton("Enable Focus Mode")
        self.focus_nav_btn.setCheckable(True)
        self.focus_nav_btn.toggled.connect(self.toggle_focus_nav_mode)
        focus_layout.addWidget(self.focus_nav_btn)
        
        reset_cam = QPushButton("🔄 Reset Camera")
        reset_cam.clicked.connect(self.reset_camera)
        focus_layout.addWidget(reset_cam)
        
        focus_group.setLayout(focus_layout)
        layout.addWidget(focus_group)
        
        layout.addStretch()
        return tab
    
    def create_clipping_tab(self):
        """Creates the Clipping tab (from musculoskeletal_system.py)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info = QLabel("Advanced clipping with visible anatomical planes")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {self.colors['accent_cyan']}; padding: 10px;")
        layout.addWidget(info)
        
        open_btn = QPushButton("🔓 Open Advanced Clipping")
        open_btn.setStyleSheet(f"background-color: {self.colors['accent_green']}; color: #1a1a2e; font-size: 14px; padding: 12px;")
        open_btn.clicked.connect(self.open_clipping_dialog)
        layout.addWidget(open_btn)
        
        layout.addStretch()
        return tab
    
    def create_mpr_tab(self):
        """Creates the MPR tab (from musculoskeletal_system.py)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info = QLabel("Curved Multi-Planar Reconstruction:\nDraw curved paths through NIfTI volumes")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {self.colors['accent_cyan']}; padding: 10px;")
        layout.addWidget(info)
        
        open_btn = QPushButton("📐 Open MPR Tool")
        open_btn.setStyleSheet(f"background-color: {self.colors['accent_green']}; color: #1a1a2e; font-size: 14px; padding: 12px;")
        open_btn.clicked.connect(self.open_mpr_dialog)
        layout.addWidget(open_btn)
        
        layout.addStretch()
        return tab

    # ==================== DATA LOADING ====================
    
    def load_segment_file(self):
        """Loads a single 3D file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load File", "", "3D Files (*.stl *.obj *.ply *.vtk)")
        if path:
            name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()
            system_type = self.detect_type(name)
            self.load_segment(path, name, system_type)
            self.update_model_center()
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
    
    def load_folder(self):
        """Loads all 3D files from a selected folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        
        extensions = ['.stl', '.obj', '.ply', '.vtk']
        files = []
        
        # Find all files with matching extensions
        for ext in extensions:
            files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(ext)])
        
        if not files:
            QMessageBox.warning(self, "No Files", "No 3D files found in folder")
            return
        
        progress = QProgressBar()
        progress.setMaximum(len(files))
        self.statusBar().addWidget(progress)
        
        for i, path in enumerate(files):
            name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()
            system_type = self.detect_type(name)
            self.load_segment(path, name, system_type)
            progress.setValue(i + 1)
            QApplication.processEvents() # Update progress bar
        
        self.statusBar().removeWidget(progress)
        self.update_model_center()
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"Loaded {len(files)} files")
    
    def detect_type(self, name):
        """Detects cardiovascular part type based on name."""
        name_lower = name.lower()
        if 'ventricle' in name_lower: return 'Ventricle'
        if 'atrium' in name_lower: return 'Atrium'
        if 'aorta' in name_lower: return 'Artery'
        if 'artery' in name_lower: return 'Artery'
        if 'cava' in name_lower: return 'Vein'
        if 'vein' in name_lower: return 'Vein'
        return 'Other'
    
    def get_color_for_name(self, name):
        """Gets the appropriate color from the heart_colors dict."""
        name_lower = name.lower()
        for key, color in self.heart_colors.items():
            if key.lower() in name_lower:
                return color
        return self.heart_colors['Default']

    def load_segment(self, path, name, system_type='Other'):
        """Universal loader for a single file segment."""
        ext = os.path.splitext(path)[1].lower()
        
        reader = None
        if ext == '.stl':
            reader = vtk.vtkSTLReader()
        elif ext == '.obj':
            reader = vtk.vtkOBJReader()
        elif ext == '.ply':
            reader = vtk.vtkPLYReader()
        elif ext == '.vtk':
            reader = vtk.vtkPolyDataReader()
        else:
            print(f"Unsupported file type: {ext}")
            return
        
        reader.SetFileName(path)
        reader.Update() # Load the data
        
        polydata = reader.GetOutput()
        if not polydata or polydata.GetNumberOfPoints() == 0:
            print(f"Failed to read file or file is empty: {path}")
            return
            
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(30)
        
        color = self.get_color_for_name(name)
        
        # Set opacity based on type
        if system_type in ['Artery', 'Vein']:
            opacity = 0.3
        else:
            opacity = 1.0
        
        self.segment_manager.add_segment(name, actor, mapper, reader, system_type, color, opacity)
        self.renderer.AddActor(actor)
        
        # Add to tree
        if not self.segment_tree.findItems(name, Qt.MatchExactly):
            item = QTreeWidgetItem([name])
            item.setCheckState(0, Qt.Checked)
            self.segment_tree.addTopLevelItem(item)
    
    def load_demo_heart(self):
        """Loads a procedural demo heart (from cardiovascular_system.py)"""
        if not self.renderer:
            return
        
        self.reset_model()
        
        # Ventricles
        cfg = {"name": "Ventricle", "pos": (0, 0, 0), "r": 40}
        source = vtk.vtkSphereSource()
        source.SetRadius(cfg["r"])
        source.SetCenter(*cfg["pos"])
        source.SetPhiResolution(30); source.SetThetaResolution(30)
        source.Update() # Generate the polydata
        color = self.heart_colors['Ventricle']
        self.add_vtk_source(source, cfg["name"], "Ventricle", color, 1.0)

        # Atria
        atria = [
            {"name": "Left Atrium", "pos": (-15, 35, 0), "r": 18},
            {"name": "Right Atrium", "pos": (15, 35, 0), "r": 18}
        ]
        color = self.heart_colors['Atrium']
        for cfg in atria:
            source = vtk.vtkSphereSource()
            source.SetRadius(cfg["r"])
            source.SetCenter(*cfg["pos"])
            source.SetPhiResolution(20); source.SetThetaResolution(20)
            source.Update()
            self.add_vtk_source(source, cfg["name"], "Atrium", color, 1.0)

        # Vessels
        vessels = [
            {"name": "Aorta", "pos": (0, 40, 0), "dir": (0, 1, 0), "r": 6, "h": 50, "type": "Artery"},
            {"name": "Superior Vena Cava", "pos": (20, 40, 0), "dir": (0, 1, 0), "r": 5, "h": 40, "type": "Vein"},
            {"name": "Pulmonary Artery", "pos": (-10, 40, 0), "dir": (0, 1, 0), "r": 5, "h": 40, "type": "Artery"},
        ]
        
        for cfg in vessels:
            source = vtk.vtkCylinderSource()
            source.SetRadius(cfg["r"])
            source.SetHeight(cfg["h"])
            # VTK cylinder center is halfway, so adjust
            center = (
                cfg["pos"][0] + cfg["dir"][0] * cfg["h"] / 2,
                cfg["pos"][1] + cfg["dir"][1] * cfg["h"] / 2,
                cfg["pos"][2] + cfg["dir"][2] * cfg["h"] / 2
            )
            source.SetCenter(center)
            # Note: This demo doesn't rotate the cylinders
            source.Update()
            
            color = self.heart_colors[cfg['type']]
            opacity = 0.3
            self.add_vtk_source(source, cfg["name"], cfg["type"], color, opacity)

        self.update_model_center()
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Demo heart loaded! Press Play to animate")

    def add_vtk_source(self, source, name, system_type, color, opacity=1.0):
        """Helper to add any VTK source (Sphere, Cylinder) to the scene."""
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(source.GetOutput()) # Use GetOutput() for sources
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(30)
        
        self.segment_manager.add_segment(name, actor, mapper, source, system_type, color, opacity)
        self.renderer.AddActor(actor)
        
        # Add to tree
        if not self.segment_tree.findItems(name, Qt.MatchExactly):
            item = QTreeWidgetItem([name])
            item.setCheckState(0, Qt.Checked)
            self.segment_tree.addTopLevelItem(item)

    def reset_model(self):
        """Resets the entire scene and UI."""
        self.stop_all_camera_motion()
        
        # Stop animation
        self.stop_all_animation() # Use new stop function
        
        if self.focus_navigator.is_active:
            self.focus_nav_btn.setChecked(False) # This will call deactivate()
        
        # Remove clipping planes
        if self.clipping_dialog:
             self.clipping_dialog.reset_all() # Resets dialog UI
             self.apply_advanced_clipping(self.clipping_dialog.get_params()) # Applies empty params
        
        # Stop sounds
        self.heartbeat_sound.stop()
        self.atrial_beep_sound.stop()
            
        for actor in self.segment_manager.get_all_actors():
            self.renderer.RemoveActor(actor)
        
        for actor in self.plane_actors:
            self.renderer.RemoveActor(actor)
        self.plane_actors.clear()
        
        self.segment_manager.clear()
        self.segment_tree.clear()
        
        self.update_model_center()
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Reset complete")

    # --- Return to Main Function Removed ---

    # ==================== EVENT HANDLERS ====================
    
    def on_segment_changed(self, item, column):
        """Handles the checkbox for visibility."""
        if column == 0:
            name = item.text(0)
            visible = item.checkState(0) == Qt.Checked
            self.segment_manager.set_visibility(name, visible)
            self.vtk_widget.GetRenderWindow().Render()
    
    def on_segment_tree_double_clicked(self, item, column):
        """Handles changing color on double-click."""
        if column == 0:
            name = item.text(0)
            if name in self.segment_manager.segments:
                segment = self.segment_manager.segments[name]
                initial_color = QColor.fromRgbF(*segment['color'])
                
                color = QColorDialog.getColor(initial_color, self, "Select Color")
                
                if color.isValid():
                    rgb_float = (color.redF(), color.greenF(), color.blueF())
                    self.segment_manager.set_color(name, rgb_float)
                    self.vtk_widget.GetRenderWindow().Render()

    def on_segment_tree_clicked(self, item, column):
        """Handles segment focusing when tree is clicked."""
        name = item.text(0)
        self.select_and_focus_segment(name)

    def on_scene_click(self, obj, event):
        """Handles clicking on an actor in the 3D scene."""
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        
        clicked_actor = self.picker.GetActor()
        
        # --- NEW: Click-to-Dive Logic ---
        if self.is_picking_dive_point and clicked_actor:
            target_point = self.picker.GetPickPosition()
            target_normal = self.picker.GetPickNormal()
            self.is_picking_dive_point = False
            self.dive_btn.setChecked(False)
            self.statusBar().showMessage(f"Starting deep dive at {target_point}...")
            self.start_deep_dive(target_point, target_normal)
            return # Don't process this click for anything else
            
        if clicked_actor:
            # Find the name of the clicked actor
            target_name = None
            for name, seg_data in self.segment_manager.segments.items():
                if seg_data['actor'] == clicked_actor:
                    target_name = name
                    break
            
            if target_name:
                self.select_and_focus_segment(target_name)
        else:
            # Pass event to interactor *style* to enable rotation
            self.interactor.GetInteractorStyle().OnLeftButtonDown()

    def select_and_focus_segment(self, name):
        """Unified function to handle all focus requests."""
        if name not in self.segment_manager.segments:
            return

        # 1. Update Tree Selection
        for i in range(self.segment_tree.topLevelItemCount()):
            item = self.segment_tree.topLevelItem(i)
            if item.text(0) == name:
                self.segment_tree.setCurrentItem(item)
                break
        
        actor = self.segment_manager.segments[name]['actor']
        
        # 2. Speak Name (if enabled)
        if HAS_TTS:
            self.speech_thread.speak(name)
            
        # 3. Animate Camera
        self.animate_camera_to_actor(actor)
        
        # 4. Isolate if mode is on
        if self.focus_navigator.is_active:
            self.focus_navigator.focus_on_segment(name)
    
    # ==================== ANIMATION ====================

    # --- NEW: Separate Animation Toggles ---
    def _start_animation_timer(self):
        """Helper to start the main timer if not already running."""
        if not self.animation_timer.isActive():
            has_heart = any(seg['system'] in ['Ventricle', 'Atrium'] 
                          for seg in self.segment_manager.segments.values())
            if not has_heart:
                 QMessageBox.warning(self, "No Heart", "Load a heart model first (e.g., 'Load Demo Heart')")
                 self.play_ecg_btn.setChecked(False)
                 self.play_heart_btn.setChecked(False)
                 self.play_sync_btn.setChecked(False)
                 return
            
            self.reset_animation_meshes()
            self.animation_timer.start(int(1000 / self.heart_animator.fs))
            self.statusBar().showMessage("Animation playing")

    def toggle_ecg_only(self, checked):
        """Runs only the ECG graph, glow, and beep sound."""
        if checked:
            self.run_ecg_graph = True
            self.run_heart_animation = False
            self.play_heart_btn.setChecked(False)
            self.play_sync_btn.setChecked(False)
            self._start_animation_timer()
        else:
            # If no other button is checked, stop everything
            if not self.play_heart_btn.isChecked() and not self.play_sync_btn.isChecked():
                self.stop_all_animation()
            # If another button is checked, this one just turns off
            else:
                self.run_ecg_graph = False


    def toggle_heart_only(self, checked):
        """Runs only the 3D contraction and lub-dub sound."""
        if checked:
            self.run_ecg_graph = False
            self.run_heart_animation = True
            self.play_ecg_btn.setChecked(False)
            self.play_sync_btn.setChecked(False)
            self._start_animation_timer()
        else:
            # If no other button is checked, stop everything
            if not self.play_ecg_btn.isChecked() and not self.play_sync_btn.isChecked():
                self.stop_all_animation()
            # If another button is checked, this one just turns off
            else:
                self.run_heart_animation = False
    
    def toggle_sync(self, checked):
        """Runs all animations and sounds."""
        if checked:
            self.run_ecg_graph = True
            self.run_heart_animation = True
            self.play_ecg_btn.setChecked(False)
            self.play_heart_btn.setChecked(False)
            self._start_animation_timer()
        else:
            # If no other button is checked, stop everything
            if not self.play_ecg_btn.isChecked() and not self.play_sync_btn.isChecked():
                self.stop_all_animation()

    def stop_all_animation(self):
        """Stops all animations and resets buttons."""
        self.animation_timer.stop()
        
        self.run_ecg_graph = False
        self.run_heart_animation = False
        
        # Uncheck all buttons
        self.play_ecg_btn.setChecked(False)
        self.play_heart_btn.setChecked(False)
        self.play_sync_btn.setChecked(False)
        
        self.heartbeat_sound.stop()
        self.atrial_beep_sound.stop()
        
        self.reset_animation() # Resets meshes and graph
        self.statusBar().showMessage("Animation stopped")

    def reset_animation(self):
        """Resets animation state and graph."""
        self.animation_frame = 0
        self.frame_label.setText("Frame: 0")
        self.heart_animator.current_time_in_cycle = 0.0 # Reset ECG
        
        self.reset_animation_meshes()
        
        # Redraw graph at start
        self.ecg_history = np.zeros(self.plot_window_size)
        self.ecg_line.set_ydata(self.ecg_history)
        self.ecg_canvas.draw()
        
        if not self.animation_timer.isActive():
             self.vtk_widget.GetRenderWindow().Render()

    def reset_animation_meshes(self):
        """Restores all meshes to their original, unscaled state."""
        for name, segment in self.segment_manager.segments.items():
            try:
                # SetUserTransform(None) resets any transformation
                segment['actor'].SetUserTransform(None)
                # --- Reset ambient light ---
                prop = segment['actor'].GetProperty()
                prop.SetAmbient(segment['original_ambient'])
            except Exception as e:
                print(f"Error resetting mesh {name}: {e}")
            
    def update_speed(self, value):
        """Update BPM from slider."""
        self.speed_label.setText(f"{value} BPM")
        # The new BPM will be picked up by the animator in the next frame
        
    def update_contraction(self, value):
        """Update contraction strength from slider."""
        # Scale strength, e.g., 100% on slider = 0.3 (30%) max scale
        strength = value / 100.0 * 0.3 
        self.heart_animator.contraction_strength = strength
        self.contraction_label.setText(f"{value}%")
        
    def update_glow(self, value):
        """Update glow strength from slider."""
        # Scale strength, e.g., 100% on slider = 1.0 (100%) max glow
        strength = value / 100.0
        self.heart_animator.glow_strength = strength
        self.glow_label.setText(f"{value}%")
        
    def update_animation(self):
        """
        Main animation loop - THIS IS THE FAST, CPU-FRIENDLY VERSION.
        It uses VTK transforms (scaling) and property changes (glowing).
        """
        # --- 0. Check if animation is running ---
        if not self.run_ecg_graph and not self.run_heart_animation:
            return
            
        bpm = self.speed_slider.value()
        # --- 1. Get new state from Conduction System ---
        ecg_state = self.heart_animator.update(bpm)
        self.current_ecg = ecg_state
        
        # --- 2. Update UI Labels ---
        self.animation_frame += 1 # Simple frame counter
        self.frame_label.setText(f"Frame: {self.animation_frame}")
        self.ecg_value_label.setText(f"ECG: {ecg_state['total']:.3f}")
        
        # --- 3. Update ECG graph (if enabled) ---
        if self.run_ecg_graph:
            self.ecg_history = np.roll(self.ecg_history, -1)
            self.ecg_history[-1] = ecg_state['total']
            self.ecg_line.set_ydata(self.ecg_history)
            self.ecg_canvas.draw()
        
        # --- 4. Play Sounds based on events (if enabled) ---
        if self.run_ecg_graph and ecg_state['play_atrial_sound']:
            self.atrial_beep_sound.play()
            
        if self.run_heart_animation and ecg_state['play_ventricular_sound']:
            self.heartbeat_sound.play()
            
        # --- 5. Animate 3D Segments (Scale and Glow) ---
        contraction_strength = self.heart_animator.contraction_strength
        glow_strength = self.heart_animator.glow_strength
        
        atrial_contraction_scale = ecg_state['atria_scale']
        ventricular_contraction_scale = ecg_state['ventricle_scale']

        for name, segment in self.segment_manager.segments.items():
            if not segment['visible']:
                continue

            system_type = segment['system']
            actor = segment['actor']
            prop = actor.GetProperty()
            center = segment['original_center']
            original_ambient = segment['original_ambient']
            
            # --- NEW LOGIC: Separate Glow (ECG) from Scale (Heartbeat) ---
            
            # --- Glow (Electrical) ---
            if self.run_ecg_graph:
                glow = 0.0
                if system_type == 'Atrium':
                    glow = atrial_contraction_scale * glow_strength
                elif system_type == 'Ventricle':
                    glow = ventricular_contraction_scale * glow_strength
                prop.SetAmbient(original_ambient + glow)
            else:
                prop.SetAmbient(original_ambient) # Reset glow if ECG is off

            # --- Scale (Mechanical) ---
            if self.run_heart_animation:
                scale = 1.0
                if system_type == 'Atrium':
                    scale = 1.0 - (atrial_contraction_scale * contraction_strength)
                elif system_type == 'Ventricle':
                    scale = 1.0 - (ventricular_contraction_scale * contraction_strength)
                
                if scale != 1.0:
                    # Create a transform that scales from the object's center
                    transform = vtk.vtkTransform()
                    transform.Translate(center[0], center[1], center[2])
                    transform.Scale(scale, scale, scale)
                    transform.Translate(-center[0], -center[1], -center[2])
                    actor.SetUserTransform(transform)
                else:
                    actor.SetUserTransform(None) # Reset transform
            else:
                actor.SetUserTransform(None) # Reset transform if heartbeat is off

        # --- 6. Render the 3D scene ---
        self.vtk_widget.GetRenderWindow().Render()
    
    # ==================== RENDERING ====================
    
    # --- Master Opacity Function ---
    def update_master_opacity(self, value):
        opacity = value / 100.0
        self.master_label.setText(f"{value}%")
        
        # Update all segments
        for name in self.segment_manager.segments.keys():
            self.segment_manager.set_opacity(name, opacity)
            
        # Resync individual sliders
        self.ventricle_opacity.setValue(int(opacity*100))
        self.atria_opacity.setValue(int(opacity*100))
        self.artery_opacity.setValue(int(opacity*100))
        self.vein_opacity.setValue(int(opacity*100))
        
        self.vtk_widget.GetRenderWindow().Render()

    def set_type_opacity(self, type_name, opacity):
        """Sets opacity for all segments of a given type."""
        # Handle special cases for arteries/veins
        search_keys = [type_name.lower()]
        if type_name == 'Artery':
            search_keys.append('aorta')
        if type_name == 'Vein':
            search_keys.append('cava')

        for name, seg in self.segment_manager.segments.items():
            name_lower = name.lower()
            for key in search_keys:
                if key in name_lower:
                    self.segment_manager.set_opacity(name, opacity)
                    break
        
        # Update sliders to stay in sync
        percent = int(opacity * 100)
        if type_name == 'Ventricle':
            self.ventricle_opacity.setValue(percent)
        elif type_name == 'Atrium':
            self.atria_opacity.setValue(percent)
        elif type_name == 'Artery':
            self.artery_opacity.setValue(percent)
        elif type_name == 'Vein':
            self.vein_opacity.setValue(percent)
            
        if self.focus_navigator.is_active:
            self.focus_navigator.activate() # Re-apply focus opacities
            current_item = self.segment_tree.currentItem()
            if current_item:
                self.focus_navigator.focus_on_segment(current_item.text(0))

        if not self.animation_timer.isActive():
            self.vtk_widget.GetRenderWindow().Render()
            
    def apply_realistic_colors(self):
        """Apply realistic colors to all loaded segments."""
        for name, segment in self.segment_manager.segments.items():
            color = self.get_color_for_name(name)
            self.segment_manager.set_color(name, color)
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Applied realistic colors")
    
    # ==================== CAMERA (from musculoskeletal_system.py) ====================
    
    def stop_all_camera_motion(self):
        """Helper to stop all conflicting camera animations."""
        self.cam_anim_timer.stop()
        
        if self.is_flight_mode:
            self.is_flight_mode = False
            self.flight_timer.stop()
            self.flight_btn.setChecked(False) # Uncheck button
            self.flight_btn.setText("🚀 Start Blood Flow Tour")
            self.stop_recording() # Stop recording if tour is cancelled
        
        if self.is_orbiting:
            self.is_orbiting = False
            self.orbit_timer.stop()
            self.orbit_btn.setChecked(False) # Uncheck button
            self.orbit_btn.setText("🔄 Start Orbit")
            
        # --- NEW: Stop picking dive point ---
        if self.is_picking_dive_point:
            self.is_picking_dive_point = False
            self.dive_btn.setChecked(False)
            self.statusBar().showMessage("Deep dive mode cancelled")
            
        # --- NEW: Stop active dive ---
        if self.is_diving:
            self.is_diving = False
            self.flight_timer.stop()
            self.stop_recording()
            # Clean up clipping planes
            for segment in self.segment_manager.segments.values():
                segment['mapper'].SetClippingPlanes(self.empty_clip_planes)

    def toggle_guided_tour(self, checked):
        """Starts or stops the 'Deep Dive' camera tour."""
        # This function is now called by the button's 'clicked' signal
        self.is_flight_mode = checked
        if self.is_flight_mode:
            self.stop_all_camera_motion() 
            self.is_flight_mode = True 
            self.is_diving = True # This is a dive
            self.flight_btn.setText("⏹️ Stop Tour")
            
            # --- NEW: Check for recording ---
            if not self.start_recording(is_tour=True): # Check if we are in record mode
                self.is_flight_mode = False # Abort tour if recording fails/cancelled
                self.is_diving = False
                self.flight_btn.setChecked(False)
                self.flight_btn.setText("🚀 Start Blood Flow Tour")
                return
            
            self.statusBar().showMessage("Starting blood flow tour...")
            
            # Store original opacity and make vessels transparent
            self.original_artery_opacity = self.artery_opacity.value() / 100.0
            self.original_vein_opacity = self.vein_opacity.value() / 100.0
            self.set_type_opacity('Artery', 0.2)
            self.set_type_opacity('Vein', 0.2)
            
            # Apply clipping planes to all mappers
            for segment in self.segment_manager.segments.values():
                segment['mapper'].SetClippingPlanes(self.flight_plane_collection)

            self.setup_tour_path() # Create the camera keyframes
            
            self.flight_step = 0
            self.flight_duration = self.flight_speed_slider.value() * 30 
            self.flight_timer.start(33) # ~30 FPS
        else:
            # This part is now handled by stop_all_camera_motion()
            self.stop_all_camera_motion()
            # Restore opacity and remove clipping planes
            self.set_type_opacity('Artery', self.original_artery_opacity * 100) # Need to pass 0-100
            self.set_type_opacity('Vein', self.original_vein_opacity * 100)
            for segment in self.segment_manager.segments.values():
                segment['mapper'].SetClippingPlanes(self.empty_clip_planes)
            self.vtk_widget.GetRenderWindow().Render()
    
    def setup_tour_path(self):
        """--- NEW: Creates a blood-flow camera path ---"""
        camera = self.renderer.GetActiveCamera()
        self.flight_interpolator.Initialize()
        self.flight_interpolator.SetInterpolationTypeToSpline()
        
        # Keyframes based on heart demo model coordinates
        # [Time, [Cam_Pos], [Focal_Point], [View_Up]]
        path = [
            (0.0, [0, 200, 50], [0, 0, 0], [0, 0, 1]),      # 1. Start far away
            (0.1, [20, 100, 10], [20, 60, 0], [0, 0, 1]),   # 2. Approach Vena Cava
            (0.2, [20, 60, 0], [20, 40, 0], [0, 0, 1]),     # 3. Enter Vena Cava
            (0.3, [20, 45, 0], [15, 35, 0], [1, 0, 1]),     # 4. Inside Cava, *banking turn* towards Right Atrium
            (0.4, [15, 35, 0], [0, 0, 0], [0, 0, 1]),       # 5. Inside Right Atrium, looking down into Ventricle
            (0.5, [10, 15, 0], [0, 0, 0], [0, 0, 1]),       # 6. Moving into Ventricle
            (0.6, [0, 10, 0], [0, 40, 0], [0, 0, 1]),       # 7. At bottom of Ventricle, looking up at Aorta
            (0.7, [0, 30, 0], [0, 50, 0], [-1, 0, 1]),      # 8. Moving up into Aorta, *banking turn* left
            (0.8, [0, 60, 0], [0, 100, 0], [0, 0, 1]),      # 9. Flying up through Aorta
            (0.9, [0, 120, 20], [0, 0, 0], [0, 0, 1]),      # 10. Exiting Aorta, pulling back
            (1.0, [0, 200, 50], [0, 0, 0], [0, 0, 1])       # 11. Back to start
        ]
        
        start_cam = vtk.vtkCamera()
        start_cam.DeepCopy(camera)
        self.flight_interpolator.AddCamera(0.0, start_cam)

        for (time, pos, fp, vup) in path:
            key_cam = vtk.vtkCamera()
            key_cam.SetPosition(pos)
            key_cam.SetFocalPoint(fp)
            key_cam.SetViewUp(vup) # Use a stable up vector
            self.flight_interpolator.AddCamera(time, key_cam)

    # --- NEW: Click-to-Dive Function (from user's dental code) ---
    def toggle_dive_mode(self, checked):
        """Toggles the 'Deep Dive at Click' mode."""
        self.is_picking_dive_point = checked
        if checked:
            self.stop_all_camera_motion()
            self.is_picking_dive_point = True
            self.dive_btn.setText("...Click on Model...")
            self.statusBar().showMessage("Deep Dive Mode: Click on the model to start.")
        else:
            self.dive_btn.setText("🚁 Deep Dive at Click")
            self.statusBar().showMessage("Deep Dive mode cancelled.")

    def start_deep_dive(self, target_point, target_normal):
        """Starts a spiral dive tour at the clicked point."""
        self.stop_all_camera_motion()
        self.is_diving = True # This is a dive
        self.statusBar().showMessage(f"Deep dive at {target_point}...")
        
        camera = self.renderer.GetActiveCamera()
        
        self.flight_interpolator.Initialize()
        self.flight_interpolator.SetInterpolationTypeToSpline()
        
        # Add current camera as start
        self.flight_interpolator.AddCamera(0.0, camera)
        
        # Calculate perpendicular vectors for spiraling
        v1 = [0,0,0]
        vtk.vtkMath.Perpendiculars(target_normal, v1, [0,0,0], 0)
        v2 = np.cross(target_normal, v1)
        
        num_keyframes = 10
        dive_depth = 60.0 # How far to dive
        spiral_radius = 15.0 # How wide to spiral
        
        for i in range(1, num_keyframes + 1):
            t = i / num_keyframes
            
            # Point along the normal vector
            dive_point = [target_point[j] - target_normal[j] * (t * dive_depth) for j in range(3)]
            
            # Spiral offsets
            angle = t * np.pi * 4 # 2 full spirals
            spiral_offset_v1 = [v * spiral_radius * np.cos(angle) * (1-t) for v in v1] # Spiral shrinks over time
            spiral_offset_v2 = [v * spiral_radius * np.sin(angle) * (1-t) for v in v2]
            
            cam_pos = [dive_point[j] + spiral_offset_v1[j] + spiral_offset_v2[j] for j in range(3)]
            
            # Focal point is further down the dive path
            focal_point = [target_point[j] - target_normal[j] * (t * dive_depth + 20) for j in range(3)]
            
            dive_cam = vtk.vtkCamera()
            dive_cam.SetPosition(cam_pos)
            dive_cam.SetFocalPoint(focal_point)
            dive_cam.SetViewUp(v2)
            
            self.flight_interpolator.AddCamera(t, dive_cam)
        
        # Apply clipping planes
        for segment in self.segment_manager.segments.values():
            segment['mapper'].SetClippingPlanes(self.flight_plane_collection)

        self.flight_step = 0
        self.flight_duration = self.flight_speed_slider.value() * 30 # Use slider for duration
        self.flight_timer.start(33)
    
    def update_flight_animation(self):
        """Update function for BOTH guided tour and deep dive."""
        self.flight_step += 1
        t = self.flight_step / self.flight_duration
        
        camera = self.renderer.GetActiveCamera()
        
        if t >= 1.0:
            t = 1.0
            self.flight_timer.stop()
            self.statusBar().showMessage("Flight complete!")
            
            # This logic cleans up both tour types
            if self.is_diving:
                self.is_diving = False
                self.is_flight_mode = False # Also turn off pre-programmed tour flag
                self.flight_btn.setChecked(False) # Uncheck button
                self.flight_btn.setText("🚀 Start Blood Flow Tour")
                self.dive_btn.setChecked(False) # Uncheck dive button
                self.dive_btn.setText("🚁 Deep Dive at Click")
                self.stop_recording() # Stop recording
                
                # Restore opacity and remove clipping planes
                self.set_type_opacity('Artery', self.original_artery_opacity * 100)
                self.set_type_opacity('Vein', self.original_vein_opacity * 100)
                for segment in self.segment_manager.segments.values():
                    segment['mapper'].SetClippingPlanes(self.empty_clip_planes)
                self.vtk_widget.GetRenderWindow().Render()
            return
        
        self.flight_interpolator.InterpolateCamera(t, camera)
        
        if self.is_diving: # This is true for both tour types
            # Update clipping plane (the "tunnel" effect)
            cam_pos = camera.GetPosition()
            
            # --- CORRECTED CLIP PLANE LOGIC ---
            # Get the camera's actual forward-facing vector
            cam_direction = camera.GetDirectionOfProjection()
            
            # The normal must point BACKWARD to clip what's behind us.
            clip_normal_backward = [-cam_direction[0], -cam_direction[1], -cam_direction[2]]
            
            # Set the plane's origin to be exactly at the camera's position
            clip_pos = cam_pos
            
            self.flight_clip_plane.SetOrigin(clip_pos)
            self.flight_clip_plane.SetNormal(clip_normal_backward) # <-- THE FIX
            # ------------------------------------
            
            self.vtk_widget.GetRenderWindow().Render()

        # --- NEW: Write video frame if recording ---
        if self.video_writer is not None:
            self.window_to_image_filter.Modified()
            self.video_writer.Write()
            
    # --- NEW: Video Recording Functions ---
    def record_guided_tour(self):
        """Starts the tour with recording enabled."""
        if self.is_flight_mode or self.is_diving:
            QMessageBox.warning(self, "Tour in Progress", "The tour is already running.")
            return
            
        # Set the record_on_start flag and call toggle_guided_tour
        self.record_on_start = True
        self.flight_btn.setChecked(True) # Manually check the button
        self.toggle_guided_tour(True) # Call with checked=True

    def start_recording(self, is_tour=False):
        """Asks for save file and initializes the video writer."""
        # If this function was called by the tour, and not by the record button,
        # we check the internal "record_on_start" flag.
        if is_tour and not getattr(self, 'record_on_start', False):
            return True # Not recording, just start the tour
            
        # Reset the flag
        self.record_on_start = False
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Tour Video", "heart_tour.ogv", "Ogg Video File (*.ogv)")
        
        if not path:
            self.statusBar().showMessage("Video recording cancelled.")
            return False
            
        try:
            self.window_to_image_filter = vtkWindowToImageFilter()
            self.window_to_image_filter.SetInput(self.vtk_widget.GetRenderWindow())
            self.window_to_image_filter.SetInputBufferTypeToRGB()
            self.window_to_image_filter.ReadFrontBufferOff()
            self.window_to_image_filter.Update()
            
            self.video_writer = vtkOggTheoraWriter()
            self.video_writer.SetFileName(path)
            self.video_writer.SetInputConnection(self.window_to_image_filter.GetOutputPort())
            # Set rate to match our animation timer (30 fps)
            self.video_writer.SetRate(30) 
            self.video_writer.Start()
            
            self.statusBar().showMessage(f"🔴 Recording tour to {path}...")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to start recording:\n{e}")
            self.video_writer = None
            self.window_to_image_filter = None
            return False

    def stop_recording(self):
        """Stops and finalizes the video file."""
        if self.video_writer is not None:
            try:
                self.video_writer.End()
                self.statusBar().showMessage("Recording finished successfully.")
            except Exception as e:
                self.statusBar().showMessage(f"Error stopping recording: {e}")
            
            self.video_writer = None
            self.window_to_image_filter = None
    # ------------------------------------

    def toggle_orbit(self, checked):
        """Starts or stops the orbiting camera."""
        self.is_orbiting = checked
        if self.is_orbiting:
            self.stop_all_camera_motion() 
            self.is_orbiting = True 
            self.orbit_btn.setText("⏹️ Stop Orbit")
            self.orbit_timer.start(50) # ~20 FPS
        else:
            self.orbit_timer.stop()
            self.orbit_btn.setText("🔄 Start Orbit")

    def update_orbit(self):
        """Update function for the orbit animation."""
        cam = self.renderer.GetActiveCamera()
        speed = self.orbit_speed_slider.value() * 0.1
        self.orbit_angle += speed
        cam.Azimuth(speed) # Rotate around the focal point
        self.vtk_widget.GetRenderWindow().Render()

    def on_fly_to_button_pressed(self):
        """Triggers focus on the currently selected tree item."""
        selected = self.segment_tree.currentItem()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Select a segment from the tree first")
            return
        
        name = selected.text(0)
        self.select_and_focus_segment(name)
            
    def toggle_focus_nav_mode(self, checked):
        """Activates/deactivates the segment isolation mode."""
        if checked:
            self.stop_all_camera_motion() # Stop tours if starting focus mode
            self.focus_navigator.activate()
            self.focus_nav_btn.setText("Disable Focus Mode")
            self.statusBar().showMessage("Focus Mode Active: Click a part to isolate it.")
            current_item = self.segment_tree.currentItem()
            if current_item:
                self.focus_navigator.focus_on_segment(current_item.text(0))
        else:
            self.focus_navigator.deactivate()
            self.focus_nav_btn.setText("Enable Focus Mode")
            self.statusBar().showMessage("Focus Mode Deactivated.")
        self.vtk_widget.GetRenderWindow().Render()

    def animate_camera_to_actor(self, actor):
        """Calculates target and starts the smooth focus animation."""
        self.stop_all_camera_motion() 
    
        cam = self.renderer.GetActiveCamera()
        
        bounds = actor.GetBounds()
        center = np.array([(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2])
        
        max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        if max_dim == 0: max_dim = 100 
        
        current_cam_pos = np.array(cam.GetPosition())
        vec_to_cam = current_cam_pos - center
        dist = np.linalg.norm(vec_to_cam)
        
        if dist < 1e-6: 
            vec_to_cam = np.array([0, 1, 0]) # Default view vector
        else:
            vec_to_cam = vec_to_cam / dist 
        
        target_dist = max_dim * 2.5 # Zoom out based on object size
        target_pos = center + vec_to_cam * target_dist
        
        self.start_cam_pos = np.array(cam.GetPosition())
        self.start_cam_fp = np.array(cam.GetFocalPoint())
        
        self.target_cam_pos = target_pos
        self.target_cam_fp = center
        
        self.cam_anim_start_time = time.time()
        self.cam_anim_timer.start(30) # ~33 FPS

    def update_camera_animation(self):
        """The update tick for the smooth focus animation."""
        elapsed = time.time() - self.cam_anim_start_time
        t = elapsed / self.cam_anim_duration 
        
        if t >= 1.0:
            t = 1.0
            self.cam_anim_timer.stop() 
            
        t_smooth = t * t * (3.0 - 2.0 * t) # Smooth step
        
        cam = self.renderer.GetActiveCamera()
        
        new_pos = self.start_cam_pos + (self.target_cam_pos - self.start_cam_pos) * t_smooth
        new_fp = self.start_cam_fp + (self.target_cam_fp - self.start_cam_fp) * t_smooth
        
        cam.SetPosition(new_pos)
        cam.SetFocalPoint(new_fp)
        
        if t == 1.0:
            self.renderer.ResetCameraClippingRange()
            
        self.vtk_widget.GetRenderWindow().Render()
    
    def reset_camera(self):
        """Resets camera and stops all camera animations."""
        self.stop_all_camera_motion()
        
        # Manually restore opacity if tour was interrupted
        self.set_type_opacity('Artery', self.original_artery_opacity * 100)
        self.set_type_opacity('Vein', self.original_vein_opacity * 100)
        for segment in self.segment_manager.segments.values():
            segment['mapper'].SetClippingPlanes(self.empty_clip_planes)

        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
    
    # ==================== CLIPPING (from musculoskeletal_system.py) ====================
    
    def open_clipping_dialog(self):
        """Opens the advanced clipping tool."""
        if self.clipping_dialog is None:
            self.clipping_dialog = ClippingDialog(self)
        self.clipping_dialog.show()
        self.clipping_dialog.raise_()
        self.clipping_dialog.activateWindow()

    def get_scene_bounds(self):
        """Get the collective bounds of all actors in the scene."""
        actors = self.segment_manager.get_all_actors()
        if not actors:
             return [-1, 1, -1, 1, -1, 1] # Default bounds

        prop_bounds = vtk.vtkBoundingBox()
        for actor in actors:
            prop_bounds.AddBounds(actor.GetBounds())
        
        bounds_array = [0.0] * 6
        prop_bounds.GetBounds(bounds_array)
        
        # Check for invalid bounds (e.g., a single point)
        if bounds_array[0] == bounds_array[1]: bounds_array[1] += 1
        if bounds_array[2] == bounds_array[3]: bounds_array[3] += 1
        if bounds_array[4] == bounds_array[5]: bounds_array[5] += 1
            
        return bounds_array

    def apply_advanced_clipping(self, params):
        """Applies VTK clipping planes to all actors."""
        if not self.renderer:
            return

        # Remove old visual planes
        for actor in self.plane_actors:
            self.renderer.RemoveActor(actor)
        self.plane_actors.clear()
        
        bounds = self.get_scene_bounds()
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        
        x_pos = xmin + params['x_pos'] * (xmax - xmin)
        y_pos = ymin + params['y_pos'] * (ymax - ymin)
        z_pos = zmin + params['z_pos'] * (zmax - zmin)
        
        planes = vtk.vtkPlaneCollection()
        
        # Normals point *inward* to the visible region
        if params['hide_left']:   # Hide -X -> Keep +X
            p = vtk.vtkPlane(); p.SetOrigin(x_pos, 0, 0); p.SetNormal(1, 0, 0); planes.AddItem(p)
        if params['hide_right']:  # Hide +X -> Keep -X
            p = vtk.vtkPlane(); p.SetOrigin(x_pos, 0, 0); p.SetNormal(-1, 0, 0); planes.AddItem(p)
            
        if params['hide_front']:  # Hide -Y -> Keep +Y
            p = vtk.vtkPlane(); p.SetOrigin(0, y_pos, 0); p.SetNormal(0, 1, 0); planes.AddItem(p)
        if params['hide_back']:   # Hide +Y -> Keep -Y
            p = vtk.vtkPlane(); p.SetOrigin(0, y_pos, 0); p.SetNormal(0, -1, 0); planes.AddItem(p)

        if params['hide_bottom']: # Hide -Z -> Keep +Z
            p = vtk.vtkPlane(); p.SetOrigin(0, 0, z_pos); p.SetNormal(0, 0, 1); planes.AddItem(p)
        if params['hide_top']:    # Hide +Z -> Keep -Z
            p = vtk.vtkPlane(); p.SetOrigin(0, 0, z_pos); p.SetNormal(0, 0, -1); planes.AddItem(p)
                
        # Apply clipping to all segment actors
        for seg in self.segment_manager.segments.values():
            mapper = seg['mapper']
            if planes.GetNumberOfItems() > 0:
                mapper.SetClippingPlanes(planes)
            else:
                mapper.RemoveAllClippingPlanes()
        
        # Show visual planes
        if params['show_axial']: # Z-plane (Axial)
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(xmin, ymin, z_pos)
            plane.SetPoint1(xmax, ymin, z_pos)
            plane.SetPoint2(xmin, ymax, z_pos)
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(plane.GetOutputPort())
            actor = vtk.vtkActor(); actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.2, 0.5, 1.0); actor.GetProperty().SetOpacity(0.4)
            self.renderer.AddActor(actor)
            self.plane_actors.append(actor)
        
        if params['show_sagittal']: # X-plane (Sagittal)
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(x_pos, ymin, zmin)
            plane.SetPoint1(x_pos, ymax, zmin)
            plane.SetPoint2(x_pos, ymin, zmax)
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(plane.GetOutputPort())
            actor = vtk.vtkActor(); actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.2, 0.2); actor.GetProperty().SetOpacity(0.4)
            self.renderer.AddActor(actor)
            self.plane_actors.append(actor)
        
        if params['show_coronal']: # Y-plane (Coronal)
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(xmin, y_pos, zmin)
            plane.SetPoint1(xmax, y_pos, zmin)
            plane.SetPoint2(xmin, y_pos, zmax)
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(plane.GetOutputPort())
            actor = vtk.vtkActor(); actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.2, 1.0, 0.2); actor.GetProperty().SetOpacity(0.4)
            self.renderer.AddActor(actor)
            self.plane_actors.append(actor)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    # ==================== MPR (from musculoskeletal_system.py) ====================
    
    def open_mpr_dialog(self):
        """Opens the curved MPR tool."""
        if self.mpr_dialog is None:
            self.mpr_dialog = CurvedMPRDialog(self)
        self.mpr_dialog.show()
        self.mpr_dialog.raise_()
        self.mpr_dialog.activateWindow()
    
    # ==================== UTILITY ====================
    
    def update_model_center(self):
        """Calculates the center of all loaded actors."""
        actors = self.segment_manager.get_all_actors()
        if not actors:
            self.model_center = np.array([0, 0, 0])
            return

        prop_bounds = vtk.vtkBoundingBox()
        for actor in actors:
            prop_bounds.AddBounds(actor.GetBounds())
        
        center_array = [0.0, 0.0, 0.0]
        prop_bounds.GetCenter(center_array)
        self.model_center = np.array(center_array)

    def closeEvent(self, event):
        """Custom close event to stop all timers."""
        self.stop_all_camera_motion()
        self.animation_timer.stop()
        self.stop_recording() # Ensure recording stops on close
        if self.clipping_dialog:
            self.clipping_dialog.close()
        if self.mpr_dialog:
            self.mpr_dialog.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Provides a modern, dark-theme-friendly style
    
    # Set dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(26, 26, 46))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(15, 15, 30))
    dark_palette.setColor(QPalette.AlternateBase, QColor(26, 26, 46))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(26, 26, 46))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(0, 212, 255))
    dark_palette.setColor(QPalette.Highlight, QColor(0, 212, 255))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    window = Medical3DVisualizationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


