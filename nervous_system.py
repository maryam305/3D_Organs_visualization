import sys
import numpy as np
import pandas as pd
from scipy import interpolate
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox,
                             QColorDialog, QFileDialog, QGroupBox, QGridLayout,
                             QTabWidget, QCheckBox, QSpinBox, QDoubleSpinBox,
                             QTreeWidget, QTreeWidgetItem, QSplitter, QProgressBar,
                             QMessageBox, QListWidget, QDialog, QTextEdit,
                             QStyleFactory, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QUrl
# ---!!! الإصلاح هنا !!! ---
from PyQt5.QtGui import QColor, QPalette, QIcon, QFont, QBrush
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import os
from collections import defaultdict
import time

# --- NEW: Imports from Dental/Musculoskeletal Code for MPR ---
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

try:
    import nibabel as nib
    HAS_NIBABEL = True
except:
    HAS_NIBABEL = False
# --- END NEW IMPORTS ---


# ==================== VTK CLASSES ====================
# --- NEW: SegmentManager from Dental Code (Supports Opacity Widgets) ---
class SegmentManager:
    """Manages individual anatomical segments with independent properties"""
    def __init__(self):
        self.segments = {}
        self.segment_groups = defaultdict(list)
        
    def add_segment(self, name, actor, mapper, reader, system, color=(1, 1, 1)):
        self.segments[name] = {
            'actor': actor,
            'mapper': mapper,
            'reader': reader,
            'opacity': 1.0,
            'color': color,
            'visible': True,
            'system': system,
            'original_transform': vtk.vtkTransform(),
            'current_transform': vtk.vtkTransform(),
            'original_color': color
        }
        self.segment_groups[system].append(name)
        actor.GetProperty().SetColor(*color)
        
    def set_opacity(self, name, opacity):
        if name in self.segments:
            self.segments[name]['opacity'] = opacity
            self.segments[name]['actor'].GetProperty().SetOpacity(opacity)
            
    def set_color(self, name, color):
        if name in self.segments:
            self.segments[name]['color'] = color
            self.segments[name]['actor'].GetProperty().SetColor(*color)
            
    def set_visibility(self, name, visible):
        if name in self.segments:
            self.segments[name]['visible'] = visible
            self.segments[name]['actor'].SetVisibility(visible)
            
    def get_segment(self, name):
        return self.segments.get(name)
    
    def get_segments_by_group(self, group_name):
        """Gets all segments belonging to a specific system/group"""
        return [seg for seg in self.segments.values() if seg['system'] == group_name]

    def get_all_actors(self):
        return [seg['actor'] for seg in self.segments.values()]
    
    def clear(self):
        self.segments.clear()
        self.segment_groups.clear()

# --- NEW: ClippingDialog from Dental Code ---
class ClippingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Clipping Planes")
        self.setGeometry(100, 100, 600, 750)
        self.parent_viewer = parent
        
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(100)
        self.update_timer.timeout.connect(self.apply_clipping_now)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
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
        
        pos_group = QGroupBox("Plane Positions (0-100)")
        pos_layout = QVBoxLayout()
        
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
        
        self.x_slider.valueChanged.connect(lambda v: (self.x_value.setText(str(v)), self.schedule_update()))
        self.y_slider.valueChanged.connect(lambda v: (self.y_value.setText(str(v)), self.schedule_update()))
        self.z_slider.valueChanged.connect(lambda v: (self.z_value.setText(str(v)), self.schedule_update()))
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        clip_group = QGroupBox("Hide Regions (Octant Clipping)")
        clip_layout = QGridLayout()
        
        self.hide_left = QCheckBox("Hide Left (-X)")
        self.hide_right = QCheckBox("Hide Right (+X)")
        self.hide_front = QCheckBox("Hide Front (-Y)")
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
        self.update_timer.start()
    
    def reset_all(self):
        self.x_slider.setValue(50)
        self.y_slider.setValue(50)
        self.z_slider.setValue(50)
        self.show_axial.setChecked(False)
        self.show_sagittal.setChecked(False)
        self.show_coronal.setChecked(False)
        for cb in [self.hide_left, self.hide_right, self.hide_front, 
                   self.hide_back, self.hide_top, self.hide_bottom]:
            cb.setChecked(False)
        self.schedule_update()
    
    def apply_clipping_now(self):
        if self.parent_viewer:
            self.parent_viewer.apply_advanced_clipping(self.get_params())
    
    def get_params(self):
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

# --- NEW: CurvedMPRDialog from Dental Code ---
class CurvedMPRDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Curved Multi-Planar Reconstruction")
        self.setGeometry(100, 100, 900, 800) # Made it a bit taller
        self.parent_viewer = parent
        self.curve_points = []
        self.volume = None
        self.current_slice = None
        
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
        # --- End New Controls ---
        
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        
        self.status = QLabel("Ready")
        self.status.setStyleSheet("padding: 5px; color: #06ffa5;")
        layout.addWidget(self.status)
        
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        self.display_placeholder()
    
    def display_placeholder(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Load volume to begin', ha='center', va='center', fontsize=14, color='gray')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.canvas.draw()
    
    def load_volume(self):
        if not HAS_NIBABEL:
            QMessageBox.warning(self, "Missing Library", "Install: pip install nibabel")
            return
        
        path, _ = QFileDialog.getOpenFileName(self, "Load NIfTI", "", "NIfTI (*.nii *.nii.gz)")
        if not path:
            return
        
        try:
            self.status.setText("Loading, please wait...")
            QApplication.processEvents()
            
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
            
            # Set the initial slice and display it
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
            self.display_slice() # Redraw canvas with new slice + existing curve
            self.status.setText(f"Displaying slice {value}. Curve points are preserved.")

    def display_slice(self):
        if self.current_slice is None:
            self.display_placeholder()
            return
        
        self.ax.clear()
        self.ax.imshow(self.current_slice.T, cmap='gray', aspect='equal', origin='lower')
        self.ax.set_title("Click to draw curve")
        
        if self.curve_points:
            pts = np.array(self.curve_points)
            self.ax.plot(pts[:, 0], pts[:, 1], 'ro-', linewidth=2, markersize=8)
        
        self.canvas.draw()
    
    def on_click(self, event):
        if event.inaxes != self.ax or self.current_slice is None:
            return
        
        self.curve_points.append([event.xdata, event.ydata])
        self.display_slice()
        self.status.setText(f"Points: {len(self.curve_points)}")
    
    def reset_curve(self):
        self.curve_points = []
        if self.volume is not None:
             self.display_slice()
        self.status.setText("Curve reset")
    
    def generate_cpr(self):
        if self.volume is None:
            QMessageBox.warning(self, "Error", "Load volume first")
            return
            
        if len(self.curve_points) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 points")
            return
        
        # --- Get slice range from UI ---
        start_z = self.start_slice_spin.value()
        end_z = self.end_slice_spin.value()

        if start_z >= end_z:
            QMessageBox.warning(self, "Error", "Start slice must be less than end slice.")
            return
        
        try:
            # Create the sub-volume for CPR based on user's selection
            # +1 because Python slicing is exclusive of the end index
            cpr_volume = self.volume[:, :, start_z:end_z+1]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to slice volume:\n{e}")
            return
        # --- End New Slice Logic ---
        
        try:
            points = np.array(self.curve_points)
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            cumulative = np.concatenate([[0], np.cumsum(distances)])
            
            num_samples = int(cumulative[-1] * 2)
            sample_distances = np.linspace(0, cumulative[-1], num_samples)
            
            interp_x = np.interp(sample_distances, cumulative, points[:, 0])
            interp_y = np.interp(sample_distances, cumulative, points[:, 1])
            
            straightened = []
            
            for x, y in zip(interp_x, interp_y):
                xi, yi = int(round(x)), int(round(y))
                
                # Check against the cpr_volume dimensions
                if 0 <= xi < cpr_volume.shape[0] and 0 <= yi < cpr_volume.shape[1]:
                    # Append the Z-stack from the *cropped* volume
                    straightened.append(cpr_volume[xi, yi, :])
                else:
                    # Append a blank Z-stack of the *cropped* height
                    straightened.append(np.zeros(cpr_volume.shape[2]))
            
            straightened = np.array(straightened).T
            
            result_fig = plt.figure(figsize=(12, 8))
            plt.imshow(straightened, cmap='gray', aspect='auto', origin='lower')
            plt.title(f"Straightened Curved MPR (Slices {start_z} to {end_z})", fontsize=16)
            plt.xlabel("Distance along curve")
            plt.ylabel(f"Depth (Slices {start_z}-{end_z})")
            plt.colorbar(label='Intensity')
            plt.tight_layout()
            plt.show()
            
            self.status.setText(f"CPR generated for slices {start_z}-{end_z}!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Generation failed:\n{e}")

# --- MODIFIED: Neural Signal Animator (Stronger Glow, Larger Area) ---
class SurfaceNeuralSignalAnimator:
    """Handles neural signal animation directly on brain surface mesh (Glow Effect)"""
    def __init__(self, renderer, segment_manager):
        self.renderer = renderer
        self.segment_manager = segment_manager
        self.active_process = None
        self.current_frame = 0
        self.signal_speed = 1.0
        self.is_animating = False
        
        self.signal_pathways = self.define_surface_pathways()
        self.original_colors = {}
        self.initial_properties = {}
        
    def define_surface_pathways(self):
        """Define neural pathways as sequences of regions that light up on the surface"""
        
        # --- MODIFIED: Increased radius for larger surface area ---
        pain_regions = [
            {'center': (0.0, -0.4, 0.0), 'radius': 0.4, 'duration': 40}, 
            {'center': (0.3, 0.3, 0.5), 'radius': 0.45, 'duration': 50}, 
            {'center': (-0.3, 0.3, 0.5), 'radius': 0.45, 'duration': 50},
        ]
        
        vision_regions = [
            {'center': (0.0, -0.9, 0.0), 'radius': 0.5, 'duration': 45}, 
            {'center': (-0.5, 0.0, 0.0), 'radius': 0.5, 'duration': 50},
            {'center': (0.0, 0.5, 0.5), 'radius': 0.55, 'duration': 55}, 
        ]
        
        thinking_regions = [
            {'center': (-0.5, 0.5, 0.3), 'radius': 0.5, 'duration': 50}, 
            {'center': (0.5, 0.5, 0.3), 'radius': 0.5, 'duration': 50},  
            {'center': (0.0, 0.5, 0.6), 'radius': 0.5, 'duration': 60},  
        ]

        pathways = {
            'pain': {
                'name': 'Pain Pathway',
                'description': 'Lower spine/Brainstem → Somatosensory cortex (Red/Orange Glow)',
                'color': (1.0, 0.2, 0.0),
                'regions': pain_regions,
                'transition_frames': 15    
            },
            'vision': {
                'name': 'Vision Pathway',
                'description': 'Occipital → Temporal → Frontal cortex (Blue/Cyan Glow)',
                'color': (0.0, 0.5, 1.0),
                'regions': vision_regions,
                'transition_frames': 20
            },
            'thinking': {
                'name': 'Cognitive Processing',
                'description': 'Frontal ↔ Parietal network (Yellow/White Glow)',
                'color': (1.0, 0.9, 0.0),
                'regions': thinking_regions,
                'transition_frames': 18
            }
        }
        return pathways
        
    def start_animation(self, process_type):
        """Start neural signal animation on surface"""
        if process_type not in self.signal_pathways:
            return
        
        self.stop_animation()
        
        for name, segment in self.segment_manager.segments.items():
            prop = segment['actor'].GetProperty()
            self.original_colors[name] = segment['original_color']
            # --- MODIFIED: Store ambient property ---
            self.initial_properties[name] = {
                'specular': prop.GetSpecular(),
                'diffuse': prop.GetDiffuse(),
                'specularPower': prop.GetSpecularPower(),
                'opacity': segment['opacity'],
                'ambient': prop.GetAmbient() # Store original ambient
            }
        
        self.active_process = process_type
        self.current_frame = 0
        self.is_animating = True
        
    def stop_animation(self):
        """Stop animation and restore original colors/properties"""
        self.is_animating = False
        
        for name, color in self.original_colors.items():
            if name in self.segment_manager.segments:
                segment = self.segment_manager.segments[name]
                prop = segment['actor'].GetProperty()
                initial_props = self.initial_properties.get(name, {})
                
                prop.SetColor(*color)
                
                # --- MODIFIED: Restore all properties ---
                if 'opacity' in initial_props:
                    prop.SetOpacity(segment['opacity']) 
                if 'specular' in initial_props:
                    prop.SetSpecular(initial_props['specular'])
                if 'diffuse' in initial_props:
                    prop.SetDiffuse(initial_props['diffuse'])
                if 'specularPower' in initial_props:
                    prop.SetSpecularPower(initial_props['specularPower'])
                if 'ambient' in initial_props:
                    prop.SetAmbient(initial_props['ambient']) # Restore ambient
                    
        self.active_process = None
        self.original_colors.clear()
        self.initial_properties.clear()

    def update_animation(self):
        """Update surface coloring based on current animation frame"""
        if not self.is_animating or self.active_process is None:
            return
        
        pathway = self.signal_pathways[self.active_process]
        regions = pathway['regions']
        transition_frames = pathway['transition_frames']
        glow_color = pathway['color']
        
        brain_bounds = self.get_brain_bounds()
        if brain_bounds[0] > brain_bounds[1]:
             return
            
        brain_center = np.array([(brain_bounds[0] + brain_bounds[1]) / 2,
                                 (brain_bounds[2] + brain_bounds[3]) / 2,
                                 (brain_bounds[4] + brain_bounds[5]) / 2])
        brain_scale = np.array([brain_bounds[1] - brain_bounds[0],
                                brain_bounds[3] - brain_bounds[2],
                                brain_bounds[5] - brain_bounds[4]]) / 2.0
        brain_scale[brain_scale == 0] = 1.0 

        total_duration = sum(r['duration'] for r in regions) + (len(regions)) * transition_frames
        
        frame = self.current_frame % total_duration
        
        cumulative_time = 0
        current_region_idx = 0
        next_region_idx = 0
        blend_factor = 0.0
        
        for idx, region in enumerate(regions):
            region_start = cumulative_time
            region_end = cumulative_time + region['duration']
            transition_end = region_end + transition_frames
            
            if frame < region_end:
                current_region_idx = idx
                next_region_idx = idx
                blend_factor = 0.0
                break
            elif idx < len(regions) - 1 and frame < transition_end:
                current_region_idx = idx
                next_region_idx = idx + 1
                blend_factor = (frame - region_end) / transition_frames
                break
            
            cumulative_time = transition_end
            
        pulse = 0.85 + 0.15 * np.sin(self.current_frame * 0.5 * self.signal_speed)
        
        for name, segment in self.segment_manager.segments.items():
            actor = segment['actor']
            original_color = self.original_colors.get(name, (0.9, 0.9, 0.9)) 
            
            bounds = actor.GetBounds()
            center = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
            normalized_pos = (center - brain_center) / brain_scale
            
            total_influence = 0.0
            
            current_region = regions[current_region_idx]
            distance_current = np.linalg.norm(normalized_pos - np.array(current_region['center']))
            influence_current = max(0, 1.0 - (distance_current / current_region['radius']))
            
            influence_next = 0.0
            if blend_factor > 0 and next_region_idx != current_region_idx:
                next_region = regions[next_region_idx]
                distance_next = np.linalg.norm(normalized_pos - np.array(next_region['center']))
                influence_next = max(0, 1.0 - (distance_next / next_region['radius']))

            if next_region_idx == current_region_idx:
                total_influence = influence_current
            else:
                total_influence = (1.0 - blend_factor) * influence_current + blend_factor * influence_next
            
            total_influence *= pulse
            total_influence = total_influence ** 1.5 

            if total_influence > 0.01:
                
                # --- MODIFIED: Stronger Glow ---
                GLOW_INTENSITY_BOOST = 4.0 # Boost factor for glow
                
                new_color = tuple(
                    oc * (1 - total_influence * 0.9) + gc * total_influence * GLOW_INTENSITY_BOOST
                    for oc, gc in zip(original_color, glow_color)
                )
                
                # Clamp at 1.5 to allow "hot" glow
                new_color = tuple(max(0, min(1.5, c)) for c in new_color) 
                
                actor.GetProperty().SetColor(*new_color)
                
                prop = actor.GetProperty()
                initial_props = self.initial_properties.get(name, {})
                
                prop.SetDiffuse(initial_props.get('diffuse', 0.8) * (1.0 - total_influence * 0.5))
                # Boost ambient light
                prop.SetAmbient(initial_props.get('ambient', 0.2) + total_influence * 0.8) 
                # Stronger specular highlight
                prop.SetSpecular(initial_props.get('specular', 0.4) + total_influence * 4.0) 
                prop.SetSpecularPower(initial_props.get('specularPower', 40) + total_influence * 200) 
                
                current_user_opacity = segment['opacity']
                prop.SetOpacity(min(1.0, current_user_opacity + total_influence * 0.2))

            else:
                # Restore original properties
                prop = actor.GetProperty()
                prop.SetColor(*original_color)
                
                initial_props = self.initial_properties.get(name, {})
                prop.SetDiffuse(initial_props.get('diffuse', 0.8))
                prop.SetSpecular(initial_props.get('specular', 0.4))
                prop.SetSpecularPower(initial_props.get('specularPower', 30))
                prop.SetOpacity(segment['opacity'])
                prop.SetAmbient(initial_props.get('ambient', 0.2)) # Restore ambient

        self.current_frame = int(self.current_frame + self.signal_speed)
    
    def get_brain_bounds(self):
        """Get bounding box of all brain segments"""
        bounds = [1e10, -1e10, 1e10, -1e10, 1e10, -1e10]
        
        for segment in self.segment_manager.segments.values():
            actor_bounds = segment['actor'].GetBounds()
            bounds[0] = min(bounds[0], actor_bounds[0])
            bounds[1] = max(bounds[1], actor_bounds[1])
            bounds[2] = min(bounds[2], actor_bounds[2])
            bounds[3] = max(bounds[3], actor_bounds[3])
            bounds[4] = min(bounds[4], actor_bounds[4])
            bounds[5] = max(bounds[5], actor_bounds[5])
        
        if bounds[0] > bounds[1]:
            return [-10, 10, -10, 10, -10, 10]
            
        return bounds
    
    def set_speed(self, speed):
        """Set animation speed (0.1 to 5.0)"""
        self.signal_speed = max(0.1, min(5.0, speed))


class FocusNavigator:
    """Handles focus navigation - From Cardiovascular code"""
    def __init__(self, segment_manager):
        self.segment_manager = segment_manager
        self.original_properties = {}
        self.is_active = False
        
    def activate(self):
        """Called when Focus Mode is turned ON"""
        self.is_active = True
        self.original_properties.clear()
        for name, segment in self.segment_manager.segments.items():
            prop = segment['actor'].GetProperty()
            self.original_properties[name] = {
                'opacity': prop.GetOpacity(),
                'ambient': prop.GetAmbient(),
            }

    def deactivate(self):
        """Called when Focus Mode is turned OFF"""
        self.is_active = False
        for name, props in self.original_properties.items():
            if name in self.segment_manager.segments:
                # --- MODIFIED: Use segment manager's opacity property ---
                segment = self.segment_manager.segments[name]
                segment['actor'].GetProperty().SetOpacity(segment['opacity']) # Restore to user-set opacity
                segment['actor'].GetProperty().SetAmbient(props['ambient'])
        self.original_properties.clear()
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()

    def focus_on_segment(self, target_segment_name):
        """Called when a segment is CLICKED in focus mode"""
        if not self.is_active:
            return # Should be activated by button
        
        for name, segment in self.segment_manager.segments.items():
            prop = segment['actor'].GetProperty()
            if name == target_segment_name:
                prop.SetOpacity(1.0)
                prop.SetAmbient(0.8)
            else:
                prop.SetOpacity(0.2)
                prop.SetAmbient(self.original_properties.get(name, {}).get('ambient', 0.2))
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()


class Brain3DVisualizationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Brain 3D Visualization (Upgraded)")
        self.setGeometry(50, 50, 1800, 1000)
        
        self.colors = {
            'bg_dark': '#1a1a2e',
            'bg_medium': '#16213e',
            'accent_purple': '#9d4edd',
            'accent_cyan': '#00d4ff',
            'accent_pink': '#ff006e',
            'accent_green': '#06ffa5',
            'accent_yellow': '#ffbe0b',
            'accent_orange': '#ff6700',
            'text_light': '#e0e0e0',
            'panel_bg': '#0f3460'
        }
        
        # --- NEW: Expanded Color/Group Map ---
        # Defines the "system" (group) and color for keywords found in filenames.
        # This is now the "brain" of the color-coding and group opacity system.
        self.color_groups = {
            # Group Name: {color_tuple}, [list_of_keywords]
            'Ventricles': {'color': (0.0, 0.8, 1.0), 'keywords': ['ventricle']},
            'Colliculi': {'color': (0.1, 1.0, 0.2), 'keywords': ['colliculus']},
            'White Matter Tracts': {'color': (1.0, 0.9, 0.1), 'keywords': ['brachium', 'callosum', 'corpus callosum', 'fasciculus', 'capsule']},
            'Limbic Structures': {'color': (1.0, 0.2, 0.8), 'keywords': ['mammillary', 'hippocampus', 'amygdala', 'fornix']},
            'Cortex': {'color': (0.7, 0.6, 0.7), 'keywords': ['hemisphere', 'frontal', 'parietal', 'temporal', 'occipital', 'gyrus', 'cortex']},
            'Cerebellum': {'color': (0.8, 0.7, 0.5), 'keywords': ['cerebellum', 'cerebellar']},
            'Brainstem': {'color': (0.8, 0.5, 0.5), 'keywords': ['brainstem', 'pons', 'medulla', 'midbrain']},
            'Deep Grey Matter': {'color': (1.0, 0.7, 0.4), 'keywords': ['thalamus', 'putamen', 'caudate', 'pallidus', 'basal ganglia']},
            'Other': {'color': (0.7, 0.7, 0.7), 'keywords': []}
        }
        self.default_group = 'Other'
        # This will store the actual QSlider widgets for the groups
        self.group_opacity_sliders = {}
        # --- END NEW ---
        
        self.apply_stylesheet()
        self.setup_vtk_early()
        
        # --- MODIFIED: Use new SegmentManager ---
        self.segment_manager = SegmentManager()
        self.focus_navigator = FocusNavigator(self.segment_manager)
        self.neural_animator = SurfaceNeuralSignalAnimator(self.renderer, self.segment_manager) 
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.neural_timer = QTimer()
        self.neural_timer.timeout.connect(self.update_neural_signals)
        
        # Flying camera
        self.flight_timer = QTimer()
        self.flight_timer.timeout.connect(self.update_flight_animation)
        self.flight_interpolator = vtk.vtkCameraInterpolator()
        self.flight_clip_plane = vtk.vtkPlane()
        self.flight_plane_collection = vtk.vtkPlaneCollection()
        self.flight_plane_collection.AddItem(self.flight_clip_plane)
        self.empty_clip_planes = vtk.vtkPlaneCollection()
        self.flight_step = 0
        self.flight_duration = 30
        self.is_flight_mode = False
        self.is_diving = False
        
        self.animation_frame = 0
        self.picker = None
        self.saved_camera_views = {}
        self.camera_angle = 0
        self.model_center = [0, 0, 0]
        
        # --- NEW: Ported dialogs ---
        self.clipping_dialog = None
        self.mpr_dialog = None
        self.plane_actors = [] # For clipping planes
        
        self.init_ui()
    
    def setup_vtk_early(self):
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.18)
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground2(0.2, 0.1, 0.3)
        
    def apply_stylesheet(self):
        # Stylesheet (copied from dental code)
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
            QPushButton:pressed {{
                background-color: {self.colors['accent_pink']};
            }}
            QPushButton:checked {{
                background-color: {self.colors['accent_green']};
            }}
            QGroupBox {{
                border: 2px solid {self.colors['accent_purple']};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: {self.colors['accent_cyan']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {self.colors['panel_bg']};
                color: {self.colors['text_light']};
                border: 2px solid {self.colors['accent_purple']};
                border-radius: 4px;
                padding: 5px;
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
            QTreeWidget, QListWidget {{
                background-color: {self.colors['bg_medium']};
                color: {self.colors['text_light']};
                border: 2px solid {self.colors['accent_purple']};
                border-radius: 4px;
            }}
            QTreeWidget::item:selected, QListWidget::item:selected {{
                background-color: {self.colors['accent_purple']};
            }}
            QCheckBox {{
                color: {self.colors['text_light']};
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid {self.colors['accent_purple']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {self.colors['accent_green']};
            }}
            QProgressBar {{
                border: 2px solid {self.colors['accent_purple']};
                border-radius: 5px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['accent_green']};
            }}
        """)
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        self.vtk_widget.setMinimumSize(800, 600)
        splitter.addWidget(self.vtk_widget)
        
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([350, 900, 350]) # Adjusted right panel size
        main_layout.addWidget(splitter)
        
        self.setup_vtk()
        self.neural_animator.renderer = self.renderer 
        self.focus_navigator.vtk_widget = self.vtk_widget
        self.statusBar().showMessage("Ready - Load brain model")
        
    def setup_vtk(self):
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        light1 = vtk.vtkLight()
        light1.SetPosition(100, 100, 100)
        light1.SetIntensity(1.0)
        self.renderer.AddLight(light1)
        
        light2 = vtk.vtkLight()
        light2.SetPosition(-100, -100, 100)
        light2.SetColor(0.6, 0.8, 1.0)
        light2.SetIntensity(0.6)
        self.renderer.AddLight(light2)
        
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.picker.PickFromListOn()
        self.picker.InitializePickList()
        self.interactor.SetPicker(self.picker)
        
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_click, 1.0)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self.on_left_up, 1.0)
        
        self.interactor.Initialize()
        
    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        title = QLabel("🧠 Brain Visualizer")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {self.colors['accent_cyan']}; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # --- MODIFIED: Simplified Data Loading ---
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout()
        
        load_segment_btn = QPushButton("📁 Load Single Segment")
        load_segment_btn.clicked.connect(self.load_segment_file)
        data_layout.addWidget(load_segment_btn)
        
        load_folder_btn = QPushButton("📂 Load Segments Folder")
        load_folder_btn.clicked.connect(self.load_segments_folder)
        data_layout.addWidget(load_folder_btn)
        
        load_demo_brain_btn = QPushButton("🧠 Load Demo Brain Model")
        load_demo_brain_btn.clicked.connect(self.load_demo_brain)
        load_demo_brain_btn.setStyleSheet(f"background-color: {self.colors['accent_green']}; font-size: 13px; padding: 10px;")
        data_layout.addWidget(load_demo_brain_btn)
        
        self.data_status_label = QLabel("No data loaded")
        self.data_status_label.setStyleSheet(f"color: {self.colors['accent_yellow']};")
        self.data_status_label.setWordWrap(True)
        data_layout.addWidget(self.data_status_label)
        
        reset_btn = QPushButton("🔄 RESET - Clear Model")
        reset_btn.setStyleSheet(f"background-color: {self.colors['accent_orange']}; font-size: 12px; padding: 10px;")
        reset_btn.clicked.connect(self.reset_all)
        data_layout.addWidget(reset_btn)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        # --- END MODIFIED DATA LOADING ---
        
        neural_group = QGroupBox("⚡ Neural Signals")
        neural_layout = QVBoxLayout()
        
        info_label = QLabel("Watch neural signals travel across the brain surface")
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {self.colors['text_light']}; font-size: 10px; padding: 5px;")
        neural_layout.addWidget(info_label)
        
        process_layout = QGridLayout()
        
        self.pain_btn = QPushButton("🔴 Pain")
        self.pain_btn.setCheckable(True)
        self.pain_btn.clicked.connect(lambda: self.start_neural_animation('pain'))
        process_layout.addWidget(self.pain_btn, 0, 0)
        
        self.vision_btn = QPushButton("🔵 Vision")
        self.vision_btn.setCheckable(True)
        self.vision_btn.clicked.connect(lambda: self.start_neural_animation('vision'))
        process_layout.addWidget(self.vision_btn, 0, 1)
        
        self.thinking_btn = QPushButton("🟡 Thinking")
        self.thinking_btn.setCheckable(True)
        self.thinking_btn.clicked.connect(lambda: self.start_neural_animation('thinking'))
        process_layout.addWidget(self.thinking_btn, 1, 0)
        
        self.stop_neural_btn = QPushButton("⏹️ Stop")
        self.stop_neural_btn.clicked.connect(self.stop_neural_animation)
        process_layout.addWidget(self.stop_neural_btn, 1, 1)
        
        neural_layout.addLayout(process_layout)
        
        neural_layout.addWidget(QLabel("Speed:"))
        self.neural_speed_slider = QSlider(Qt.Horizontal)
        self.neural_speed_slider.setMinimum(10)
        self.neural_speed_slider.setMaximum(300)
        self.neural_speed_slider.setValue(100)
        self.neural_speed_slider.valueChanged.connect(self.update_neural_speed)
        neural_layout.addWidget(self.neural_speed_slider)
        
        self.neural_speed_label = QLabel("Speed: 1.0x")
        self.neural_speed_label.setAlignment(Qt.AlignCenter)
        neural_layout.addWidget(self.neural_speed_label)
        
        self.neural_info_label = QLabel("Ready")
        self.neural_info_label.setAlignment(Qt.AlignCenter)
        self.neural_info_label.setStyleSheet(f"color: {self.colors['accent_cyan']}; padding: 5px;")
        self.neural_info_label.setWordWrap(True)
        neural_layout.addWidget(self.neural_info_label)
        
        neural_group.setLayout(neural_layout)
        layout.addWidget(neural_group)
        
        # --- MODIFIED: Segment Tree with Opacity Column ---
        segment_group = QGroupBox("Brain Segments")
        segment_layout = QVBoxLayout()
        
        self.segment_tree = QTreeWidget()
        self.segment_tree.setHeaderLabels(["Segment", "Opacity"])
        self.segment_tree.setColumnWidth(0, 150)
        self.segment_tree.itemChanged.connect(self.on_segment_tree_changed)
        self.segment_tree.itemClicked.connect(self.on_segment_clicked)
        segment_layout.addWidget(self.segment_tree)
        
        segment_group.setLayout(segment_layout)
        layout.addWidget(segment_group)
        
        # --- REMOVED: Old "Quick Opacity" Group ---
        
        layout.addStretch()
        return panel
        
    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        tabs = QTabWidget()
        
        # --- MODIFIED: Added Clipping and new MPR tab ---
        tabs.addTab(self.create_visualization_tab(), "🎨 Rendering")
        tabs.addTab(self.create_clipping_tab(), "✂️ Clipping")
        tabs.addTab(self.create_mpr_tab(), "📐 Curved MPR")
        tabs.addTab(self.create_navigation_tab(), "🧭 Navigation")
        
        layout.addWidget(tabs)
        return panel
    
    # --- NEW: create_mpr_tab from Dental ---
    def create_mpr_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info = QLabel("Curved Multi-Planar Reconstruction:\nLoad and draw paths through VOLUMES (NIfTI)")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {self.colors['accent_cyan']}; padding: 10px;")
        layout.addWidget(info)
        
        open_btn = QPushButton("📐 Open MPR Tool (For Volumes)")
        open_btn.setStyleSheet(f"background-color: {self.colors['accent_green']}; font-size: 14px; padding: 12px;")
        open_btn.clicked.connect(self.open_mpr_dialog)
        layout.addWidget(open_btn)
        
        layout.addStretch()
        return tab

    # --- NEW: create_clipping_tab from Dental ---
    def create_clipping_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info = QLabel("Advanced clipping with visible anatomical planes. This tool allows for complex octant clipping.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {self.colors['accent_cyan']}; padding: 10px;")
        layout.addWidget(info)
        
        open_btn = QPushButton("🔓 Open Advanced Clipping")
        open_btn.setStyleSheet(f"background-color: {self.colors['accent_green']}; font-size: 14px; padding: 12px;")
        open_btn.clicked.connect(self.open_clipping_dialog)
        layout.addWidget(open_btn)
        
        layout.addStretch()
        return tab
    
    # --- NEW: Helper for Master Opacity ---
    def create_master_opacity_group(self):
        master_group = QGroupBox("Master Opacity Control")
        master_layout = QVBoxLayout()
        
        self.master_opacity_slider = QSlider(Qt.Horizontal)
        self.master_opacity_slider.setMinimum(0)
        self.master_opacity_slider.setMaximum(100)
        self.master_opacity_slider.setValue(100)
        self.master_opacity_slider.valueChanged.connect(self.update_master_opacity)
        master_layout.addWidget(self.master_opacity_slider)
        
        self.master_opacity_label = QLabel("100%")
        self.master_opacity_label.setAlignment(Qt.AlignCenter)
        master_layout.addWidget(self.master_opacity_label)
        
        master_group.setLayout(master_layout)
        return master_group
    
    # --- NEW: Helper for Color Group Opacity ---
    def create_color_group_opacity_group(self):
        """Dynamically creates sliders for each color group."""
        group = QGroupBox("Color Group Opacity")
        layout = QVBoxLayout()
        
        # Clear old slider references
        self.group_opacity_sliders.clear()

        # Create sliders in the order defined in the map
        for group_name, info in self.color_groups.items():
            # Get color for styling the label
            r, g, b = [int(c * 255) for c in info['color']]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            
            label = QLabel(group_name)
            label.setStyleSheet(f"color: {hex_color}; font-weight: bold;")
            layout.addWidget(label)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(100)
            # Use a lambda with a default argument to capture the group_name
            slider.valueChanged.connect(lambda val, g=group_name: self.update_group_opacity(g, val))
            layout.addWidget(slider)
            
            # Store the slider widget
            self.group_opacity_sliders[group_name] = slider

        group.setLayout(layout)
        return group
        
    def create_visualization_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # --- NEW: Master Opacity ---
        layout.addWidget(self.create_master_opacity_group())
        
        # --- NEW: Color Group Opacity ---
        layout.addWidget(self.create_color_group_opacity_group())
        
        rendering_group = QGroupBox("Rendering")
        rendering_layout = QVBoxLayout()
        
        self.smooth_checkbox = QCheckBox("Smooth Shading")
        self.smooth_checkbox.setChecked(True)
        self.smooth_checkbox.stateChanged.connect(self.toggle_smooth_shading)
        rendering_layout.addWidget(self.smooth_checkbox)
        
        self.edge_checkbox = QCheckBox("Show Edges")
        self.edge_checkbox.stateChanged.connect(self.toggle_edges)
        rendering_layout.addWidget(self.edge_checkbox)
        
        rendering_group.setLayout(rendering_layout)
        layout.addWidget(rendering_group)
        
        layout.addStretch()
        return tab
        
    def create_navigation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Flying Camera Controls
        flight_group = QGroupBox("✈️ Flying Camera")
        flight_layout = QVBoxLayout()
        
        self.flight_btn = QPushButton("✈️ Select Deep Dive Target")
        self.flight_btn.setCheckable(True)
        self.flight_btn.clicked.connect(self.toggle_flight_mode)
        flight_layout.addWidget(self.flight_btn)
        
        flight_layout.addWidget(QLabel("Flight Duration:"))
        self.flight_speed_slider = QSlider(Qt.Horizontal)
        self.flight_speed_slider.setMinimum(10)
        self.flight_speed_slider.setMaximum(50)
        self.flight_speed_slider.setValue(30)
        flight_layout.addWidget(self.flight_speed_slider)
        
        flight_group.setLayout(flight_layout)
        layout.addWidget(flight_group)
        
        # Focus Navigation
        focus_group = QGroupBox("🎯 Focus Navigation")
        focus_layout = QVBoxLayout()
        
        self.focus_nav_btn = QPushButton("🎯 Focus Navigation")
        self.focus_nav_btn.setCheckable(True)
        self.focus_nav_btn.clicked.connect(self.toggle_focus_navigation)
        focus_layout.addWidget(self.focus_nav_btn)
        
        focus_info = QLabel("Click on segments to focus and glow")
        focus_info.setWordWrap(True)
        focus_info.setStyleSheet(f"color: {self.colors['text_light']}; font-size: 10px;")
        focus_layout.addWidget(focus_info)
        
        focus_group.setLayout(focus_layout)
        layout.addWidget(focus_group)
        
        anim_group = QGroupBox("Rotation Animation")
        anim_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_animation)
        btn_layout.addWidget(self.play_btn)
        
        self.reset_anim_btn = QPushButton("⏮️ Reset")
        self.reset_anim_btn.clicked.connect(self.reset_animation)
        btn_layout.addWidget(self.reset_anim_btn)
        anim_layout.addLayout(btn_layout)
        
        anim_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(100)
        anim_layout.addWidget(self.speed_slider)
        
        anim_group.setLayout(anim_layout)
        layout.addWidget(anim_group)
        
        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout()
        
        reset_camera_btn = QPushButton("🎥 Reset Camera")
        reset_camera_btn.clicked.connect(self.reset_camera)
        camera_layout.addWidget(reset_camera_btn)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        layout.addStretch()
        return tab
    
     # ==================== Event Handlers ====================
    def on_left_click(self, obj, event):
        handled = False
        
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        
        if self.picker.GetCellId() != -1:
            target_point = self.picker.GetPickPosition()
            target_normal = self.picker.GetPickNormal()
            
            clicked_actor = self.picker.GetActor()
            segment_name = None
            for name, segment in self.segment_manager.segments.items():
                if segment['actor'] == clicked_actor:
                    segment_name = name
                    break
            
            if self.is_flight_mode:
                self.start_deep_dive(target_point, target_normal)
                self.is_flight_mode = False
                self.flight_btn.setChecked(False)
                self.flight_btn.setText("✈️ Select Deep Dive Target")
                handled = True
                
            elif self.focus_navigator.is_active:
                if segment_name:
                    self.focus_navigator.focus_on_segment(segment_name)
                    self.start_focus_flight(target_point, target_normal)
                    self.statusBar().showMessage(f"Focused on: {segment_name}")
                    handled = True
        
        if not handled:
            self.interactor.GetInteractorStyle().OnLeftButtonDown()
    
    def on_left_up(self, obj, event):
        self.interactor.GetInteractorStyle().OnLeftButtonUp()
    
    # --- NEW: on_segment_tree_changed from Dental ---
    def on_segment_tree_changed(self, item, column):
        """Handle visibility change for both groups and individual segments"""
        if column == 0:
            self.segment_tree.blockSignals(True) 
            
            is_checked = item.checkState(0) == Qt.Checked
            
            if item.parent() is None: # Top-level group item
                for i in range(item.childCount()):
                    child_item = item.child(i)
                    child_name = child_item.text(0)
                    self.segment_manager.set_visibility(child_name, is_checked)
                    child_item.setCheckState(0, Qt.Checked if is_checked else Qt.Unchecked)
            else: # Leaf segment item
                segment_name = item.text(0)
                self.segment_manager.set_visibility(segment_name, is_checked)
                
                # Update parent state
                parent = item.parent()
                all_unchecked = True
                some_checked = False
                for i in range(parent.childCount()):
                    if parent.child(i).checkState(0) == Qt.Checked:
                        all_unchecked = False
                        some_checked = True
                        break
                
                if all_unchecked:
                    parent.setCheckState(0, Qt.Unchecked)
                elif some_checked:
                    parent.setCheckState(0, Qt.Checked) 

            self.segment_tree.blockSignals(False)
            self.vtk_widget.GetRenderWindow().Render()
            
    def on_segment_clicked(self, item, column):
        segment_name = item.text(0)
        
        if item.parent() is None:
            item.setExpanded(not item.isExpanded())
            return
            
        if segment_name in self.segment_manager.segments:
            segment = self.segment_manager.segments[segment_name]
            self.statusBar().showMessage(
                f"Selected: {segment_name} | Opacity: {segment['opacity']*100:.0f}%"
            )
            
            if self.focus_navigator.is_active:
                # Find the actor's center to fly to
                actor = segment['actor']
                bounds = actor.GetBounds()
                center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
                normal = [0, 0, 1] # Dummy normal, just fly to center
                
                self.focus_navigator.focus_on_segment(segment_name)
                self.start_focus_flight(center, normal)
                self.vtk_widget.GetRenderWindow().Render()
    
    # ==================== Model Center Calculation ====================
    def update_model_center(self):
        actors = self.segment_manager.get_all_actors()
        if not actors:
            self.model_center = [0, 0, 0]
            return

        bounds = vtk.vtkBoundingBox()
        for actor in actors:
            bounds.AddBounds(actor.GetBounds())
        
        min_pt = [0,0,0]
        max_pt = [0,0,0]
        bounds.GetMinPoint(min_pt)
        bounds.GetMaxPoint(max_pt)
        
        self.model_center = [(min_pt[i] + max_pt[i]) / 2.0 for i in range(3)]
        self.renderer.ResetCameraClippingRange()
    
    # ==================== Flying Camera Methods ====================
    def toggle_flight_mode(self, checked):
        self.is_flight_mode = checked
        if checked:
            self.flight_btn.setText("🎯 Click on Model to Dive")
            self.statusBar().showMessage("Select target to fly to")
            # --- MODIFIED: Disable other modes ---
            if self.focus_navigator.is_active:
                self.focus_navigator.deactivate()
                self.focus_nav_btn.setChecked(False)
        else:
            self.flight_btn.setText("✈️ Select Deep Dive Target")
            self.statusBar().showMessage("Flight mode deactivated")
    
    def start_deep_dive(self, target_point, target_normal):
        self.statusBar().showMessage(f"Deep dive at {target_point}...")
        
        camera = self.renderer.GetActiveCamera()
        
        self.flight_interpolator.Initialize()
        self.flight_interpolator.SetInterpolationTypeToSpline()
        
        self.flight_interpolator.AddCamera(0.0, camera)
        
        v1 = [0,0,0]
        vtk.vtkMath.Perpendiculars(target_normal, v1, [0,0,0], 0)
        v2 = np.cross(target_normal, v1)
        
        num_keyframes = 10
        dive_depth = 60.0
        spiral_radius = 15.0
        
        for i in range(1, num_keyframes + 1):
            t = i / num_keyframes
            
            dive_point = [target_point[j] - target_normal[j] * (t * dive_depth) for j in range(3)]
            
            angle = t * np.pi * 4
            spiral_offset_v1 = [v * spiral_radius * np.cos(angle) for v in v1]
            spiral_offset_v2 = [v * spiral_radius * np.sin(angle) for v in v2]
            
            cam_pos = [dive_point[j] + spiral_offset_v1[j] + spiral_offset_v2[j] for j in range(3)]
            
            focal_point = [target_point[j] - target_normal[j] * (t * dive_depth + 20) for j in range(3)]
            
            dive_cam = vtk.vtkCamera()
            dive_cam.SetPosition(cam_pos)
            dive_cam.SetFocalPoint(focal_point)
            dive_cam.SetViewUp(v2)
            
            self.flight_interpolator.AddCamera(t, dive_cam)
        
        self.is_diving = True
        for segment in self.segment_manager.segments.values():
            segment['mapper'].SetClippingPlanes(self.flight_plane_collection)

        self.flight_step = 0
        self.flight_duration = self.flight_speed_slider.value() * 3 
        self.flight_timer.start(33)
    
    def start_focus_flight(self, target_point, target_normal):
        # Simple camera dolly, no complex flight
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(target_point)
        camera.Dolly(1.1)
        self.vtk_widget.GetRenderWindow().Render()
    
    def update_flight_animation(self):
        self.flight_step += 1
        t = self.flight_step / self.flight_duration
        
        camera = self.renderer.GetActiveCamera()
        
        if t >= 1.0:
            t = 1.0
            self.flight_timer.stop()
            self.statusBar().showMessage("Flight complete!")
            
            if self.is_diving:
                self.is_diving = False
                for segment in self.segment_manager.segments.values():
                    segment['mapper'].SetClippingPlanes(self.empty_clip_planes)
                self.vtk_widget.GetRenderWindow().Render()
            return
        
        self.flight_interpolator.InterpolateCamera(t, camera)
        
        if self.is_diving:
            cam_pos = camera.GetPosition()
            cam_focal = camera.GetFocalPoint()
            
            cam_normal = [cam_focal[i] - cam_pos[i] for i in range(3)]
            vtk.vtkMath.Normalize(cam_normal)
            
            clip_pos = [cam_pos[i] + cam_normal[i] * 1.0 for i in range(3)]
            
            self.flight_clip_plane.SetOrigin(clip_pos)
            self.flight_clip_plane.SetNormal(cam_normal)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_focus_navigation(self, checked):
        if checked:
            self.is_flight_mode = False
            self.flight_btn.setChecked(False)

            self.focus_navigator.activate()
            self.focus_nav_btn.setText("🔴 Focus Mode ON")
            self.statusBar().showMessage("Click segments to focus")
        else:
            self.focus_navigator.deactivate()
            self.focus_nav_btn.setText("🎯 Focus Navigation")
            self.statusBar().showMessage("Focus mode disabled")
    
    # ==================== Neural Animation Methods ====================
    def start_neural_animation(self, process_type):
        if not self.segment_manager.segments:
            QMessageBox.warning(self, "No Brain", "Load brain model first")
            self.pain_btn.setChecked(False)
            self.vision_btn.setChecked(False)
            self.thinking_btn.setChecked(False)
            return

        button_map = {'pain': self.pain_btn, 'vision': self.vision_btn, 'thinking': self.thinking_btn}
        pressed_btn = button_map.get(process_type)
        
        if pressed_btn and not pressed_btn.isChecked():
            self.stop_neural_animation()
            return
        
        for ptype, btn in button_map.items():
            if ptype != process_type:
                btn.setChecked(False)
        
        self.neural_animator.start_animation(process_type)
        if not self.neural_timer.isActive():
            self.neural_timer.start(33)
        
        pathway_info = self.neural_animator.signal_pathways[process_type]
        self.neural_info_label.setText(f"Active: {pathway_info['name']}\n{pathway_info['description']}") 
        self.statusBar().showMessage(f"✨ {pathway_info['name']} started")
        
        self.vtk_widget.GetRenderWindow().Render()
        
    def stop_neural_animation(self):
        self.neural_timer.stop()
        self.neural_animator.stop_animation()
        self.pain_btn.setChecked(False)
        self.vision_btn.setChecked(False)
        self.thinking_btn.setChecked(False)
        self.neural_info_label.setText("Ready")
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Neural animation stopped")
    
    def update_neural_signals(self):
        if self.neural_animator.is_animating:
            self.neural_animator.update_animation()
            self.vtk_widget.GetRenderWindow().Render()
    
    def update_neural_speed(self, value):
        speed = value / 100.0
        self.neural_animator.set_speed(speed)
        self.neural_speed_label.setText(f"Speed: {speed:.1f}x")
    
    # ==================== Data Loading (MODIFIED) ====================
    
    # --- NEW: `add_segment_to_tree` from Dental, adapted for Brain ---
    def add_segment_to_tree(self, segment_name, system):
        """Add segment under a hierarchical group and set up opacity slider."""
        item = QTreeWidgetItem([segment_name, "100%"])
        item.setCheckState(0, Qt.Checked)
        
        opacity_widget = QWidget()
        opacity_layout = QHBoxLayout(opacity_widget)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(100)
        opacity_slider.valueChanged.connect(
            lambda val, name=segment_name: self.update_segment_opacity(name, val)
        )
        opacity_layout.addWidget(opacity_slider)
        
        # --- MODIFIED: Use the 'system' var directly as the group name ---
        root_name = system 

        root_item = None
        for i in range(self.segment_tree.topLevelItemCount()):
            temp_item = self.segment_tree.topLevelItem(i)
            if temp_item.text(0) == root_name:
                root_item = temp_item
                break
        
        if root_item is None:
            root_item = QTreeWidgetItem([root_name, "Group"])
            # Set color of group item
            group_color = self.color_groups.get(root_name, {}).get('color', (1,1,1))
            r, g, b = [int(c * 255) for c in group_color]
            item_color = QColor(r, g, b)
            # ---!!! FIXED: Added QBrush import ---
            root_item.setForeground(0, QBrush(item_color))
            root_item.setFont(0, QFont("Segoe UI", 9, QFont.Bold))
            
            root_item.setCheckState(0, Qt.Checked)
            self.segment_tree.addTopLevelItem(root_item)
            
        root_item.addChild(item)
        self.segment_tree.setItemWidget(item, 1, opacity_widget)
        root_item.setExpanded(True)
            
    # --- NEW: `load_segment_file` from Dental ---
    def load_segment_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Segment", "",
            "3D Files (*.stl *.obj *.ply *.vtk *.glb);;All Files (*)"
        )
        if file_path:
            filename = os.path.basename(file_path)
            segment_name = os.path.splitext(filename)[0]
            
            self.load_segment(file_path, segment_name)
            self.update_model_center()
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
    # --- NEW: `load_segments_folder` from Dental ---
    def load_segments_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with 3D Models")
        if folder_path:
            files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.stl', '.obj', '.ply', '.vtk', '.glb'))]
            
            if not files:
                QMessageBox.warning(self, "No Files", "No 3D model files found in folder")
                return
            
            for i, filename in enumerate(files):
                file_path = os.path.join(folder_path, filename)
                segment_name = os.path.splitext(filename)[0]
                
                # --- MODIFIED: Color/System assigned in load_segment ---
                self.load_segment(file_path, segment_name)
            
            self.update_model_center()
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.statusBar().showMessage(f"Loaded {len(files)} segments from folder")
            self.data_status_label.setText(f"✓ {len(files)} segments loaded")
            self.data_status_label.setStyleSheet(f"color: {self.colors['accent_green']};")

    # --- MODIFIED: `load_segment` (replaces `_load_single_segment_file`) ---
    def load_segment(self, file_path, segment_name):
        ext = os.path.splitext(file_path)[1].lower()
        
        reader = None
        if ext == '.stl':
            reader = vtk.vtkSTLReader()
        elif ext == '.obj':
            reader = vtk.vtkOBJReader()
        elif ext == '.ply':
            reader = vtk.vtkPLYReader()
        elif ext == '.vtk':
            reader = vtk.vtkPolyDataReader()
        elif ext == '.glb':
            if hasattr(vtk, 'vtkGLTFReader'):
                reader = vtk.vtkGLTFReader()
            else:
                print("GLTF Reader not available in this VTK version.")
                return
        else:
            return
        
        reader.SetFileName(file_path)
        reader.Update()
        
        polydata = reader.GetOutput()
        
        if not polydata or polydata.GetNumberOfPoints() == 0:
            print(f"Warning: Could not read or file is empty: {file_path}")
            return

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(polydata)
        smoother.SetNumberOfIterations(15)
        smoother.BoundarySmoothingOn()
        
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
        prop.SetSpecular(0.5)
        prop.SetSpecularPower(30)
        prop.SetAmbient(0.2)
        prop.SetDiffuse(0.8)
        
        # --- NEW: Assign color and system based on name ---
        def get_segment_info(segment_name):
            name_lower = segment_name.lower().replace('_', ' ')
            # Iterate through groups to find a keyword match
            for group_name, info in self.color_groups.items():
                for keyword in info['keywords']:
                    if keyword in name_lower:
                        return group_name, info['color']
            # No match found, return default
            return self.default_group, self.color_groups[self.default_group]['color']

        system, color = get_segment_info(segment_name)
        # --- END NEW ---
        
        self.segment_manager.add_segment(segment_name, actor, mapper, reader, system, color)
        self.renderer.AddActor(actor)
        self.picker.AddPickList(actor)
        
        self.add_segment_to_tree(segment_name, system) # Use new tree function
        
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"Loaded: {segment_name}")
        return True

    def load_demo_brain(self):
        self.reset_all() # Clear everything first
        
        # --- NEW: Use color map ---
        cortex_group, cortex_color = self.default_group, self.color_groups[self.default_group]['color']
        for g, i in self.color_groups.items():
            if 'hemisphere' in i['keywords']:
                cortex_group, cortex_color = g, i['color']
                break

        brainstem_group, brainstem_color = self.default_group, self.color_groups[self.default_group]['color']
        for g, i in self.color_groups.items():
            if 'brainstem' in i['keywords']:
                brainstem_group, brainstem_color = g, i['color']
                break

        OPACITY = 1.0

        # Left hemisphere
        sphere1 = vtk.vtkParametricEllipsoid()
        sphere1.SetXRadius(45); sphere1.SetYRadius(50); sphere1.SetZRadius(40)
        source1 = vtk.vtkParametricFunctionSource()
        source1.SetParametricFunction(sphere1)
        source1.SetUResolution(100); source1.SetVResolution(100); source1.Update()
        mapper1 = vtk.vtkPolyDataMapper()
        mapper1.SetInputConnection(source1.GetOutputPort())
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper1)
        actor1.GetProperty().SetOpacity(OPACITY)
        actor1.SetPosition(-5, 0, 0)
        
        self.segment_manager.add_segment("Left_Hemisphere", actor1, mapper1, source1, cortex_group, cortex_color)
        self.renderer.AddActor(actor1)
        self.picker.AddPickList(actor1)
        self.add_segment_to_tree("Left_Hemisphere", cortex_group)
        
        # Right hemisphere
        sphere2 = vtk.vtkParametricEllipsoid()
        sphere2.SetXRadius(45); sphere2.SetYRadius(50); sphere2.SetZRadius(40)
        source2 = vtk.vtkParametricFunctionSource()
        source2.SetParametricFunction(sphere2)
        source2.SetUResolution(100); source2.SetVResolution(100); source2.Update()
        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputConnection(source2.GetOutputPort())
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)
        actor2.GetProperty().SetOpacity(OPACITY)
        actor2.SetPosition(5, 0, 0)
        
        self.segment_manager.add_segment("Right_Hemisphere", actor2, mapper2, source2, cortex_group, cortex_color)
        self.renderer.AddActor(actor2)
        self.picker.AddPickList(actor2)
        self.add_segment_to_tree("Right_Hemisphere", cortex_group)
        
        # Brainstem
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(15); cylinder.SetHeight(35); cylinder.SetResolution(50); cylinder.Update()
        mapper3 = vtk.vtkPolyDataMapper()
        mapper3.SetInputConnection(cylinder.GetOutputPort())
        actor3 = vtk.vtkActor()
        actor3.SetMapper(mapper3)
        actor3.GetProperty().SetOpacity(OPACITY)
        actor3.SetPosition(0, -45, -20)
        actor3.RotateX(90)
        
        self.segment_manager.add_segment("Brainstem", actor3, mapper3, cylinder, brainstem_group, brainstem_color)
        self.renderer.AddActor(actor3)
        self.picker.AddPickList(actor3)
        self.add_segment_to_tree("Brainstem", brainstem_group)
        
        # --- MODIFIED: Update new sliders ---
        self.master_opacity_slider.setValue(100)
        for slider in self.group_opacity_sliders.values():
            slider.setValue(100)
        # --- END MODIFIED ---
        
        self.update_model_center()
        
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(30)
        camera.Elevation(20)
        camera.Dolly(1.2)
        
        self.vtk_widget.GetRenderWindow().Render()
        
        self.data_status_label.setText("✓ Demo brain loaded")
        self.data_status_label.setStyleSheet(f"color: {self.colors['accent_green']};")
        self.statusBar().showMessage("✓ Demo brain ready for neural signals")
    
    # ==================== Visualization (Opacity) ====================
    
    # --- NEW: `update_segment_opacity` from Dental ---
    def update_segment_opacity(self, segment_name, value):
        """Update opacity for a single segment based on its slider"""
        opacity = value / 100.0
        self.segment_manager.set_opacity(segment_name, opacity)
        
        # Update the corresponding group slider
        segment = self.segment_manager.get_segment(segment_name)
        if not segment: return
        
        group_name = segment['system']
        if group_name in self.group_opacity_sliders:
            slider = self.group_opacity_sliders[group_name]
            slider.blockSignals(True)
            
            # Check if all other sliders in this group match
            all_match = True
            for seg_name, seg in self.segment_manager.segments.items():
                if seg['system'] == group_name:
                    if int(seg['opacity'] * 100) != value:
                        all_match = False
                        break
            
            if all_match:
                slider.setValue(value)
            else:
                # If they don't match, maybe set slider to a "mixed" state?
                # For now, we'll just reflect the last changed item.
                slider.setValue(value) # Simple update
                
            slider.blockSignals(False)
            
        self.vtk_widget.GetRenderWindow().Render()
        
    # --- NEW: `update_master_opacity` from Dental ---
    def update_master_opacity(self, value):
        opacity = value / 100.0
        self.master_opacity_label.setText(f"{value}%")
        
        for name in self.segment_manager.segments.keys():
            self.segment_manager.set_opacity(name, opacity)
        
        # Update all other sliders to match
        for slider in self.group_opacity_sliders.values():
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        
        # Update all sliders in the tree
        for i in range(self.segment_tree.topLevelItemCount()):
            root_item = self.segment_tree.topLevelItem(i)
            for j in range(root_item.childCount()):
                child_item = root_item.child(j)
                widget = self.segment_tree.itemWidget(child_item, 1)
                if widget:
                    slider = widget.findChild(QSlider)
                    if slider:
                        slider.blockSignals(True)
                        slider.setValue(value)
                        slider.blockSignals(False)
                        
        self.vtk_widget.GetRenderWindow().Render()

    # --- NEW: `update_group_opacity` for new sliders ---
    def update_group_opacity(self, group_name, value):
        """Updates all segments and tree sliders belonging to a specific group."""
        opacity = value / 100.0
        
        # Update all segments in this group
        for name, segment in self.segment_manager.segments.items():
            if segment['system'] == group_name:
                self.segment_manager.set_opacity(name, opacity)
                
        # Update all sliders in the tree for this group
        root_item = None
        for i in range(self.segment_tree.topLevelItemCount()):
            temp_item = self.segment_tree.topLevelItem(i)
            if temp_item.text(0) == group_name:
                root_item = temp_item
                break
        
        if root_item:
            for i in range(root_item.childCount()):
                child_item = root_item.child(i)
                widget = self.segment_tree.itemWidget(child_item, 1)
                if widget:
                    slider = widget.findChild(QSlider)
                    if slider:
                        slider.blockSignals(True)
                        slider.setValue(value)
                        slider.blockSignals(False)
                        
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"{group_name} group opacity set to {value}%")

    # --- REMOVED old opacity functions ---

    def toggle_smooth_shading(self, state):
        for segment in self.segment_manager.segments.values():
            if state == Qt.Checked:
                segment['actor'].GetProperty().SetInterpolationToPhong()
            else:
                segment['actor'].GetProperty().SetInterpolationToFlat()
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_edges(self, state):
        for segment in self.segment_manager.segments.values():
            segment['actor'].GetProperty().SetEdgeVisibility(state == Qt.Checked)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    # ==================== NEW: Clipping Methods ====================
    def open_clipping_dialog(self):
        if self.clipping_dialog is None:
            self.clipping_dialog = ClippingDialog(self)
        self.clipping_dialog.show()
        self.clipping_dialog.raise_()
        self.clipping_dialog.activateWindow()
    
    def apply_advanced_clipping(self, params):
        for actor in self.plane_actors:
            self.renderer.RemoveActor(actor)
        self.plane_actors.clear()
        
        actors = self.segment_manager.get_all_actors()
        if not actors:
            bounds_array = [0, 1, 0, 1, 0, 1]
        else:
            prop_bounds = vtk.vtkBoundingBox()
            for actor in actors:
                prop_bounds.AddBounds(actor.GetBounds())
            
            bounds_array = [0.0] * 6
            prop_bounds.GetBounds(bounds_array)
        
        xmin, xmax, ymin, ymax, zmin, zmax = bounds_array
        
        x_pos = xmin + params['x_pos'] * (xmax - xmin)
        y_pos = ymin + params['y_pos'] * (ymax - ymin)
        z_pos = zmin + params['z_pos'] * (zmax - zmin)
        
        for seg in self.segment_manager.segments.values():
            seg['mapper'].RemoveAllClippingPlanes()
            planes = vtk.vtkPlaneCollection()
            
            if params['hide_left']:
                p = vtk.vtkPlane()
                p.SetOrigin(x_pos, 0, 0)
                p.SetNormal(1, 0, 0)
                planes.AddItem(p)
            
            if params['hide_right']:
                p = vtk.vtkPlane()
                p.SetOrigin(x_pos, 0, 0)
                p.SetNormal(-1, 0, 0)
                planes.AddItem(p)
            
            if params['hide_front']:
                p = vtk.vtkPlane()
                p.SetOrigin(0, y_pos, 0)
                p.SetNormal(0, 1, 0)
                planes.AddItem(p)
            
            if params['hide_back']:
                p = vtk.vtkPlane()
                p.SetOrigin(0, y_pos, 0)
                p.SetNormal(0, -1, 0)
                planes.AddItem(p)
            
            if params['hide_bottom']:
                p = vtk.vtkPlane()
                p.SetOrigin(0, 0, z_pos)
                p.SetNormal(0, 0, 1)
                planes.AddItem(p)
            
            if params['hide_top']:
                p = vtk.vtkPlane()
                p.SetOrigin(0, 0, z_pos)
                p.SetNormal(0, 0, -1)
                planes.AddItem(p)
            
            if planes.GetNumberOfItems() > 0:
                seg['mapper'].SetClippingPlanes(planes)
        
        if params['show_axial']:
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(xmin, ymin, z_pos)
            plane.SetPoint1(xmax, ymin, z_pos)
            plane.SetPoint2(xmin, ymax, z_pos)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.2, 0.5, 1.0)
            actor.GetProperty().SetOpacity(0.4)
            self.renderer.AddActor(actor)
            self.plane_actors.append(actor)
        
        if params['show_sagittal']:
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(x_pos, ymin, zmin)
            plane.SetPoint1(x_pos, ymax, zmin)
            plane.SetPoint2(x_pos, ymin, zmax)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.2, 0.2)
            actor.GetProperty().SetOpacity(0.4)
            self.renderer.AddActor(actor)
            self.plane_actors.append(actor)
        
        if params['show_coronal']:
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(xmin, y_pos, zmin)
            plane.SetPoint1(xmax, y_pos, zmin)
            plane.SetPoint2(xmin, y_pos, zmax)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.2, 1.0, 0.2)
            actor.GetProperty().SetOpacity(0.4)
            self.renderer.AddActor(actor)
            self.plane_actors.append(actor)
        
        self.vtk_widget.GetRenderWindow().Render()

    # ==================== NEW: MPR Methods ====================
    def open_mpr_dialog(self):
        if not HAS_NIBABEL:
            QMessageBox.warning(self, "Missing Libraries", "This feature requires 'nibabel' and 'matplotlib'.\nInstall with: pip install nibabel matplotlib")
            return
            
        if self.mpr_dialog is None:
            self.mpr_dialog = CurvedMPRDialog(self)
        self.mpr_dialog.show()
        self.mpr_dialog.raise_()
        self.mpr_dialog.activateWindow()

    # ==================== Navigation / Reset ====================
    def toggle_animation(self):
        if self.play_btn.isChecked():
            self.animation_timer.start(50)
            self.statusBar().showMessage("Animation started")
        else:
            self.animation_timer.stop()
            self.statusBar().showMessage("Animation stopped")
    
    def update_animation(self):
        speed = self.speed_slider.value() / 100.0
        self.animation_frame += 1
        
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(speed)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def reset_animation(self):
        self.animation_frame = 0
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
    
    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Camera reset")
    
    def reset_all(self):
        reply = QMessageBox.question(
            self, "Reset All",
            "Clear all data and reset?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.animation_timer.stop()
            self.neural_timer.stop()
            self.flight_timer.stop()
            
            self.stop_neural_animation()
            
            # Clear all actors
            for actor in self.segment_manager.get_all_actors():
                self.renderer.RemoveActor(actor)
            
            # Clear clipping plane actors
            for actor in self.plane_actors:
                self.renderer.RemoveActor(actor)
            self.plane_actors.clear()
            
            # Clear segment manager and tree
            self.segment_manager.clear()
            self.segment_tree.clear()
            
            self.play_btn.setChecked(False)
            self.flight_btn.setChecked(False)
            self.focus_nav_btn.setChecked(False)
            
            self.is_flight_mode = False
            self.is_diving = False
            
            if self.focus_navigator.is_active:
                self.focus_navigator.deactivate()
            
            self.data_status_label.setText("No data loaded")
            self.data_status_label.setStyleSheet(f"color: {self.colors['accent_yellow']};")
            self.neural_info_label.setText("Ready")
            
            # --- NEW: Reset all opacity sliders ---
            self.master_opacity_slider.setValue(100)
            for slider in self.group_opacity_sliders.values():
                if slider: # Check if it exists
                    slider.setValue(100)
            # --- END NEW ---
            
            self.model_center = [0, 0, 0]
            
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.statusBar().showMessage("Reset complete")


def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion')) # Use Fusion style
    
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = Brain3DVisualizationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
