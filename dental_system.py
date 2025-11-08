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
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QLocale, QUrl, QFileInfo
from PyQt5.QtGui import QColor, QPalette, QIcon, QFont
from PyQt5.QtMultimedia import QSoundEffect
import vtk
from vtk import vtkMath
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import os
from collections import defaultdict
import time

# --- NEW: Imports from Musculoskeletal Code for MPR ---
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
        return [seg for name, seg in self.segments.items() if group_name in name]
    
    def get_all_actors(self):
        return [seg['actor'] for seg in self.segments.values()]
    
    def clear(self):
        self.segments.clear()
        self.segment_groups.clear()


# ==================== NEURAL SIGNAL ANIMATOR (V4) ====================
class NeuralSignalAnimator:
    """
    Handles neural signal animation on teeth surfaces.
    """
    def __init__(self, renderer):
        self.renderer = renderer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_signal_animation)
        self.animation_step = 0
        self.animation_duration = 75  # ~2.5 seconds (30 fps)
        self.is_animating = False
        self.teeth_data = []
        self.completion_callback = None
        
    def prepare_teeth_for_signal(self, segment_manager):
        """
        Extract ONLY visible segments identified as 'tooth'
        and determine orientation based on 'lower'/'upper' in the name.
        """
        self.teeth_data.clear()

        for name, segment in segment_manager.segments.items():
            name_lower = name.lower()
            
            if 'tooth' in name_lower and segment['visible']:
                mapper = segment['mapper']
                polydata = mapper.GetInput()
                actor = segment['actor']
                
                is_lower = 'lower' in name_lower
                
                original_bounds = [0] * 6
                actor.GetMapper().GetInput().GetBounds(original_bounds)
                z_segment_center = (original_bounds[4] + original_bounds[5]) / 2.0
                
                z_signal_start, z_signal_end = 0, 0
                if is_lower:
                    z_signal_start = original_bounds[5] 
                    z_signal_end = original_bounds[4] 
                else: 
                    z_signal_start = original_bounds[4]
                    z_signal_end = original_bounds[5]

                prop = actor.GetProperty()
                original_color = prop.GetColor()
                original_ambient = prop.GetAmbient()
                original_diffuse = prop.GetDiffuse()
                
                signal_color = (0.7, 0.85, 1.0) 
                
                self.teeth_data.append({
                    'name': name,
                    'polydata': polydata,
                    'z_signal_start': z_signal_start,
                    'z_signal_end': z_signal_end,
                    'actor': actor,
                    'is_lower': is_lower,
                    'z_center': z_segment_center,
                    'original_color': original_color,
                    'original_ambient': original_ambient,
                    'original_diffuse': original_diffuse,
                    'signal_color': signal_color 
                })
        
    def restore_original_properties(self):
        """Restores all animated teeth to their original properties"""
        for tooth_data in self.teeth_data:
            actor = tooth_data.get('actor')
            if actor:
                prop = actor.GetProperty()
                prop.SetColor(*tooth_data['original_color'])
                prop.SetAmbient(tooth_data['original_ambient'])
                prop.SetDiffuse(tooth_data['original_diffuse'])

    def start_signal_animation(self, completion_callback=None):
        """Start the neural signal animation"""
        if self.is_animating:
            return
        
        self.restore_original_properties() 
        
        self.animation_step = 0
        self.is_animating = True
        self.completion_callback = completion_callback
        self.animation_timer.start(33)
    
    def update_signal_animation(self):
        """Update signal animation for current frame using 'whole-actor-flash'"""
        self.animation_step += 1
        progress = self.animation_step / self.animation_duration
        
        if progress >= 1.0:
            self.animation_timer.stop()
            self.is_animating = False
            self.restore_original_properties()
            if self.completion_callback:
                self.completion_callback()
            self.renderer.GetRenderWindow().Render() 
            return
        
        for tooth_data in self.teeth_data:
            actor = tooth_data['actor']
            prop = actor.GetProperty()
            is_lower = tooth_data['is_lower']
            
            z_start = tooth_data['z_signal_start']
            z_end = tooth_data['z_signal_end']
            
            z_range = abs(z_end - z_start)
            if z_range == 0: z_range = 1 
                
            signal_thickness = z_range * 0.20 
            
            current_z_untransformed = 0
            if is_lower:
                current_z_untransformed = z_start - (progress * z_range)
            else:
                current_z_untransformed = z_start + (progress * z_range)
            
            z_center = tooth_data['z_center']
            distance = abs(z_center - current_z_untransformed)
            
            intensity = 0.0
            if distance < signal_thickness:
                intensity = 1.0 - (distance / signal_thickness)
                intensity = intensity ** 0.5 
            
            orig_color = tooth_data['original_color']
            sig_color = tooth_data['signal_color']
            
            r = orig_color[0] * (1 - intensity) + sig_color[0] * intensity
            g = orig_color[1] * (1 - intensity) + sig_color[1] * intensity
            b = orig_color[2] * (1 - intensity) + sig_color[2] * intensity
            
            prop.SetColor(r, g, b)
            prop.SetAmbient(tooth_data['original_ambient'] + intensity * 0.2) 
            prop.SetDiffuse(tooth_data['original_diffuse'] + intensity * 0.1)

        self.renderer.GetRenderWindow().Render()
    
    def stop_animation(self):
        """Stop the animation immediately"""
        self.animation_timer.stop()
        self.is_animating = False
        self.restore_original_properties()
        self.renderer.GetRenderWindow().Render()

# ==================== END OF MODIFIED CLASS ====================


class JawMovementController:
    """Controls independent jaw movement animation"""
    def __init__(self, segment_manager):
        self.segment_manager = segment_manager
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_jaw_animation)
        self.animation_step = 0
        self.animation_duration = 30  # ~1 second (30 steps * 33ms)
        self.is_opening = True
        self.is_animating = False
        self.jaw_open_angle = 15.0  # degrees
        self.upper_jaw_names = []
        self.lower_jaw_names = []
        self.reference_center = [0, 0, 0]
        self.completion_callback = None 
        
    def identify_jaw_parts(self):
        """Identify upper and lower jaw components based on naming convention"""
        self.upper_jaw_names.clear()
        self.lower_jaw_names.clear()
        
        for name in self.segment_manager.segments.keys():
            name_lower = name.lower()
            if 'upper' in name_lower:
                self.upper_jaw_names.append(name)
            elif 'lower' in name_lower:
                self.lower_jaw_names.append(name)
        
        if self.upper_jaw_names or self.lower_jaw_names:
            all_actors = [self.segment_manager.segments[name]['actor'] for name in self.upper_jaw_names + self.lower_jaw_names if name in self.segment_manager.segments]
            if all_actors:
                bounds = vtk.vtkBoundingBox()
                for actor in all_actors:
                    actor_polydata = actor.GetMapper().GetInput()
                    if actor_polydata and actor_polydata.GetNumberOfPoints() > 0:
                        actor_bounds = actor_polydata.GetBounds() 
                        bounds.AddBounds(actor_bounds)
                
                center = [(bounds.GetBound(0) + bounds.GetBound(1)) / 2.0, 
                          (bounds.GetBound(2) + bounds.GetBound(3)) / 2.0, 
                          (bounds.GetBound(4) + bounds.GetBound(5)) / 2.0]
                
                self.reference_center = [center[0], center[1] + 50, center[2]]
        
    def start_jaw_movement(self, opening=True, completion_callback=None):
        """Start jaw opening or closing animation"""
        if self.is_animating:
            return
        
        self.identify_jaw_parts()
        
        if not self.upper_jaw_names and not self.lower_jaw_names:
            print("JawMovementController: No 'upper' or 'lower' segments found to move.")
            return

        self.animation_step = 0
        self.is_opening = opening
        self.is_animating = True
        self.completion_callback = completion_callback 
        self.animation_timer.start(33)
    
    def update_jaw_animation(self):
        """Update jaw animation for current frame"""
        self.animation_step += 1
        progress = self.animation_step / self.animation_duration
        
        if progress >= 1.0:
            progress = 1.0
            self.animation_timer.stop()
            self.is_animating = False
            if self.completion_callback:
                callback = self.completion_callback
                self.completion_callback = None 
                callback() 
        
        progress_smooth = 0.5 - 0.5 * np.cos(progress * np.pi)
        
        if not self.is_opening:
            progress_smooth = 1.0 - progress_smooth
        
        angle = progress_smooth * self.jaw_open_angle
        
        upper_transform = vtk.vtkTransform()
        upper_transform.Translate(self.reference_center[0], self.reference_center[1], self.reference_center[2])
        upper_transform.RotateX(-angle * 0.1) 
        upper_transform.Translate(-self.reference_center[0], -self.reference_center[1], -self.reference_center[2])
        upper_transform.Translate(0, 0, angle * 0.1) 
        
        lower_transform = vtk.vtkTransform()
        lower_transform.Translate(self.reference_center[0], self.reference_center[1], self.reference_center[2])
        lower_transform.RotateX(angle) 
        lower_transform.Translate(-self.reference_center[0], -self.reference_center[1], -self.reference_center[2])
        lower_transform.Translate(0, 0, -angle * 1.5) 
        
        
        for name in self.upper_jaw_names:
            if name in self.segment_manager.segments:
                segment = self.segment_manager.segments[name]
                segment['actor'].SetUserTransform(upper_transform)
        
        for name in self.lower_jaw_names:
            if name in self.segment_manager.segments:
                segment = self.segment_manager.segments[name]
                segment['actor'].SetUserTransform(lower_transform)
        
        if self.segment_manager.segments:
            list(self.segment_manager.segments.values())[0]['actor'].GetMapper().GetInput().Modified()
        
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()
    
    def reset_jaw_position(self):
        """Reset jaw to original position"""
        self.identify_jaw_parts() 
        for name in self.upper_jaw_names + self.lower_jaw_names:
            if name in self.segment_manager.segments:
                segment = self.segment_manager.segments[name]
                segment['actor'].SetUserTransform(None)
        
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()


# --- NEW: Ported Advanced Clipping Dialog from Musculoskeletal Code ---
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
# --- END: Ported Clipping Dialog ---


# --- NEW: Ported CurvedMPRDialog from Musculoskeletal Code ---
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

        # --- NEW: Slice Selection Controls ---
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
        
        # --- NEW: Get slice range from UI ---
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

# --- END: Ported CurvedMPRDialog ---


# --- REMOVED: CurvedMPRGenerator class is gone ---


class FocusNavigator:
    """Handles focus navigation with glow effect"""
    def __init__(self, segment_manager):
        self.segment_manager = segment_manager
        self.original_properties = {}
        self.is_active = False
        
    def activate(self):
        """Store original properties when activating focus mode"""
        self.is_active = True
        self.original_properties.clear()
        for name, segment in self.segment_manager.segments.items():
            prop = segment['actor'].GetProperty()
            self.original_properties[name] = {
                'opacity': prop.GetOpacity(),
                'ambient': prop.GetAmbient(),
            }
    
    def deactivate(self):
        """Restore original properties when deactivating"""
        self.is_active = False
        for name, props in self.original_properties.items():
            if name in self.segment_manager.segments:
                segment = self.segment_manager.segments[name]
                segment['actor'].GetProperty().SetOpacity(props['opacity'])
                segment['actor'].GetProperty().SetAmbient(props['ambient'])
        self.original_properties.clear()
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()
    
    def focus_on_segment(self, target_segment_name):
        """Focus on target segment with glow, dim others"""
        if not self.is_active:
            return
        
        for name, segment in self.segment_manager.segments.items():
            prop = segment['actor'].GetProperty()
            if name == target_segment_name:
                prop.SetOpacity(1.0)
                prop.SetAmbient(0.8)
            else:
                prop.SetOpacity(0.2)
                original_ambient = self.original_properties.get(name, {}).get('ambient', 0.2)
                prop.SetAmbient(original_ambient)
        
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()

# ==================== MAIN GUI CLASS (MODIFIED) ====================

class Dental3DVisualizationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦷 Advanced Dental 3D Visualization System")
        self.setGeometry(50, 50, 1600, 1000)
        
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
        
        self.apply_stylesheet()
        
        self.segment_manager = SegmentManager()
        # --- MODIFIED: MPR Generator removed
        self.mpr_dialog = None
        self.focus_navigator = FocusNavigator(self.segment_manager)
        
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
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_rotation_animation)
        self.animation_frame = 0
        
        self.is_picking_points = False # This is no longer used by a button, but kept for safety
        self.picker = None
        self.saved_camera_views = {}
        self.camera_angle = 0
        
        self.standard_plane_actors = {}
        self.model_center = [0, 0, 0]
        
        self.clipping_dialog = None
        self.plane_actors = []
        
        self.dental_colors = [
            (1.0, 1.0, 0.9), (0.95, 0.95, 0.85), (0.9, 0.9, 0.8), (0.85, 0.85, 0.75), 
            (1.0, 0.9, 0.8), (0.95, 0.85, 0.75), (0.9, 0.8, 0.7), (1.0, 0.7, 0.7)
        ]
        
        self.neural_signal_animator = None
        self.jaw_movement_controller = None
        self.jaw_is_open = False
        
        self.init_ui()
        
        # --- MODIFIED: Initialize Sound with "teeth.wav" ---
        self.jaw_close_sound = QSoundEffect()
        sound_file = "teeth.wav" # --- MODIFIED: Filename changed ---
        if os.path.exists(sound_file):
            self.jaw_close_sound.setSource(QUrl.fromLocalFile(os.path.abspath(sound_file)))
            self.jaw_close_sound.setVolume(1.0)
        else:
            print(f"Warning: Sound file '{sound_file}' not found. Place it in the script directory.")
        
    def apply_stylesheet(self):
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
            QPushButton:disabled {{
                background-color: #555555;
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
        """)
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.vtk_widget.setMinimumSize(800, 600)
        splitter.addWidget(self.vtk_widget)
        
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([350, 900, 350])
        main_layout.addWidget(splitter)
        
        self.setup_vtk()
        self.focus_navigator.vtk_widget = self.vtk_widget
        self.statusBar().showMessage("Ready - Load dental models | Neural Signal & Jaw Animation Available")
        
    def setup_vtk(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.18)
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground2(0.2, 0.1, 0.3)
        
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        light1 = vtk.vtkLight()
        light1.SetPosition(100, 100, 100)
        light1.SetIntensity(1.0)
        light1.SetColor(1, 1, 1)
        self.renderer.AddLight(light1)
        
        light2 = vtk.vtkLight()
        light2.SetPosition(-100, -100, 100)
        light2.SetColor(0.6, 0.8, 1.0)
        light2.SetIntensity(0.6)
        self.renderer.AddLight(light2)
        
        light3 = vtk.vtkLight()
        light3.SetPosition(0, -100, 0)
        light3.SetColor(1.0, 0.9, 0.8)
        light3.SetIntensity(0.4)
        self.renderer.AddLight(light3)
        
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.picker.PickFromListOn()
        self.picker.InitializePickList()
        self.interactor.SetPicker(self.picker)
        
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_click, 1.0)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self.on_left_up, 1.0)
        
        self.neural_signal_animator = NeuralSignalAnimator(self.renderer)
        
        self.jaw_movement_controller = JawMovementController(self.segment_manager)
        self.jaw_movement_controller.vtk_widget = self.vtk_widget
        
        self.interactor.Initialize()
        
    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        title = QLabel("🦷 Dental 3D Viewer")
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {self.colors['accent_cyan']}; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout()
        
        load_segment_btn = QPushButton("📁 Load Single Segment")
        load_segment_btn.clicked.connect(self.load_segment_file)
        data_layout.addWidget(load_segment_btn)
        
        load_folder_btn = QPushButton("📂 Load Folder (Multiple Segments)")
        load_folder_btn.clicked.connect(self.load_folder_segments)
        data_layout.addWidget(load_folder_btn)
        
        load_demo_btn = QPushButton("🦷 Load Demo Dental Model")
        load_demo_btn.clicked.connect(self.load_demo_dental)
        data_layout.addWidget(load_demo_btn)
        
        reset_btn = QPushButton("🔄 RESET - Clear Model")
        reset_btn.setStyleSheet(f"background-color: {self.colors['accent_orange']}; font-size: 12px; padding: 10px;")
        reset_btn.clicked.connect(self.reset_current_model)
        data_layout.addWidget(reset_btn)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        segment_group = QGroupBox("Loaded Segments")
        segment_layout = QVBoxLayout()
        
        self.segment_tree = QTreeWidget()
        self.segment_tree.setHeaderLabels(["Segment", "Opacity"])
        self.segment_tree.setColumnWidth(0, 150)
        
        self.segment_tree.itemChanged.connect(self.on_segment_tree_changed)
        self.segment_tree.itemClicked.connect(self.on_segment_clicked)
        segment_layout.addWidget(self.segment_tree)
        
        segment_group.setLayout(segment_layout)
        layout.addWidget(segment_group)
        
        layout.addStretch()
        return panel
        
    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        tabs = QTabWidget()
        
        tabs.addTab(self.create_visualization_tab(), "🎨 Rendering")
        tabs.addTab(self.create_clipping_tab(), "✂️ Clipping")
        tabs.addTab(self.create_mpr_tab(), "📐 Curved MPR")
        tabs.addTab(self.create_animation_tab(), "⚡ Neural & Jaw")
        tabs.addTab(self.create_navigation_tab(), "🧭 Navigation")
        
        layout.addWidget(tabs)
        return panel

    def create_jaw_opacity_group(self):
        """Method for independent Upper/Lower jaw opacity control (Name-based)"""
        jaw_opacity_group = QGroupBox("Independent Jaw Opacity")
        jaw_opacity_layout = QVBoxLayout()
        
        # Upper Jaw Control
        upper_jaw_label = QLabel("Upper Jaw Opacity:")
        jaw_opacity_layout.addWidget(upper_jaw_label)
        self.upper_opacity_slider = QSlider(Qt.Horizontal)
        self.upper_opacity_slider.setMinimum(0)
        self.upper_opacity_slider.setMaximum(100)
        self.upper_opacity_slider.setValue(100)
        self.upper_opacity_slider.valueChanged.connect(lambda val: self.update_group_opacity("Upper", val))
        jaw_opacity_layout.addWidget(self.upper_opacity_slider)
        
        # Lower Jaw Control
        lower_jaw_label = QLabel("Lower Jaw Opacity:")
        jaw_opacity_layout.addWidget(lower_jaw_label)
        self.lower_opacity_slider = QSlider(Qt.Horizontal)
        self.lower_opacity_slider.setMinimum(0)
        self.lower_opacity_slider.setMaximum(100)
        self.lower_opacity_slider.setValue(100)
        self.lower_opacity_slider.valueChanged.connect(lambda val: self.update_group_opacity("Lower", val))
        jaw_opacity_layout.addWidget(self.lower_opacity_slider)
        
        jaw_opacity_group.setLayout(jaw_opacity_layout)
        return jaw_opacity_group
        
    def create_visualization_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
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
        layout.addWidget(master_group)
        
        layout.addWidget(self.create_jaw_opacity_group())
        
        quality_group = QGroupBox("Rendering Quality")
        quality_layout = QVBoxLayout()
        
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low", "Medium", "High", "Ultra"])
        self.quality_combo.setCurrentIndex(2)
        self.quality_combo.currentTextChanged.connect(self.update_quality)
        quality_layout.addWidget(self.quality_combo)
        
        self.smooth_checkbox = QCheckBox("Smooth Shading")
        self.smooth_checkbox.setChecked(True)
        self.smooth_checkbox.stateChanged.connect(self.toggle_smooth_shading)
        quality_layout.addWidget(self.smooth_checkbox)
        
        self.edge_checkbox = QCheckBox("Show Edges")
        self.edge_checkbox.stateChanged.connect(self.toggle_edges)
        quality_layout.addWidget(self.edge_checkbox)
        
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        color_group = QGroupBox("Color Presets")
        color_layout = QGridLayout()
        
        natural_btn = QPushButton("Natural White")
        natural_btn.setStyleSheet(f"background-color: #FFFEF5;")
        natural_btn.clicked.connect(lambda: self.apply_dental_colors())
        color_layout.addWidget(natural_btn, 0, 0)
        
        enamel_btn = QPushButton("Enamel")
        enamel_btn.setStyleSheet(f"background-color: #F5F5DC;")
        enamel_btn.clicked.connect(lambda: self.apply_single_color((1.0, 1.0, 0.9)))
        color_layout.addWidget(enamel_btn, 0, 1)
        
        dentin_btn = QPushButton("Dentin")
        dentin_btn.setStyleSheet(f"background-color: #D9D9B8;")
        dentin_btn.clicked.connect(lambda: self.apply_single_color((0.85, 0.85, 0.72)))
        color_layout.addWidget(dentin_btn, 1, 0)
        
        bone_btn = QPushButton("Bone")
        bone_btn.setStyleSheet(f"background-color: #E6E6C8;")
        bone_btn.clicked.connect(lambda: self.apply_single_color((0.9, 0.9, 0.78)))
        color_layout.addWidget(bone_btn, 1, 1)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        layout.addStretch()
        return tab

    def create_animation_tab(self):
        """Neural Signal and Jaw Movement"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        neural_group = QGroupBox("⚡ Neural Signal Animation")
        neural_layout = QVBoxLayout()
        
        info_neural = QLabel("Simulates nerve activation from roots to crown tips (Relies on 'tooth' and 'lower'/'upper' in names)")
        info_neural.setWordWrap(True)
        info_neural.setStyleSheet(f"color: {self.colors['accent_cyan']}; font-size: 10px; padding: 5px;")
        neural_layout.addWidget(info_neural)
        
        self.neural_signal_btn = QPushButton("⚡ Start Neural Signal")
        self.neural_signal_btn.clicked.connect(self.trigger_neural_signal)
        neural_layout.addWidget(self.neural_signal_btn)
        
        neural_group.setLayout(neural_layout)
        layout.addWidget(neural_group)
        
        jaw_group = QGroupBox("🦴 Jaw Movement Control")
        jaw_layout = QVBoxLayout()
        
        info_jaw = QLabel("Independent control for upper and lower jaw movement (Relies on 'lower'/'upper' in names)")
        info_jaw.setWordWrap(True)
        info_jaw.setStyleSheet(f"color: {self.colors['accent_cyan']}; font-size: 10px; padding: 5px;")
        jaw_layout.addWidget(info_jaw)
        
        self.jaw_cycle_btn = QPushButton("🦴 Start Jaw Cycle (Open/Close)")
        self.jaw_cycle_btn.clicked.connect(self.trigger_automatic_jaw_cycle)
        jaw_layout.addWidget(self.jaw_cycle_btn)
        
        self.jaw_status_label = QLabel("Status: Closed")
        self.jaw_status_label.setAlignment(Qt.AlignCenter)
        self.jaw_status_label.setStyleSheet(f"color: {self.colors['accent_green']}; font-size: 11px; font-weight: bold;")
        jaw_layout.addWidget(self.jaw_status_label)
        
        jaw_layout.addWidget(QLabel("Jaw Opening Angle:"))
        self.jaw_angle_slider = QSlider(Qt.Horizontal)
        self.jaw_angle_slider.setMinimum(5)
        self.jaw_angle_slider.setMaximum(30)
        self.jaw_angle_slider.setValue(15)
        self.jaw_angle_slider.valueChanged.connect(self.update_jaw_angle)
        jaw_layout.addWidget(self.jaw_angle_slider)
        
        self.jaw_angle_label = QLabel("15°")
        self.jaw_angle_label.setAlignment(Qt.AlignCenter)
        jaw_layout.addWidget(self.jaw_angle_label)
        
        jaw_group.setLayout(jaw_layout)
        layout.addWidget(jaw_group)
        
        combined_group = QGroupBox("🎬 Combined Animation")
        combined_layout = QVBoxLayout()
        
        info_combined = QLabel("Runs full sequence: Signal -> Open -> Signal -> Close")
        info_combined.setWordWrap(True)
        info_combined.setStyleSheet(f"color: {self.colors['accent_yellow']}; font-size: 10px; padding: 5px;")
        combined_layout.addWidget(info_combined)
        
        self.combined_anim_btn = QPushButton("🎬 Start Full Sequence")
        self.combined_anim_btn.clicked.connect(self.trigger_combined_animation)
        combined_layout.addWidget(self.combined_anim_btn)
        
        combined_group.setLayout(combined_layout)
        layout.addWidget(combined_group)
        
        layout.addStretch()
        return tab
    
    # --- MODIFIED: Clipping tab now uses the Advanced Dialog ---
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
    
    # --- MODIFIED: MPR tab now uses the Volume-based Dialog ---
    def create_mpr_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info = QLabel("Curved Multi-Planar Reconstruction:\nDraw curved paths through VOLUMES (NIfTI)")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {self.colors['accent_cyan']}; padding: 10px;")
        layout.addWidget(info)
        
        open_btn = QPushButton("📐 Open MPR Tool (For Volumes)")
        open_btn.setStyleSheet(f"background-color: {self.colors['accent_green']}; font-size: 14px; padding: 12px;")
        open_btn.clicked.connect(self.open_mpr_dialog)
        layout.addWidget(open_btn)
        
        layout.addStretch()
        return tab
        
    def create_navigation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        flight_group = QGroupBox("✈️ Guided Tour (Fly-Through)")
        flight_layout = QVBoxLayout()
        
        info_flight = QLabel("Fly through the jaw arch in a 'funnel' view.")
        info_flight.setWordWrap(True)
        info_flight.setStyleSheet(f"color: {self.colors['text_light']}; font-size: 10px;")
        flight_layout.addWidget(info_flight)
        
        self.flight_btn = QPushButton("🚀 Start Guided Tour")
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
        self.play_btn.clicked.connect(self.toggle_rotation_animation)
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
        
        rotation_group = QGroupBox("Precise Rotation Control")
        rotation_layout = QVBoxLayout()
        
        rotation_layout.addWidget(QLabel("X-Axis Rotation (degrees):"))
        self.rotation_x = QDoubleSpinBox()
        self.rotation_x.setMinimum(-360)
        self.rotation_x.setMaximum(360)
        self.rotation_x.setSingleStep(5)
        self.rotation_x.valueChanged.connect(self.apply_precise_rotation)
        rotation_layout.addWidget(self.rotation_x)
        
        rotation_layout.addWidget(QLabel("Y-Axis Rotation (degrees):"))
        self.rotation_y = QDoubleSpinBox()
        self.rotation_y.setMinimum(-360)
        self.rotation_y.setMaximum(360)
        self.rotation_y.setSingleStep(5)
        self.rotation_y.valueChanged.connect(self.apply_precise_rotation)
        rotation_layout.addWidget(self.rotation_y)
        
        rotation_layout.addWidget(QLabel("Z-Axis Rotation (degrees):"))
        self.rotation_z = QDoubleSpinBox()
        self.rotation_z.setMinimum(-360)
        self.rotation_z.setMaximum(360)
        self.rotation_z.setSingleStep(5)
        self.rotation_z.valueChanged.connect(self.apply_precise_rotation)
        rotation_layout.addWidget(self.rotation_z)
        
        reset_rotation_btn = QPushButton("Reset Rotation")
        reset_rotation_btn.clicked.connect(self.reset_rotation)
        rotation_layout.addWidget(reset_rotation_btn)
        
        rotation_group.setLayout(rotation_layout)
        layout.addWidget(rotation_group)
        
        preset_group = QGroupBox("Camera Presets")
        preset_layout = QVBoxLayout()
        
        save_view_btn = QPushButton("💾 Save Current View")
        save_view_btn.clicked.connect(self.save_camera_view)
        preset_layout.addWidget(save_view_btn)
        
        load_view_btn = QPushButton("📥 Load Saved View")
        load_view_btn.clicked.connect(self.load_camera_view)
        preset_layout.addWidget(load_view_btn)
        
        reset_camera_btn = QPushButton("🎥 Reset Camera")
        reset_camera_btn.clicked.connect(self.reset_camera)
        preset_layout.addWidget(reset_camera_btn)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        layout.addStretch()
        return tab
    
    def set_animation_buttons_enabled(self, enabled):
        """Helper function to enable/disable animation buttons"""
        self.neural_signal_btn.setEnabled(enabled)
        self.jaw_cycle_btn.setEnabled(enabled)
        self.combined_anim_btn.setEnabled(enabled)

    # ==================== NEURAL SIGNAL & JAW MOVEMENT ====================
    
    # --- MODIFIED: Added sound ---
    def trigger_neural_signal(self):
        """Start neural signal animation on all segments named 'tooth'"""
        if self.neural_signal_animator.is_animating or self.jaw_movement_controller.is_animating:
            self.statusBar().showMessage("Animation already in progress...")
            return
        
        self.neural_signal_animator.prepare_teeth_for_signal(self.segment_manager)
        
        if not self.neural_signal_animator.teeth_data:
                 QMessageBox.warning(self, "No Visible Teeth", "No visible 'tooth' segments found. Make sure segments are loaded, named correctly ('...tooth...'), and checked in the list.")
                 return
        
        # --- NEW: PLAY SOUND ---
        if self.jaw_close_sound.isLoaded():
            self.jaw_close_sound.play()
        # ---------------------
                 
        self.set_animation_buttons_enabled(False)
        self.statusBar().showMessage("Neural signal propagating from roots to crowns...")

        def on_signal_complete():
            self.statusBar().showMessage("Signal complete.")
            self.set_animation_buttons_enabled(True) 
        
        self.neural_signal_animator.start_signal_animation(completion_callback=on_signal_complete)

    # --- MODIFIED: Added sound on Open ---
    def trigger_automatic_jaw_cycle(self):
        """Triggers a full Open -> Close jaw animation cycle."""
        if self.neural_signal_animator.is_animating or self.jaw_movement_controller.is_animating:
            self.statusBar().showMessage("Animation already in progress...")
            return
            
        self.jaw_movement_controller.identify_jaw_parts()
        if not self.jaw_movement_controller.upper_jaw_names and not self.jaw_movement_controller.lower_jaw_names:
            QMessageBox.warning(self, "No Jaw Parts Found", "Jaw movement requires segments with 'Upper' or 'Lower' in their names.")
            return

        self.set_animation_buttons_enabled(False)
        
        def on_jaw_close_complete():
            self.jaw_status_label.setText("Status: Closed")
            self.jaw_is_open = False
            self.statusBar().showMessage("Jaw cycle complete.")
            self.set_animation_buttons_enabled(True)
            
        def on_jaw_open_complete():
            self.jaw_status_label.setText("Status: Open")
            self.jaw_is_open = True
            self.statusBar().showMessage("Jaw open. Closing...")
            
            if self.jaw_close_sound.isLoaded():
                self.jaw_close_sound.play()

            self.jaw_movement_controller.start_jaw_movement(opening=False, completion_callback=on_jaw_close_complete)

        # --- NEW: PLAY SOUND ON OPEN ---
        if self.jaw_close_sound.isLoaded():
            self.jaw_close_sound.play()
        # -----------------------------

        self.jaw_status_label.setText("Status: Opening...")
        self.statusBar().showMessage("Starting jaw cycle...")
        self.jaw_movement_controller.start_jaw_movement(opening=True, completion_callback=on_jaw_open_complete)
    
    # --- MODIFIED: Added sound on Open ---
    def trigger_combined_animation(self):
        """Start neural signal -> open -> signal -> close (FIXED)"""
        if self.neural_signal_animator.is_animating or self.jaw_movement_controller.is_animating:
            self.statusBar().showMessage("Animation already in progress...")
            return
        
        self.jaw_movement_controller.identify_jaw_parts()
        
        if not self.jaw_movement_controller.upper_jaw_names and not self.jaw_movement_controller.lower_jaw_names:
            QMessageBox.warning(self, "No Jaw Parts Found", "Jaw movement requires segments with 'Upper' or 'Lower' in their names.")
            return

        self.neural_signal_animator.prepare_teeth_for_signal(self.segment_manager)
        if not self.neural_signal_animator.teeth_data:
                 QMessageBox.warning(self, "No Visible Teeth", "No visible 'tooth' segments found for neural animation.")
                 return
        
        self.set_animation_buttons_enabled(False)
        self.statusBar().showMessage("Sequence Step 1/4: Starting Signal 1...")

        def on_jaw_close_complete():
            self.jaw_status_label.setText("Status: Closed")
            self.jaw_is_open = False
            self.statusBar().showMessage("Full sequence complete.")
            self.set_animation_buttons_enabled(True)

        def on_signal_2_complete():
            self.statusBar().showMessage("Sequence Step 4/4: Closing Jaw...")
            self.jaw_status_label.setText("Status: Closing...")
            
            if self.jaw_close_sound.isLoaded():
                self.jaw_close_sound.play()
            
            self.jaw_movement_controller.start_jaw_movement(opening=False, completion_callback=on_jaw_close_complete)

        def on_jaw_open_complete():
            self.jaw_status_label.setText("Status: Open")
            self.jaw_is_open = True
            self.statusBar().showMessage("Sequence Step 3/4: Starting Signal 2...")
            
            if self.jaw_close_sound.isLoaded():
                self.jaw_close_sound.play()
                
            self.neural_signal_animator.start_signal_animation(completion_callback=on_signal_2_complete)
        
        def on_signal_1_complete():
            self.statusBar().showMessage("Sequence Step 2/4: Opening Jaw...")
            self.jaw_status_label.setText("Status: Opening...")
            
            # --- NEW: PLAY SOUND ON OPEN ---
            if self.jaw_close_sound.isLoaded():
                self.jaw_close_sound.play()
            # -----------------------------

            self.jaw_movement_controller.start_jaw_movement(opening=True, completion_callback=on_jaw_open_complete)
        
        # --- NEW: PLAY SOUND ON NERVE 1 ---
        if self.jaw_close_sound.isLoaded():
            self.jaw_close_sound.play()
        # ----------------------------------
        
        self.neural_signal_animator.start_signal_animation(completion_callback=on_signal_1_complete)

    
    def update_jaw_angle(self, value):
        """Update jaw opening angle (This function was missing)"""
        if self.jaw_movement_controller:
            self.jaw_movement_controller.jaw_open_angle = value
            self.jaw_angle_label.setText(f"{value}°")
    
    def reset_jaw_position(self):
        """Reset jaw to closed position (FIXED)"""
        self.jaw_movement_controller.animation_timer.stop()
        self.jaw_movement_controller.is_animating = False
        
        self.jaw_movement_controller.reset_jaw_position()
        self.jaw_is_open = False
        self.jaw_status_label.setText("Status: Closed")
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Jaw position reset")

    
    # ==================== EVENT HANDLERS (MODIFIED) ====================
    
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
                self.focus_navigator.focus_on_segment(segment_name)
                self.vtk_widget.GetRenderWindow().Render()
            
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
            
            # --- MODIFIED: Removed MPR picking logic ---
            
            if self.focus_navigator.is_active:
                if segment_name:
                    self.focus_navigator.focus_on_segment(segment_name)
                    self.start_focus_flight(target_point, target_normal)
                    self.statusBar().showMessage(f"Focused on: {segment_name}")
                    handled = True
        
        if not handled:
            self.interactor.GetInteractorStyle().OnLeftButtonDown()
    
    def on_left_up(self, obj, event):
        self.interactor.GetInteractorStyle().OnLeftButtonUp()
    
    # ==================== MODEL CENTER ====================
    
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
    
    # ==================== FLYING CAMERA METHODS (MODIFIED) ====================
    
    def toggle_flight_mode(self, checked):
        self.is_flight_mode = checked
        if self.is_flight_mode:
            # Stop other modes
            # self.is_picking_points = False # No longer relevant
            # self.start_mpr_btn.setChecked(False) # No longer relevant
            if self.focus_navigator.is_active:
                self.focus_navigator.deactivate()
                self.focus_nav_btn.setChecked(False)
            
            self.flight_btn.setText("⏹️ Stop Tour")
            self.statusBar().showMessage("Starting guided tour...")
            
            self.is_diving = True 
            for segment in self.segment_manager.segments.values():
                segment['mapper'].SetClippingPlanes(self.flight_plane_collection)

            self.setup_dental_tour_path()
            
            self.flight_step = 0
            self.flight_duration = self.flight_speed_slider.value() * 3 
            self.flight_timer.start(33) # ~30 FPS
        else:
            self.is_diving = False
            self.flight_timer.stop()
            self.flight_btn.setText("🚀 Start Guided Tour")
            self.statusBar().showMessage("Guided tour stopped")
            for segment in self.segment_manager.segments.values():
                segment['mapper'].SetClippingPlanes(self.empty_clip_planes)
            self.vtk_widget.GetRenderWindow().Render()

    def setup_dental_tour_path(self):
        """Creates the camera keyframes for the dental fly-through."""
        camera = self.renderer.GetActiveCamera()
        self.flight_interpolator.Initialize()
        self.flight_interpolator.SetInterpolationTypeToSpline()

        path = [
            (0.1, [60, 0, 0], [40, 0, 0]),     
            (0.3, [40, 20, 0], [20, 30, 0]),  
            (0.5, [0, 40, 0], [-20, 35, 0]),  
            (0.7, [-20, 35, 0], [-40, 20, 0]), 
            (0.9, [-40, 20, 0], [-60, 0, 0]), 
            (1.0, [-60, 0, 0], [-40, 0, 0])   
        ]
        
        start_cam = vtk.vtkCamera()
        start_cam.DeepCopy(camera)
        self.flight_interpolator.AddCamera(0.0, start_cam)

        for (time, pos, fp) in path:
            key_cam = vtk.vtkCamera()
            key_cam.SetPosition(pos)
            key_cam.SetFocalPoint(fp)
            key_cam.SetViewUp(0, 0, 1) # Z-up for dental model
            self.flight_interpolator.AddCamera(time, key_cam)
    
    def start_focus_flight(self, target_point, target_normal):
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
                self.is_flight_mode = False
                self.flight_btn.setChecked(False)
                self.flight_btn.setText("🚀 Start Guided Tour")
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
            # self.is_picking_points = False # No longer relevant
            self.flight_btn.setChecked(False)
            # self.start_mpr_btn.setChecked(False) # No longer relevant

            self.focus_navigator.activate()
            self.focus_nav_btn.setText("🔴 Focus Mode ON")
            self.statusBar().showMessage("Click segments to focus")
        else:
            self.focus_navigator.deactivate()
            self.focus_nav_btn.setText("🎯 Focus Navigation")
            self.statusBar().showMessage("Focus mode disabled")
    
    # ==================== ROTATION ANIMATION ====================
    
    def toggle_rotation_animation(self):
        if self.play_btn.isChecked():
            self.animation_timer.start(50)
            self.statusBar().showMessage("Rotation animation started")
        else:
            self.animation_timer.stop()
            self.statusBar().showMessage("Rotation animation stopped")
    
    def update_rotation_animation(self):
        speed = self.speed_slider.value() / 100.0
        self.animation_frame += 1
        
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(speed)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def reset_animation(self):
        self.animation_frame = 0
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
    
    # ==================== DATA LOADING (MODIFIED) ====================
    
    def load_segment_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Segment", "",
            "3D Files (*.stl *.obj *.ply *.vtk);;All Files (*)"
        )
        if file_path:
            filename = os.path.basename(file_path)
            segment_name = os.path.splitext(filename)[0]
            
            self.load_segment(file_path, segment_name, "Dental")
            self.update_model_center()
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
    def load_folder_segments(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with 3D Models")
        if folder_path:
            files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.stl', '.obj', '.ply', '.vtk'))]
            
            if not files:
                QMessageBox.warning(self, "No Files", "No 3D model files found in folder")
                return
            
            for i, filename in enumerate(files):
                file_path = os.path.join(folder_path, filename)
                segment_name = os.path.splitext(filename)[0]
                color = self.dental_colors[i % len(self.dental_colors)]
                
                self.load_segment(file_path, segment_name, "Dental", color)
            
            self.update_model_center()
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.statusBar().showMessage(f"Loaded {len(files)} segments from folder")
            
    def load_segment(self, file_path, segment_name, system, color=None):
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
        
        if color is None:
            color = (1.0, 1.0, 0.9)
        
        self.segment_manager.add_segment(segment_name, actor, mapper, reader, system, color)
        self.renderer.AddActor(actor)
        self.picker.AddPickList(actor)
        
        self.add_segment_to_tree(segment_name, system)
        
        self.vtk_widget.GetRenderWindow().Render()
        
    def add_segment_to_tree(self, segment_name, system):
        """Add segment under a hierarchical group (Upper/Lower Jaw) and set up opacity slider."""
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
        
        root_name = ""
        name_lower = segment_name.lower()
        if 'upper' in name_lower:
            root_name = "Upper Jaw (Maxilla)"
        elif 'lower' in name_lower:
            root_name = "Lower Jaw (Mandible)"
        else:
            root_name = "Other Segments"

        root_item = None
        for i in range(self.segment_tree.topLevelItemCount()):
            temp_item = self.segment_tree.topLevelItem(i)
            if temp_item.text(0) == root_name:
                root_item = temp_item
                break
        
        if root_item is None:
            root_item = QTreeWidgetItem([root_name, "Group"])
            root_item.setCheckState(0, Qt.Checked)
            self.segment_tree.addTopLevelItem(root_item)
            
        root_item.addChild(item)
        self.segment_tree.setItemWidget(item, 1, opacity_widget)
        root_item.setExpanded(True)
            
    def load_demo_dental(self):
        """Load procedural demo dental model with separated upper and lower jaws"""
        self.reset_current_model()
        
        # Upper jaw teeth (16 teeth)
        for i in range(16):
            angle = (i / 16.0) * np.pi
            
            if i < 4 or i > 11: height, radius_top, radius_bottom = 20, 3, 4
            elif i < 8 or i > 7: height, radius_top, radius_bottom = 18, 4, 5
            else: height, radius_top, radius_bottom = 16, 5, 6
            
            x = 40 * np.cos(angle)
            y = 40 * np.sin(angle)
            z_crown = 10 
            
            crown = vtk.vtkConeSource()
            crown.SetHeight(height)
            crown.SetRadius(radius_top)
            crown.SetResolution(20)
            crown.SetDirection(0, 0, 1) 
            crown.SetCenter(x, y, z_crown + height/2.0) 
            
            root = vtk.vtkCylinderSource()
            root.SetHeight(height * 0.8)
            root.SetRadius(radius_bottom * 0.7)
            root.SetResolution(20)
            root.SetCenter(x, y, z_crown - height * 0.4) 

            append_filter = vtk.vtkAppendPolyData()
            append_filter.AddInputConnection(crown.GetOutputPort())
            append_filter.AddInputConnection(root.GetOutputPort())
            append_filter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(append_filter.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetInterpolationToPhong()
            prop.SetSpecular(0.7)
            prop.SetSpecularPower(40)
            
            tooth_name = f"Upper_Tooth_{i+1}"
            color_idx = i % len(self.dental_colors)
            
            self.segment_manager.add_segment(
                tooth_name, actor, mapper, None, "Dental", self.dental_colors[color_idx]
            )
            self.renderer.AddActor(actor)
            self.picker.AddPickList(actor)
            self.add_segment_to_tree(tooth_name, "Dental")
            
        # Lower jaw teeth (16 teeth)
        for i in range(16):
            angle = (i / 16.0) * np.pi
            
            if i < 4 or i > 11: height, radius_top, radius_bottom = 18, 3, 4
            elif i < 8 or i > 7: height, radius_top, radius_bottom = 16, 4, 5
            else: height, radius_top, radius_bottom = 15, 5, 6
            
            x = 38 * np.cos(angle)
            y = 38 * np.sin(angle)
            z_crown = -10 
            
            crown = vtk.vtkConeSource()
            crown.SetHeight(height)
            crown.SetRadius(radius_top)
            crown.SetResolution(20)
            crown.SetDirection(0, 0, -1) 
            crown.SetCenter(x, y, z_crown - height/2.0)
            
            root = vtk.vtkCylinderSource()
            root.SetHeight(height * 0.8)
            root.SetRadius(radius_bottom * 0.7)
            root.SetResolution(20)
            root.SetCenter(x, y, z_crown + height * 0.4)

            append_filter = vtk.vtkAppendPolyData()
            append_filter.AddInputConnection(crown.GetOutputPort())
            append_filter.AddInputConnection(root.GetOutputPort())
            append_filter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(append_filter.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetInterpolationToPhong()
            prop.SetSpecular(0.7)
            prop.SetSpecularPower(40)
            
            tooth_name = f"Lower_Tooth_{i+1}"
            color_idx = i % len(self.dental_colors)
            
            self.segment_manager.add_segment(
                tooth_name, actor, mapper, None, "Dental", self.dental_colors[color_idx]
            )
            self.renderer.AddActor(actor)
            self.picker.AddPickList(actor)
            self.add_segment_to_tree(tooth_name, "Dental")
            
        # Upper jaw bone
        upper_jaw = vtk.vtkCubeSource()
        upper_jaw.SetXLength(90)
        upper_jaw.SetYLength(90)
        upper_jaw.SetZLength(15)
        upper_jaw.SetCenter(0, 0, 22) 
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(upper_jaw.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
        prop.SetSpecular(0.3)
        prop.SetSpecularPower(20)
        prop.SetOpacity(0.4)
        
        self.segment_manager.add_segment(
            "Upper_Jaw_Bone", actor, mapper, None, "Dental", (0.9, 0.85, 0.75)
        )
        self.renderer.AddActor(actor)
        self.picker.AddPickList(actor)
        self.add_segment_to_tree("Upper_Jaw_Bone", "Dental")
        
        # Lower jaw bone
        lower_jaw = vtk.vtkCubeSource()
        lower_jaw.SetXLength(85)
        lower_jaw.SetYLength(85)
        lower_jaw.SetZLength(15)
        lower_jaw.SetCenter(0, 0, -22) 
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(lower_jaw.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
        prop.SetSpecular(0.3)
        prop.SetSpecularPower(20)
        prop.SetOpacity(0.4)
        
        self.segment_manager.add_segment(
            "Lower_Jaw_Bone", actor, mapper, None, "Dental", (0.85, 0.8, 0.7)
        )
        self.renderer.AddActor(actor)
        self.picker.AddPickList(actor)
        self.add_segment_to_tree("Lower_Jaw_Bone", "Dental")
        
        self.update_model_center()
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Demo dental model loaded - Neural signal & jaw animation ready")
            
    def reset_current_model(self):
        """Clear current model and reset all systems (Ensures jaw reset before clear)"""
        self.animation_timer.stop()
        self.flight_timer.stop()
        if self.neural_signal_animator:
            self.neural_signal_animator.stop_animation()
        if self.jaw_movement_controller:
            self.jaw_movement_controller.animation_timer.stop()
            self.reset_jaw_position() 
        
        self.set_animation_buttons_enabled(True)
        self.play_btn.setChecked(False)
        self.flight_btn.setChecked(False)
        
        self.jaw_is_open = False
        self.jaw_status_label.setText("Status: Closed")
        
        # --- MODIFIED: Removed MPR Generator reset ---
        
        for actor in self.segment_manager.get_all_actors():
            self.renderer.RemoveActor(actor)
        
        self.segment_manager.clear()
        self.segment_tree.clear()
        
        for actor in self.plane_actors:
            self.renderer.RemoveActor(actor)
        self.plane_actors.clear()
        
        for plane_actor in self.standard_plane_actors.values():
            self.renderer.RemoveActor(plane_actor)
        self.standard_plane_actors.clear()
        
        if self.focus_navigator.is_active:
            self.focus_navigator.deactivate()
            self.focus_nav_btn.setChecked(False)
            
        self.is_flight_mode = False
        self.is_diving = False
        self.is_picking_points = False
        self.model_center = [0, 0, 0]
        
        if hasattr(self, 'upper_opacity_slider'):
            self.upper_opacity_slider.setValue(100)
        if hasattr(self, 'lower_opacity_slider'):
            self.lower_opacity_slider.setValue(100)
        self.master_opacity_slider.setValue(100)
        
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Model reset - Ready to load new dental model")
    
    # ==================== VISUALIZATION CONTROLS (MODIFIED) ====================
    
    def update_segment_opacity(self, segment_name, value):
        """Update opacity for a single segment based on its slider"""
        opacity = value / 100.0
        self.segment_manager.set_opacity(segment_name, opacity)
        self.vtk_widget.GetRenderWindow().Render()
        
    def update_group_opacity(self, group_name_prefix, value):
        """Update opacity for a whole group (e.g., Upper/Lower) based on control tab sliders"""
        opacity = value / 100.0
        
        for name, segment in self.segment_manager.segments.items():
            if group_name_prefix.lower() in name.lower():
                self.segment_manager.set_opacity(name, opacity)
                
        root_name_map = {"Upper": "Upper Jaw (Maxilla)", "Lower": "Lower Jaw (Mandible)"}
        root_name = root_name_map.get(group_name_prefix)

        if root_name:
            root_item = None
            for i in range(self.segment_tree.topLevelItemCount()):
                temp_item = self.segment_tree.topLevelItem(i)
                if temp_item.text(0) == root_name:
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
        self.statusBar().showMessage(f"{group_name_prefix} Jaw opacity set to {value}%")

    def update_master_opacity(self, value):
        opacity = value / 100.0
        self.master_opacity_label.setText(f"{value}%")
        
        for segment in self.segment_manager.segments.values():
            segment['actor'].GetProperty().SetOpacity(opacity)
        
        if hasattr(self, 'upper_opacity_slider'):
            self.upper_opacity_slider.setValue(value)
        if hasattr(self, 'lower_opacity_slider'):
            self.lower_opacity_slider.setValue(value)
            
        self.vtk_widget.GetRenderWindow().Render()
        
    def update_quality(self, quality):
        for segment in self.segment_manager.segments.values():
            prop = segment['actor'].GetProperty()
            if quality == "Low":
                prop.SetInterpolationToFlat()
                prop.SetSpecular(0.1)
            elif quality == "Medium":
                prop.SetInterpolationToGouraud()
                prop.SetSpecular(0.3)
            elif quality == "High":
                prop.SetInterpolationToPhong()
                prop.SetSpecular(0.5)
                prop.SetSpecularPower(30)
            else: # Ultra
                prop.SetInterpolationToPhong()
                prop.SetSpecular(0.8)
                prop.SetSpecularPower(50)
        
        self.vtk_widget.GetRenderWindow().Render()
        
    def toggle_smooth_shading(self, state):
        for segment in self.segment_manager.segments.values():
            prop = segment['actor'].GetProperty()
            if state:
                prop.SetInterpolationToPhong()
            else:
                prop.SetInterpolationToFlat()
        
        self.vtk_widget.GetRenderWindow().Render()
        
    def toggle_edges(self, state):
        for segment in self.segment_manager.segments.values():
            segment['actor'].GetProperty().SetEdgeVisibility(state)
        
        self.vtk_widget.GetRenderWindow().Render()
        
    def apply_dental_colors(self):
        segments = list(self.segment_manager.segments.values())
        for i, segment in enumerate(segments):
            color = self.dental_colors[i % len(self.dental_colors)]
            segment['actor'].GetProperty().SetColor(*color)
        
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Applied natural dental color preset")
        
    def apply_single_color(self, color):
        for segment in self.segment_manager.segments.values():
            segment['actor'].GetProperty().SetColor(*color)
        
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"Applied single color to all segments")
    
    # --- NEW: Methods for Advanced Clipping ---
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
    
    # --- END: New Clipping Methods ---
    
    def toggle_intersection_lines(self, state):
        if state:
            self.statusBar().showMessage("Intersection lines enabled (Not Implemented)")
        else:
            self.statusBar().showMessage("Intersection lines disabled")
    
    # ==================== MPR ====================
    
    # --- NEW: Method to launch the new MPR dialog ---
    def open_mpr_dialog(self):
        if not HAS_NIBABEL:
            QMessageBox.warning(self, "Missing Libraries", "This feature requires 'nibabel' and 'matplotlib'.\nInstall with: pip install nibabel matplotlib")
            return
            
        if self.mpr_dialog is None:
            self.mpr_dialog = CurvedMPRDialog(self)
        self.mpr_dialog.show()
        self.mpr_dialog.raise_()
        self.mpr_dialog.activateWindow()
    
    # --- REMOVED: All old 3D-click MPR methods ---
    # (toggle_mpr_picking, clear_mpr_path, generate_mpr_slices)
    # (navigate_mpr_slice, toggle_all_slices, add_point_marker)
    
    # ==================== NAVIGATION ====================
                        
    def apply_precise_rotation(self):
        transform = vtk.vtkTransform()
        transform.RotateX(self.rotation_x.value())
        transform.RotateY(self.rotation_y.value())
        transform.RotateZ(self.rotation_z.value())
        
        for segment in self.segment_manager.segments.values():
            segment['actor'].SetUserTransform(transform)
        
        self.vtk_widget.GetRenderWindow().Render()
        
    def reset_rotation(self):
        self.rotation_x.setValue(0)
        self.rotation_y.setValue(0)
        self.rotation_z.setValue(0)
        
        for segment in self.segment_manager.segments.values():
            segment['actor'].SetUserTransform(None)
        
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Rotation reset")
        
    def save_camera_view(self):
        camera = self.renderer.GetActiveCamera()
        view_data = {
            'position': camera.GetPosition(),
            'focal_point': camera.GetFocalPoint(),
            'view_up': camera.GetViewUp(),
            'view_angle': camera.GetViewAngle()
        }
        
        self.saved_camera_views['dental'] = view_data
        self.statusBar().showMessage("Camera view saved for dental model")
    
    def load_camera_view(self):
        if 'dental' in self.saved_camera_views:
            view_data = self.saved_camera_views['dental']
            camera = self.renderer.GetActiveCamera()
            
            camera.SetPosition(view_data['position'])
            camera.SetFocalPoint(view_data['focal_point'])
            camera.SetViewUp(view_data['view_up'])
            camera.SetViewAngle(view_data['view_angle'])
            
            self.vtk_widget.GetRenderWindow().Render()
            self.statusBar().showMessage("Loaded saved camera view")
        else:
            QMessageBox.information(self, "No Saved View", "No saved camera view available")
    
    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Camera reset to default view")


def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = Dental3DVisualizationGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
