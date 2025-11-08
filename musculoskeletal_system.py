import sys
import numpy as np
import pandas as pd
from scipy import interpolate
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox,
                             QColorDialog, QFileDialog, QGroupBox, QGridLayout,
                             QTabWidget, QCheckBox, QSpinBox, QDoubleSpinBox,
                             QTreeWidget, QTreeWidgetItem, QSplitter, QProgressBar,
                             QMessageBox, QListWidget, QDialog, QTextEdit, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QPalette, QIcon, QFont, QBrush
import vtk
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


class NeuralSignalAnimator:
    """Animates neural signal propagation on surface meshes - Motor signals only"""
    def __init__(self):
        self.active_animations = {}
        self.timer = None
        self.start_time = 0
        self.duration = 1.2
        self.signal_speed = 0.3
        
    def create_neural_signal(self, actor):
        """Create motor neural signal animation for an actor"""
        # --- MODIFIED: Light, transparent blue signal ---
        signal_color = (0.7, 0.9, 1.0) # Light Blue
        
        original_color = actor.GetProperty().GetColor()
        original_ambient = actor.GetProperty().GetAmbient()
        original_diffuse = actor.GetProperty().GetDiffuse()
        original_specular = actor.GetProperty().GetSpecular()
        
        mapper = actor.GetMapper()
        polydata = mapper.GetInput()
        bounds = polydata.GetBounds()
        
        animation_data = {
            'actor': actor,
            'signal_color': signal_color,
            'original_color': original_color,
            'original_ambient': original_ambient,
            'original_diffuse': original_diffuse,
            'original_specular': original_specular,
            'signal_type': 'motor',
            'start_time': time.time(),
            'bounds': bounds
        }
        
        return animation_data
    
    def update_signal_animation(self, animation_data, current_time):
        """Update the signal animation - top to bottom wave"""
        actor = animation_data['actor']
        elapsed = current_time - animation_data['start_time']
        
        if elapsed > self.duration:
            animation_data['start_time'] = current_time
            elapsed = 0
        
        progress = elapsed / self.duration
        
        bounds = animation_data['bounds']
        y_min = bounds[2]
        y_max = bounds[3]
        y_range = y_max - y_min if y_max > y_min else 1
        
        signal_y_position = y_max - (progress * y_range)
        signal_width = y_range * 0.15
        
        mapper = actor.GetMapper()
        polydata = mapper.GetInput()
        
        if polydata and polydata.GetNumberOfPoints() > 0:
            points = polydata.GetPoints()
            
            avg_y = 0
            for i in range(points.GetNumberOfPoints()):
                point = points.GetPoint(i)
                avg_y += point[1]
            avg_y /= points.GetNumberOfPoints()
            
            distance = abs(avg_y - signal_y_position)
            
            if distance < signal_width:
                intensity = 1.0 - (distance / signal_width)
                intensity = intensity ** 0.5
            else:
                intensity = 0
            
            r = animation_data['original_color'][0] * (1 - intensity) + animation_data['signal_color'][0] * intensity
            g = animation_data['original_color'][1] * (1 - intensity) + animation_data['signal_color'][1] * intensity
            b = animation_data['original_color'][2] * (1 - intensity) + animation_data['signal_color'][2] * intensity
            
            actor.GetProperty().SetColor(r, g, b)
            
            # --- MODIFIED: Adjusted for a "glowing blue" feel ---
            actor.GetProperty().SetAmbient(0.1 + intensity * 0.4)
            actor.GetProperty().SetDiffuse(0.7 + intensity * 0.1)
            actor.GetProperty().SetSpecular(0.3 + intensity * 0.3)
            actor.GetProperty().SetSpecularPower(10 + intensity * 10)
        
        return True


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
    
    def get_segments_by_system(self, system):
        return [self.segments[name] for name in self.segment_groups.get(system, [])]
    
    def get_all_actors(self):
        return [seg['actor'] for seg in self.segments.values()]
    
    def clear(self):
        self.segments.clear()
        self.segment_groups.clear()


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


# ===================================================================
# === MODIFIED: StairClimbAnimator Class (USER REQUEST) ===
# ===================================================================

class StairClimbAnimator:
    """
    Manages the user-requested animation sequence:
    Signal (Full Leg) -> Rotate (Lower Leg) -> Return to Start
    """
    def __init__(self, segment_manager, neural_animator, vtk_widget):
        self.segment_manager = segment_manager
        self.neural_animator = neural_animator
        self.vtk_widget = vtk_widget # This can be None initially
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        
        self.state = "IDLE"
        self.animation_step = 0
        self.animation_duration_frames = 30 # Frames for move up/down
        self.signal_duration_ms = int(self.neural_animator.duration * 1000)
        
        self.active_signals = []
        self.original_transforms = {} # Stores {actor: vtkTransform}

        # Store the calculated parts for the animation cycle
        self.right_moving_actors = [] # Lower leg + lower leg muscles
        self.left_moving_actors = []  # Lower leg + lower leg muscles
        self.right_signal_actors = [] # Full leg + all muscles
        self.left_signal_actors = []  # Full leg + all muscles
        self.right_pivot_point = [0, 0, 0]
        self.left_pivot_point = [0, 0, 0]

    def _get_all_leg_actors(self, side):
        """
        Gets ALL actors for a specific leg (upper and lower)
        for the signal animation.
        """
        actors = []
        prefix = f"VHF_{side}_"
        for name, segment in self.segment_manager.segments.items():
            if name.startswith(prefix):
                # Exclude pelvis/sacrum as they aren't 'leg'
                if "Pelvis" not in name and "Sacrum" not in name and "Coccyx" not in name:
                    actors.append(segment['actor'])
        return actors

    def _get_lower_leg_parts(self, side):
        """
        Finds the MOVING parts (lower leg) and the PIVOT point (knee)
        based on keywords in their names.
        """
        moving_actors = []
        pivot_point = None
        
        # --- MODIFIED: Explicit lists based on user's FULL dataset ---
        
        # Keywords for parts that move (LOWER LEG)
        lower_leg_bone_keywords = [
            "Bone_Tibia", "Bone_Fibula", "Bone_Patella", 
            "Bone_Calcaneous", "Bone_Cuboid", "Bone_IntermediateCuneiform", 
            "Bone_LateralCuneiform", "Bone_MedialCuneiform", "Bone_Navicular", 
            "Bone_Phalanges", "Bone_Talus"
        ]
        lower_leg_cartilage_keywords = [
            "Cartilage_TibiaDistal", "Cartilage_TibiaLateral", "Cartilage_TibiaMedial",
            "Cartilage_Patella", "Cartilage_Talus"
        ]
        # These are muscles known to be in the lower leg (from your file list)
        lower_leg_muscle_keywords = [
            "Muscle_ExtensorDigitorumLongus",
            "Muscle_ExtensorHallucisLongus",
            "Muscle_FlexorDigitorumLongus",
            "Muscle_FlexorHallucisLongus",
            "Muscle_GastrocnemiusLateral",
            "Muscle_GastrocnemiusMedial",
            "Muscle_PeroneusLongus",
            "Muscle_Plantaris",
            "Muscle_Popliteus",
            "Muscle_Soleus",
            "Muscle_TibialisAnterior",
            "Muscle_TibialisPosterior"
        ]
        
        # Keywords for pivot point
        pivot_cartilage_keyword = "Cartilage_FemurDistal"
        pivot_bone_keyword = "Bone_Femur"
        
        prefix = f"VHF_{side}_" # e.g., "VHF_Right_"
        
        femur_bone_actor = None
        femur_distal_actor = None
        
        for name, segment in self.segment_manager.segments.items():
            if name.startswith(prefix):
                part_name_full = name[len(prefix):] # e.g., "Bone_Tibia_smooth"
                part_name = part_name_full.replace("_smooth", "") # e.g., "Bone_Tibia"
                
                # 1. Check for Pivot Actors
                if part_name == pivot_cartilage_keyword:
                    femur_distal_actor = segment['actor']
                elif part_name == pivot_bone_keyword:
                    femur_bone_actor = segment['actor']

                # 2. Check if it's a moving part (LOWER leg)
                is_moving_part = False
                
                if part_name in lower_leg_bone_keywords:
                    is_moving_part = True
                elif part_name in lower_leg_cartilage_keywords:
                    is_moving_part = True
                elif part_name in lower_leg_muscle_keywords:
                    is_moving_part = True
                
                if is_moving_part:
                    moving_actors.append(segment['actor'])

        
        # 3. Determine the pivot point
        if femur_distal_actor:
            pivot_point = femur_distal_actor.GetCenter()
            print(f"Pivot for {side}: Found {prefix}{pivot_cartilage_keyword}")
        elif femur_bone_actor:
            bounds = femur_bone_actor.GetBounds()
            pivot_point = [(bounds[0] + bounds[1]) / 2, bounds[2], (bounds[4] + bounds[5]) / 2]
            print(f"Pivot for {side}: {prefix}{pivot_cartilage_keyword} not found. "
                  f"Falling back to bottom-center of {prefix}{pivot_bone_keyword}.")
        else:
            print(f"Error: No pivot actor found for {side} (missing {pivot_cartilage_keyword} or {pivot_bone_keyword}).")
            
        return moving_actors, pivot_point

    def start(self):
        """
        Prepares and starts the animation sequence.
        Calculates pivots and moving parts before starting.
        """
        if self.state != "IDLE":
            print("Animation already running.")
            return

        if not self.vtk_widget:
            print("Error: StairClimbAnimator has no vtk_widget set.")
            return

        # --- 1. Find all parts for Right Leg ---
        self.right_moving_actors, self.right_pivot_point = self._get_lower_leg_parts("Right")
        self.right_signal_actors = self._get_all_leg_actors("Right")
        
        # --- MODIFIED: Added check for signal actors ---
        if not self.right_moving_actors or not self.right_pivot_point or not self.right_signal_actors:
            QMessageBox.warning(self.vtk_widget, "Missing Right Leg Parts",
                                "Could not find required parts for the right leg.\n"
                                "Need 'VHF_Right_...FemurDistal' (ideal) OR 'VHF_Right_Bone_Femur' (as pivot)\n"
                                "AND moving parts like 'VHF_Right_Bone_Tibia'.")
            return
        
        # --- 2. Find all parts for Left Leg ---
        self.left_moving_actors, self.left_pivot_point = self._get_lower_leg_parts("Left")
        self.left_signal_actors = self._get_all_leg_actors("Left")

        # --- MODIFIED: Added check for signal actors ---
        if not self.left_moving_actors or not self.left_pivot_point or not self.left_signal_actors:
            QMessageBox.warning(self.vtk_widget, "Missing Left Leg Parts",
                                "Could not find required parts for the left leg.\n"
                                "Need 'VHF_Left_...FemurDistal' (ideal) OR 'VHF_Left_Bone_Femur' (as pivot)\n"
                                "AND moving parts like 'VHF_Left_Bone_Tibia'.")
            return

        # --- 3. Store transforms and start ---
        print(f"Starting stair climb. Right Pivot: {self.right_pivot_point}, Left Pivot: {self.left_pivot_point}")
        self._store_original_transforms()
        # --- MODIFIED: Start with LEFT leg per user request ---
        self.state = "SIGNAL_LEFT_START"
        self.timer.start(33) # ~30 FPS

    def stop(self):
        """Stops the animation and resets all transforms."""
        print("Stopping stair climb sequence.")
        self.timer.stop()
        self._stop_active_signals()
        self._reset_all_transforms()
        self.state = "IDLE"
        if self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()

    def _tick(self):
        """The main animation loop callback."""
        current_time = time.time()
        
        if self.active_signals:
            self._update_active_signals(current_time)

        # --- MODIFIED: State Machine now starts with LEFT leg ---
        if self.state == "SIGNAL_LEFT_START":
            self._stop_active_signals()
            # --- Signal ALL leg actors ---
            print(f"State: SIGNAL_LEFT_START (Signaling {len(self.left_signal_actors)} actors)")
            self._start_signal_on_actors(self.left_signal_actors)
            self.animation_step = 0
            self.state = "SIGNAL_LEFT_RUN"
            QTimer.singleShot(self.signal_duration_ms, lambda: self._advance_state_to("MOVE_LEFT_UP"))

        elif self.state == "MOVE_LEFT_UP":
            progress = self.animation_step / self.animation_duration_frames
            transform = self._get_rotation_transform(progress, self.left_pivot_point, angle=-60)
            # --- Move ONLY the moving actors ---
            self._apply_transform_to_moving_parts(self.left_moving_actors, transform)
            
            self.animation_step += 1
            if self.animation_step > self.animation_duration_frames:
                self.animation_step = 0
                self.state = "MOVE_LEFT_DOWN"
        
        elif self.state == "MOVE_LEFT_DOWN":
            progress = 1.0 - (self.animation_step / self.animation_duration_frames)
            transform = self._get_rotation_transform(progress, self.left_pivot_point, angle=-60)
            self._apply_transform_to_moving_parts(self.left_moving_actors, transform)
            
            self.animation_step += 1
            if self.animation_step > self.animation_duration_frames:
                self.animation_step = 0
                # --- Transition to RIGHT leg ---
                self.state = "SIGNAL_RIGHT_START"

        elif self.state == "SIGNAL_RIGHT_START":
            self._stop_active_signals()
            # --- Signal ALL leg actors ---
            print(f"State: SIGNAL_RIGHT_START (Signaling {len(self.right_signal_actors)} actors)")
            self._start_signal_on_actors(self.right_signal_actors)
            self.animation_step = 0
            self.state = "SIGNAL_RIGHT_RUN"
            QTimer.singleShot(self.signal_duration_ms, lambda: self._advance_state_to("MOVE_RIGHT_UP"))
        
        elif self.state == "MOVE_RIGHT_UP":
            progress = self.animation_step / self.animation_duration_frames
            transform = self._get_rotation_transform(progress, self.right_pivot_point, angle=-60)
            # --- Move ONLY the moving actors ---
            self._apply_transform_to_moving_parts(self.right_moving_actors, transform)
            
            self.animation_step += 1
            if self.animation_step > self.animation_duration_frames:
                self.animation_step = 0
                self.state = "MOVE_RIGHT_DOWN"

        elif self.state == "MOVE_RIGHT_DOWN":
            progress = 1.0 - (self.animation_step / self.animation_duration_frames)
            transform = self._get_rotation_transform(progress, self.right_pivot_point, angle=-60)
            self._apply_transform_to_moving_parts(self.right_moving_actors, transform)
            
            self.animation_step += 1
            if self.animation_step > self.animation_duration_frames:
                self.animation_step = 0
                self.state = "IDLE" # Cycle complete
                self.timer.stop()
                print("Sequence complete.")

        if self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()

    def _advance_state_to(self, new_state):
        """Callback for QTimer.singleShot to advance state after signal."""
        if self.state.endswith("_RUN"):
            print(f"Signal finished, advancing to {new_state}")
            self.animation_step = 0
            self.state = new_state
            self._stop_active_signals()

    def _get_rotation_transform(self, progress, pivot_point, angle=-60):
        """
        Calculates the vtkTransform for rotating around a pivot point.
        """
        final_angle = progress * angle
        px, py, pz = pivot_point
        
        transform = vtk.vtkTransform()
        transform.Translate(px, py, pz)        # 3. Move pivot back to original position
        transform.RotateX(final_angle)      # 2. Rotate around origin (X-axis for knee bend)
        transform.Translate(-px, -py, -pz)   # 1. Move pivot to origin
        
        return transform

    def _apply_transform_to_moving_parts(self, moving_actors, anim_transform):
        """Applies a new transform to all actors in a list."""
        for actor in moving_actors:
            original_t = self.original_transforms.get(actor, vtk.vtkTransform())
            
            combined_t = vtk.vtkTransform()
            combined_t.Concatenate(anim_transform) # Animation transform first
            combined_t.Concatenate(original_t)   # Then original transform
            
            actor.SetUserTransform(combined_t)
            
    def _store_original_transforms(self):
        """Stores the current transform of all moving actors."""
        self.original_transforms.clear()
        all_moving_actors = self.right_moving_actors + self.left_moving_actors
        for actor in all_moving_actors:
            t = vtk.vtkTransform()
            if actor.GetUserTransform():
                t.DeepCopy(actor.GetUserTransform())
            self.original_transforms[actor] = t
            
    def _reset_all_transforms(self):
        """Resets all actors in original_transforms to their stored transforms."""
        for actor, t in self.original_transforms.items():
            actor.SetUserTransform(t)
        self.original_transforms.clear()

    def _start_signal_on_actors(self, actors):
        """Starts the neural signal animator on a specific list of actors."""
        self._stop_active_signals()
        for actor in actors:
            anim_data = self.neural_animator.create_neural_signal(actor)
            self.active_signals.append(anim_data)
            
    def _update_active_signals(self, current_time):
        """Called by _tick() to update any running signals."""
        for anim_data in self.active_signals:
            self.neural_animator.update_signal_animation(anim_data, current_time)
            
    def _stop_active_signals(self):
        """Stops and cleans up all active neural signal animations."""
        for anim_data in self.active_signals:
            actor = anim_data['actor']
            actor.GetProperty().SetColor(*anim_data['original_color'])
            actor.GetProperty().SetAmbient(anim_data['original_ambient'])
            actor.GetProperty().SetDiffuse(anim_data['original_diffuse'])
            actor.GetProperty().SetSpecular(anim_data['original_specular'])
        self.active_signals.clear()


# ===================================================================
# === ClippingDialog Class (from brain_mpr.py) ===
# ===================================================================

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

# ===================================================================
# === CurvedMPRDialog CLASS (from brain_mpr.py) ===
# ===================================================================

class CurvedMPRDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Curved Multi-Planar Reconstruction")
        self.setGeometry(100, 100, 900, 800) 
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
            self.reset_curve() 
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
            self.display_slice() 
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
        
        start_z = self.start_slice_spin.value()
        end_z = self.end_slice_spin.value()

        if start_z >= end_z:
            QMessageBox.warning(self, "Error", "Start slice must be less than end slice.")
            return
        
        try:
            cpr_volume = self.volume[:, :, start_z:end_z+1]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to slice volume:\n{e}")
            return
        
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
                
                if 0 <= xi < cpr_volume.shape[0] and 0 <= yi < cpr_volume.shape[1]:
                    straightened.append(cpr_volume[xi, yi, :])
                else:
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

# ===================================================================
# === END NEW CLASSES ===
# ===================================================================


class Muscle3DVisualizationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("💪 Musculoskeletal 3D Visualization System - Complete Edition")
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
        self.focus_navigator = FocusNavigator(self.segment_manager)
        self.neural_animator = NeuralSignalAnimator()
        
        self.clipping_dialog = None
        self.mpr_dialog = None
        self.plane_actors = []
        
        self.stair_climb_animator = StairClimbAnimator(self.segment_manager, self.neural_animator, None)

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
        
        self.is_picking_points = False 
        self.picker = None
        self.saved_camera_views = {}
        self.camera_angle = 0
        
        self.standard_plane_actors = {}
        
        self.system_colors = {
            'Musculoskeletal': [(0.9, 0.85, 0.75), (0.95, 0.9, 0.8), (0.85, 0.8, 0.7), (1.0, 1.0, 1.0)]
        }
        
        self.part_sliders = {}
        self.model_center = [0, 0, 0]
        
        self.init_ui()
        
        self.stair_climb_animator.vtk_widget = self.vtk_widget
        
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
        self.statusBar().showMessage("Ready - Load musculoskeletal models | Enhanced Navigation Available")
        
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
        
        self.interactor.Initialize()
        self.interactor.Start()
        
    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        title = QLabel("💪 Musculoskeletal 3D Viewer")
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {self.colors['accent_cyan']}; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout()
        
        # --- MODIFIED: This is the main button now ---
        load_folder_btn = QPushButton("📂 Load Model Folder (Left & Right)")
        load_folder_btn.setStyleSheet(f"background-color: {self.colors['accent_green']}; color: #111; font-size: 13px; padding: 10px;")
        load_folder_btn.clicked.connect(self.load_model_folder)
        data_layout.addWidget(load_folder_btn)
        
        load_segment_btn = QPushButton("📁 Load Single Segment")
        load_segment_btn.clicked.connect(self.load_segment_file)
        data_layout.addWidget(load_segment_btn)
        
        load_demo_muscle_btn = QPushButton("💪 Load Demo Muscle System")
        load_demo_muscle_btn.clicked.connect(self.load_demo_muscle)
        data_layout.addWidget(load_demo_muscle_btn)
        
        reset_btn = QPushButton("🔄 RESET - Clear Current Model")
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
        tabs.addTab(self.create_navigation_tab(), "🧭 Navigation")
        
        tabs.addTab(self.create_neural_signal_tab(), "⚡ Leg Animation")
        
        layout.addWidget(tabs)
        return panel
    
    def create_neural_signal_tab(self):
        """
        --- MODIFIED: Text changed to English and updated logic ---
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info_group = QGroupBox("Stair Climb Animation")
        info_layout = QVBoxLayout()
        
        info_label = QLabel(
            "This sequence triggers the neural signal then flexes the knee.\n\n"
            "1. Signal on Left Leg (Upper & Lower)\n"
            "2. Left knee flexes (Lower Leg Parts), then returns\n"
            "3. Signal on Right Leg (Upper & Lower)\n"
            "4. Right knee flexes (Lower Leg Parts), then returns"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {self.colors['accent_cyan']}; padding: 10px; font-size: 12px;")
        info_layout.addWidget(info_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        control_group = QGroupBox("Animation Controls")
        control_layout = QVBoxLayout()
        
        self.start_neural_btn = QPushButton("▶️ START Stair Climb Sequence")
        self.start_neural_btn.clicked.connect(self.start_stair_climb_sequence) 
        self.start_neural_btn.setStyleSheet(f"background-color: {self.colors['accent_green']}; color: #111; font-size: 16px; padding: 15px; font-weight: bold;")
        control_layout.addWidget(self.start_neural_btn)
        
        self.stop_neural_btn = QPushButton("⏹️ STOP Sequence & Reset")
        self.stop_neural_btn.clicked.connect(self.stop_stair_climb_sequence) 
        self.stop_neural_btn.setStyleSheet(f"background-color: {self.colors['accent_orange']}; font-size: 16px; padding: 15px; font-weight: bold;")
        control_layout.addWidget(self.stop_neural_btn)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        status_group = QGroupBox("Animation Status")
        status_layout = QVBoxLayout()
        
        self.neural_status_label = QLabel("Status: Ready")
        self.neural_status_label.setAlignment(Qt.AlignCenter)
        self.neural_status_label.setStyleSheet(f"color: {self.colors['accent_green']}; font-size: 14px; padding: 10px; font-weight: bold;")
        status_layout.addWidget(self.neural_status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        return tab
        
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
        
        part_opacity_group = QGroupBox("Individual Part Opacity")
        part_opacity_layout = QGridLayout()
        
        part_keys = {
            "Bones": "bone",
            "Muscles": "muscle",
            "Ligaments": "ligament",
            "Cartilage": "cartilage",
        }
        
        row = 0
        for label_text, search_key in part_keys.items():
            label = QLabel(f"{label_text} Opacity:")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(100)
            
            slider.valueChanged.connect(
                lambda value, key=search_key: self.update_individual_part_opacity(key, value)
            )
            
            part_opacity_layout.addWidget(label, row, 0)
            part_opacity_layout.addWidget(slider, row, 1)
            
            self.part_sliders[search_key] = slider
            row += 1
            
        part_opacity_group.setLayout(part_opacity_layout)
        layout.addWidget(part_opacity_group)
        
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
        
        color_group = QGroupBox("Color Presets (Apply to All)")
        color_layout = QGridLayout()
        
        preset_buttons = [
            ("Bone White", (1.0, 1.0, 1.0)),
            ("Muscle Tone", (0.8, 0.4, 0.4)),
            ("Cartilage Blue", (0.9, 0.9, 1.0)),
            ("Custom Color", None)
        ]
        
        for i, (name, color) in enumerate(preset_buttons):
            btn = QPushButton(name)
            if color:
                btn.clicked.connect(lambda checked, c=color: self.apply_muscle_colors(c))
            else:
                btn.clicked.connect(self.choose_custom_color)
            color_layout.addWidget(btn, i // 2, i % 2)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        layout.addStretch()
        return tab
        
    def create_clipping_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info = QLabel("Advanced clipping with visible anatomical planes")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {self.colors['accent_cyan']}; padding: 10px;")
        layout.addWidget(info)
        
        open_btn = QPushButton("🔓 Open Advanced Clipping")
        open_btn.setStyleSheet(f"background-color: {self.colors['accent_green']}; font-size: 14px; padding: 12px;")
        open_btn.clicked.connect(self.open_clipping_dialog)
        layout.addWidget(open_btn)
        
        layout.addStretch()
        return tab
    
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
        
    def create_navigation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        flight_group = QGroupBox("✈️ Flying Camera (Deep Dive)")
        flight_layout = QVBoxLayout()
        
        info_flight = QLabel("Click on model to dive deep with spiral camera animation")
        info_flight.setWordWrap(True)
        info_flight.setStyleSheet(f"color: {self.colors['text_light']}; font-size: 10px;")
        flight_layout.addWidget(info_flight)
        
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
    
    # ==================== EVENT HANDLERS ====================
    
    def on_segment_tree_changed(self, item, column):
        """Handles the checkbox for visibility."""
        if column == 0:
            name = item.text(0)
            visible = item.checkState(0) == Qt.Checked
            self.segment_manager.set_visibility(name, visible)
            self.vtk_widget.GetRenderWindow().Render()
            
    def on_segment_clicked(self, item, column):
        segment_name = item.text(0)
        if segment_name in self.segment_manager.segments:
            segment = self.segment_manager.segments[segment_name]
            self.statusBar().showMessage(
                f"Selected: {segment_name} | System: {segment['system']} | "
                f"Opacity: {segment['opacity']*100:.0f}%"
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
    
    # ==================== FLYING CAMERA METHODS ====================
    
    def toggle_flight_mode(self, checked):
        self.is_flight_mode = checked
        if checked:
            self.flight_btn.setText("🎯 Click on Model to Dive")
            self.statusBar().showMessage("Select target to fly to")
            self.is_picking_points = False
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
            self.is_picking_points = False
            self.flight_btn.setChecked(False)

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
    
    # ==================== NEURAL SIGNAL / ANIMATION (MODIFIED) ====================
    
    def start_stair_climb_sequence(self):
        """NEW: Starts the main stair climb animation sequence."""
        self.stair_climb_animator.start()
        self.neural_status_label.setText("Status: Sequence ACTIVE")
        self.neural_status_label.setStyleSheet(f"color: {self.colors['accent_cyan']}; font-size: 14px; padding: 10px; font-weight: bold;")
        self.statusBar().showMessage("Stair climb sequence started...")

    def stop_stair_climb_sequence(self):
        """NEW: Stops the main stair climb animation sequence."""
        self.stair_climb_animator.stop()
        self.neural_status_label.setText("Status: Stopped & Reset")
        self.neural_status_label.setStyleSheet(f"color: {self.colors['accent_orange']}; font-size: 14px; padding: 10px; font-weight: bold;")
        self.statusBar().showMessage("Stair climb sequence stopped and reset.")
    
    # ==================== DATA LOADING (MODIFIED) ====================

    def load_model_folder(self):
        """
        NEW: Replaces mirror function. Loads all files from a directory.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with ALL 3D Models (Left & Right)")
        if not folder_path:
            return
            
        files = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.stl', '.obj', '.ply', '.vtk'))]
        
        if not files:
            QMessageBox.warning(self, "No Files", "No 3D model files found in folder")
            return
            
        num_loaded = 0
        for i, filename in enumerate(files):
            file_path = os.path.join(folder_path, filename)
            segment_name = os.path.splitext(filename)[0]
            
            # --- MODIFIED: Assign color based on type ---
            if "muscle" in segment_name.lower():
                color = (0.8, 0.4, 0.4) # Red-ish for muscle
            elif "cartilage" in segment_name.lower():
                color = (0.9, 0.9, 1.0) # Light blue/white for cartilage
            elif "ligament" in segment_name.lower():
                color = (0.9, 0.7, 0.9) # Light purple for ligament
            else:
                color = (0.9, 0.85, 0.75) # Default bone color
            
            self.load_segment(file_path, segment_name, "Musculoskeletal", color)
            num_loaded += 1
        
        self.update_model_center()
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"Loaded {num_loaded} segments from folder. Ready for animation.")

    def load_segment_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Segment", "",
            "3D Files (*.stl *.obj *.ply *.vtk);;All Files (*)"
        )
        if file_path:
            filename = os.path.basename(file_path)
            segment_name = os.path.splitext(filename)[0]
            
            self.load_segment(file_path, segment_name, "Musculoskeletal")
            self.update_model_center()
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
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
            print(f"Skipping unknown file type: {ext}")
            return
        
        reader.SetFileName(file_path)
        reader.Update()
        
        if reader.GetOutput().GetNumberOfPoints() == 0:
            print(f"Warning: File {file_path} is empty or unreadable.")
            return

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(reader.GetOutputPort())
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
            color = (0.9, 0.85, 0.75)
        
        self.segment_manager.add_segment(segment_name, actor, mapper, reader, system, color)
        self.renderer.AddActor(actor)
        self.picker.AddPickList(actor)
        
        self.add_segment_to_tree(segment_name, system)
        
    def add_segment_to_tree(self, segment_name, system):
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
        
        self.segment_tree.addTopLevelItem(item)
        self.segment_tree.setItemWidget(item, 1, opacity_widget)
        
    def load_demo_muscle(self):
        """Load procedural demo muscle system with bones"""
        self.reset_current_model()
        
        bone_configs = [
            {"name": "Humerus_Bone", "pos": (0, 0, 0), "radius": 4, "height": 80},
            {"name": "Radius_Bone", "pos": (10, -50, 0), "radius": 3, "height": 70},
            {"name": "Ulna_Bone", "pos": (-10, -50, 0), "radius": 3, "height": 70}
        ]
        
        for config in bone_configs:
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(config["radius"])
            cylinder.SetHeight(config["height"])
            cylinder.SetResolution(20)
            cylinder.SetCenter(*config["pos"])
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cylinder.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetInterpolationToPhong()
            prop.SetSpecular(0.7)
            prop.SetSpecularPower(40)
            
            self.segment_manager.add_segment(
                config["name"], actor, mapper, None, "Musculoskeletal", (1.0, 1.0, 1.0)
            )
            self.renderer.AddActor(actor)
            self.picker.AddPickList(actor)
            self.add_segment_to_tree(config["name"], "Musculoskeletal")
        
        muscle_configs = [
            {"name": "Biceps", "pos": (8, 15, 0), "radius": 12, "height": 50, "color": (0.9, 0.75, 0.65)},
            {"name": "Triceps", "pos": (-8, 15, 0), "radius": 10, "height": 48, "color": (0.85, 0.7, 0.6)},
            {"name": "Forearm_Flexor", "pos": (10, -50, 0), "radius": 8, "height": 60, "color": (0.88, 0.73, 0.63)},
            {"name": "Forearm_Extensor", "pos": (-10, -50, 0), "radius": 8, "height": 60, "color": (0.92, 0.77, 0.67)}
        ]
        
        for config in muscle_configs:
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(config["radius"])
            cylinder.SetHeight(config["height"])
            cylinder.SetResolution(25)
            cylinder.SetCenter(*config["pos"])
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cylinder.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetInterpolationToPhong()
            prop.SetSpecular(0.4)
            prop.SetSpecularPower(15)
            
            self.segment_manager.add_segment(
                config["name"], actor, mapper, None, "Musculoskeletal", config["color"]
            )
            self.renderer.AddActor(actor)
            self.picker.AddPickList(actor)
            self.add_segment_to_tree(config["name"], "Musculoskeletal")
        
        self.update_model_center()
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Demo muscle system loaded | Enhanced navigation & motor signal ready")
        
    def reset_current_model(self):
        """Clear current model and reset all systems"""
        self.animation_timer.stop()
        self.flight_timer.stop()
        self.play_btn.setChecked(False)
        self.flight_btn.setChecked(False)
        
        self.stair_climb_animator.stop()
        
        for actor in self.plane_actors:
            self.renderer.RemoveActor(actor)
        self.plane_actors.clear()
        if self.clipping_dialog:
            self.clipping_dialog.reset_all()
        
        for actor in self.segment_manager.get_all_actors():
            self.renderer.RemoveActor(actor)
        
        self.segment_manager.clear()
        self.segment_tree.clear()
        
        for plane_actor in self.standard_plane_actors.values():
            self.renderer.RemoveActor(plane_actor)
        self.standard_plane_actors.clear()
        
        if self.focus_navigator.is_active:
            self.focus_navigator.deactivate()
            self.focus_nav_btn.setChecked(False)
        
        for slider in self.part_sliders.values():
            slider.blockSignals(True)
            slider.setValue(100)
            slider.blockSignals(False)
        self.master_opacity_slider.setValue(100)
        
        self.is_flight_mode = False
        self.is_diving = False
        self.is_picking_points = False
        
        self.model_center = [0, 0, 0]
        
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Model reset - Ready to load new system")
    
    # ==================== VISUALIZATION CONTROLS ====================
    
    def update_segment_opacity(self, segment_name, value):
        opacity = value / 100.0
        self.segment_manager.set_opacity(segment_name, opacity)
        self.vtk_widget.GetRenderWindow().Render()
        
    def update_master_opacity(self, value):
        opacity = value / 100.0
        self.master_opacity_label.setText(f"{value}%")
        
        for segment in self.segment_manager.segments.values():
            segment['actor'].GetProperty().SetOpacity(opacity)
        
        for slider in self.part_sliders.values():
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def update_individual_part_opacity(self, part_key, value):
        """Update opacity for specific part type"""
        opacity = value / 100.0
        
        search_keys = [part_key]
        # --- MODIFIED: Added all known muscle/bone types from user files ---
        if part_key == 'muscle':
            search_keys.extend(['biceps', 'triceps', 'flexor', 'extensor', 'vastus',
                                'adductor', 'gluteus', 'gracilis', 'illiacus', 'gemellus',
                                'obturator', 'pectineus', 'peroneus', 'piriformis', 'plantaris',
                                'popliteus', 'psoas', 'quadratus', 'rectus', 'sartorius',
                                'semimembranosus', 'semitendinosus', 'soleus', 'tensor', 'tibialis'])
        if part_key == 'bone':
            search_keys.extend(['femur', 'tibia', 'fibula', 'patella', 'pelvis', 'sacrum',
                                'calcaneous', 'coccyx', 'cuboid', 'cuneiform', 'navicular',
                                'phalanges', 'talus'])
        if part_key == 'ligament':
            search_keys.extend(['ligament'])
        if part_key == 'cartilage':
            search_keys.extend(['cartilage'])
        
        for name in self.segment_manager.segments.keys():
            name_lower = name.lower()
            for key in search_keys:
                if key in name_lower:
                    self.segment_manager.set_opacity(name, opacity)
                    break
        
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
            else:  # Ultra
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
        
    def apply_muscle_colors(self, color):
        segments = list(self.segment_manager.segments.values())
        for segment in segments:
            segment['actor'].GetProperty().SetColor(*color)
        
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"Applied muscle color preset")
    
    def choose_custom_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b = color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0
            self.apply_muscle_colors((r, g, b))
    
    # ==================== CLIPPING (NEW) ====================
    
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
    
    # ==================== MPR (NEW) ====================
    
    def open_mpr_dialog(self):
        if not HAS_NIBABEL:
            QMessageBox.warning(self, "Missing Libraries", "This feature requires 'nibabel' and 'matplotlib'.\nInstall with: pip install nibabel matplotlib")
            return
            
        if self.mpr_dialog is None:
            self.mpr_dialog = CurvedMPRDialog(self)
        self.mpr_dialog.show()
        self.mpr_dialog.raise_()
        self.mpr_dialog.activateWindow()
    
    # ==================== NAVIGATION ====================
            
    def apply_precise_rotation(self):
        transform = vtk.vtkTransform()
        transform.RotateX(self.rotation_x.value())
        transform.RotateY(self.rotation_y.value())
        transform.RotateZ(self.rotation_z.value())
        
        for segment in self.segment_manager.segments.values():
            if segment['actor'] not in self.stair_climb_animator.original_transforms:
                segment['actor'].SetUserTransform(transform)
        
        self.vtk_widget.GetRenderWindow().Render()
        
    def reset_rotation(self):
        self.rotation_x.setValue(0)
        self.rotation_y.setValue(0)
        self.rotation_z.setValue(0)
        
        for segment in self.segment_manager.segments.values():
             if segment['actor'] not in self.stair_climb_animator.original_transforms:
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
        
        self.saved_camera_views["Musculoskeletal"] = view_data
        self.statusBar().showMessage(f"Camera view saved for Musculoskeletal system")
    
    def load_camera_view(self):
        if "Musculoskeletal" in self.saved_camera_views:
            view_data = self.saved_camera_views["Musculoskeletal"]
            camera = self.renderer.GetActiveCamera()
            
            camera.SetPosition(view_data['position'])
            camera.SetFocalPoint(view_data['focal_point'])
            camera.SetViewUp(view_data['view_up'])
            camera.SetViewAngle(view_data['view_angle'])
            
            self.vtk_widget.GetRenderWindow().Render()
            self.statusBar().showMessage(f"Loaded saved view for Musculoskeletal system")
        else:
            QMessageBox.information(self, "No Saved View", f"No saved camera view for Musculoskeletal system")
    
    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Camera reset to default view")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = Muscle3DVisualizationGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
