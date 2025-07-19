import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import time
import cv2
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
from collections import deque
import queue
import warnings
warnings.filterwarnings('ignore')

# Advanced Deep Learning and Reinforcement Learning components
class DeepQNetwork:
    """Deep Q-Network for reinforcement learning"""
    def __init__(self, state_size, action_size, hidden_layers=[128, 64]):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Simple neural network layers
        layer_sizes = [state_size] + hidden_layers + [action_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, state):
        activation = state
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activation, weight) + bias
            if i < len(self.weights) - 1:  # Apply ReLU to all layers except output
                activation = self.relu(z)
            else:
                activation = z  # Linear output for Q-values
        return activation
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.uniform(-1, 1, self.action_size)  # Random exploration
        
        q_values = self.forward(state.reshape(1, -1))
        return q_values[0]
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += 0.95 * np.max(self.forward(next_state.reshape(1, -1)))
            
            target_f = self.forward(state.reshape(1, -1))
            
            # Update Q-values (simplified backpropagation)
            error = target - np.mean(target_f)
            
            # Simple gradient descent update
            for i in range(len(self.weights)):
                self.weights[i] += self.learning_rate * error * 0.001 * np.random.randn(*self.weights[i].shape)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class ComputerVisionSystem:
    """Computer vision system for target detection"""
    def __init__(self):
        self.cap = None
        self.target_position = [0, 0, 0]
        self.detection_active = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.detection_thread = None
        self.latest_frame = None
        self.target_detected = False
        
        # Color ranges for target detection (HSV)
        self.color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)]
        }
        self.current_color = 'red'
        
    def initialize_camera(self):
        """Initialize webcam"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Warning: Could not open webcam. Using simulated targets.")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully!")
            return True
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return False
    
    def start_detection(self):
        """Start the computer vision detection in a separate thread"""
        if not self.detection_active:
            self.detection_active = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
    
    def stop_detection(self):
        """Stop computer vision detection"""
        self.detection_active = False
        if self.cap:
            self.cap.release()
    
    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.detection_active:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.latest_frame = frame.copy()
                    target_pos, detected = self._detect_target(frame)
                    if detected:
                        # Convert pixel coordinates to 3D world coordinates
                        self.target_position = self._pixel_to_world(target_pos, frame.shape)
                        self.target_detected = True
                    else:
                        self.target_detected = False
            else:
                # Simulate moving targets when no camera
                t = time.time()
                self.target_position = [
                    1.5 * np.sin(t * 0.5),
                    1.0 * np.cos(t * 0.3),
                    0.5 + 0.3 * np.sin(t * 0.7)
                ]
                self.target_detected = True
            
            time.sleep(0.033)  # ~30 FPS
    
    def _detect_target(self, frame):
        """Detect colored target in frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for target color
        lower, upper = self.color_ranges[self.current_color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 500:  # Minimum area threshold
                # Get centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw detection on frame
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f'Target: ({cx}, {cy})', (cx-50, cy-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    return (cx, cy), True
        
        return (0, 0), False
    
    def _pixel_to_world(self, pixel_pos, frame_shape):
        """Convert pixel coordinates to 3D world coordinates"""
        height, width = frame_shape[:2]
        
        # Normalize pixel coordinates to [-1, 1]
        norm_x = (pixel_pos[0] - width/2) / (width/2)
        norm_y = (height/2 - pixel_pos[1]) / (height/2)  # Flip Y axis
        
        # Map to world coordinates (adjust scale as needed)
        world_x = norm_x * 2.0
        world_y = norm_y * 2.0
        world_z = 1.0  # Default height
        
        return [world_x, world_y, world_z]
    
    def get_frame_for_display(self):
        """Get current frame for display"""
        return self.latest_frame if self.latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

@dataclass
class JointState3D:
    angle: float
    velocity: float
    torque: float
    
@dataclass
class TaskMetrics:
    completion_time: float
    energy_consumed: float
    path_efficiency: float
    success_rate: float

class RoboticArm3D:
    """3D Robotic arm with multiple joints"""
    def __init__(self, num_joints=6):
        self.num_joints = num_joints
        self.joint_states = [JointState3D(0.0, 0.0, 0.0) for _ in range(num_joints)]
        
        # 3D link parameters
        self.link_lengths = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2][:num_joints]
        self.link_offsets = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0][:num_joints]  # Z-offsets for 3D
        
        # Joint constraints
        self.joint_limits = [(-np.pi, np.pi) for _ in range(num_joints)]
        self.max_velocities = [2.0, 2.0, 3.0, 3.0, 4.0, 4.0][:num_joints]
        
        # 3D position history
        self.position_history = deque(maxlen=200)
        self.current_task = None
        
        # DH parameters for proper 3D kinematics
        self.dh_params = self._initialize_dh_parameters()
        
    def _initialize_dh_parameters(self):
        """Initialize Denavit-Hartenberg parameters for 3D kinematics"""
        # Standard 6-DOF arm DH parameters [a, alpha, d, theta]
        dh = []
        for i in range(self.num_joints):
            a = self.link_lengths[i] if i > 0 else 0
            alpha = np.pi/2 if i % 2 == 0 else 0
            d = self.link_offsets[i]
            theta = 0  # Will be updated with joint angles
            dh.append([a, alpha, d, theta])
        return dh
    
    def forward_kinematics_3d(self):
        """Calculate 3D forward kinematics using DH parameters"""
        positions = [(0, 0, 0)]  # Base position
        
        # Transformation matrix (start with identity)
        T = np.eye(4)
        
        for i, (joint, dh_params) in enumerate(zip(self.joint_states, self.dh_params)):
            a, alpha, d, _ = dh_params
            theta = joint.angle
            
            # DH transformation matrix
            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            
            T_i = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])
            
            T = np.dot(T, T_i)
            
            # Extract position
            pos = T[:3, 3]
            positions.append((pos[0], pos[1], pos[2]))
        
        end_effector_pos = positions[-1]
        return positions, end_effector_pos
    
    def inverse_kinematics_3d(self, target_pos):
        """3D inverse kinematics using iterative method"""
        target = np.array(target_pos)
        
        # Iterative solution
        for iteration in range(50):  # Max iterations
            positions, current_end = self.forward_kinematics_3d()
            current_end = np.array(current_end)
            
            error = target - current_end
            if np.linalg.norm(error) < 0.01:  # Convergence threshold
                break
            
            # Jacobian approximation (simplified)
            jacobian = self._compute_jacobian()
            
            # Pseudo-inverse solution
            if jacobian.size > 0:
                try:
                    delta_theta = np.linalg.pinv(jacobian) @ error
                    
                    # Apply joint updates with limits
                    for i, delta in enumerate(delta_theta[:self.num_joints]):
                        self.joint_states[i].angle += delta * 0.1  # Damping factor
                        
                        # Apply joint limits
                        min_angle, max_angle = self.joint_limits[i]
                        self.joint_states[i].angle = np.clip(
                            self.joint_states[i].angle, min_angle, max_angle
                        )
                except:
                    # Fallback to simple geometric solution
                    break
        
        return [joint.angle for joint in self.joint_states]
    
    def _compute_jacobian(self):
        """Compute Jacobian matrix for inverse kinematics"""
        # Simplified Jacobian computation
        epsilon = 0.001
        jacobian = []
        
        positions, original_end = self.forward_kinematics_3d()
        original_end = np.array(original_end)
        
        for i in range(self.num_joints):
            # Perturb joint angle
            original_angle = self.joint_states[i].angle
            self.joint_states[i].angle += epsilon
            
            # Compute perturbed end effector position
            _, perturbed_end = self.forward_kinematics_3d()
            perturbed_end = np.array(perturbed_end)
            
            # Compute partial derivative
            partial_derivative = (perturbed_end - original_end) / epsilon
            jacobian.append(partial_derivative)
            
            # Restore original angle
            self.joint_states[i].angle = original_angle
        
        return np.array(jacobian).T if jacobian else np.array([])
    
    def move_to_position_3d(self, target_pos, dt=0.1):
        """Move to target 3D position"""
        target_angles = self.inverse_kinematics_3d(target_pos)
        self.move_to_angles(target_angles, dt)
    
    def move_to_angles(self, target_angles, dt=0.1):
        """Move joints to target angles with velocity constraints"""
        for i, (current, target) in enumerate(zip(self.joint_states, target_angles)):
            angle_diff = target - current.angle
            max_change = self.max_velocities[i] * dt
            
            if abs(angle_diff) > max_change:
                change = max_change if angle_diff > 0 else -max_change
            else:
                change = angle_diff
            
            current.angle += change
            current.velocity = change / dt
            current.torque = abs(change) * 10
            
            # Apply joint limits
            min_angle, max_angle = self.joint_limits[i]
            current.angle = np.clip(current.angle, min_angle, max_angle)

class AdvancedPathOptimizer:
    """Advanced AI path optimizer using Deep Reinforcement Learning"""
    def __init__(self, arm: RoboticArm3D):
        self.arm = arm
        
        # State: joint angles + velocities + target position + distance to target
        state_size = arm.num_joints * 2 + 3 + 1  # angles + velocities + target_xyz + distance
        action_size = arm.num_joints  # joint angle adjustments
        
        self.dqn = DeepQNetwork(state_size, action_size)
        self.optimization_history = []
        self.learning_enabled = True
        self.total_rewards = 0
        self.episode_count = 0
        self.success_count = 0
        
        # Performance tracking
        self.performance_metrics = {
            'episode_rewards': deque(maxlen=100),
            'success_rate': deque(maxlen=100),
            'average_error': deque(maxlen=100),
            'learning_progress': deque(maxlen=1000)
        }
        
    def get_state(self, target_pos):
        """Get current state vector"""
        joint_angles = [joint.angle for joint in self.arm.joint_states]
        joint_velocities = [joint.velocity for joint in self.arm.joint_states]
        
        _, current_pos = self.arm.forward_kinematics_3d()
        distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
        
        state = joint_angles + joint_velocities + list(target_pos) + [distance]
        return np.array(state)
    
    def optimize_path_rl(self, target_pos):
        """Use Deep RL to find optimal path"""
        state = self.get_state(target_pos)
        
        # Get action from DQN
        if self.learning_enabled:
            action = self.dqn.act(state)
        else:
            # Fallback to inverse kinematics
            return self.arm.inverse_kinematics_3d(target_pos)
        
        # Convert action to joint angles (action is in [-1, 1])
        target_angles = []
        for i, act in enumerate(action):
            min_angle, max_angle = self.arm.joint_limits[i]
            angle = min_angle + (max_angle - min_angle) * (act + 1) / 2
            target_angles.append(angle)
        
        return target_angles[:self.arm.num_joints]
    
    def evaluate_and_learn(self, old_state, action, target_pos):
        """Evaluate performance and update the neural network"""
        # Calculate new state after action
        new_state = self.get_state(target_pos)
        
        # Calculate reward
        _, current_pos = self.arm.forward_kinematics_3d()
        distance_error = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
        
        # Multi-objective reward function
        accuracy_reward = max(0, 2.0 - distance_error)  # Higher for closer positions
        energy_penalty = sum(abs(joint.torque) for joint in self.arm.joint_states) * 0.001
        smoothness_reward = -sum(abs(joint.velocity) for joint in self.arm.joint_states) * 0.01
        
        reward = accuracy_reward - energy_penalty + smoothness_reward
        
        # Success bonus
        success = distance_error < 0.1
        if success:
            reward += 5.0
            self.success_count += 1
        
        # Store experience and learn
        if self.learning_enabled:
            done = success or distance_error > 5.0  # Episode ends on success or failure
            self.dqn.remember(old_state, action, reward, new_state, done)
            self.dqn.replay()
        
        # Update metrics
        self.total_rewards += reward
        self.performance_metrics['episode_rewards'].append(reward)
        self.performance_metrics['average_error'].append(distance_error)
        self.performance_metrics['learning_progress'].append(reward)
        
        if len(self.performance_metrics['episode_rewards']) > 0:
            recent_success_rate = sum(1 for r in list(self.performance_metrics['episode_rewards'])[-10:] 
                                    if r > 3) / min(10, len(self.performance_metrics['episode_rewards']))
            self.performance_metrics['success_rate'].append(recent_success_rate)
        
        return reward, distance_error, success

class Advanced3DDashboard:
    """Advanced 3D visualization dashboard with computer vision integration"""
    def __init__(self, arm: RoboticArm3D, optimizer: AdvancedPathOptimizer):
        self.arm = arm
        self.optimizer = optimizer
        self.vision_system = ComputerVisionSystem()
        
        # Visualization components
        self.fig = None
        self.axes = {}
        self.animation = None
        
        # Control variables
        self.running = True
        self.cv_enabled = False
        self.frame_count = 0
        self.target_position = [1.5, 1.0, 1.0]
        
        # Performance tracking
        self.metrics_history = {
            'completion_time': deque(maxlen=100),
            'energy_consumed': deque(maxlen=100),
            'distance_error': deque(maxlen=100),
            'success_rate': deque(maxlen=100),
            'rl_rewards': deque(maxlen=100)
        }
        
        # Initialize with some data
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize metrics with some sample data"""
        for _ in range(20):
            self.metrics_history['completion_time'].append(random.uniform(0.01, 0.05))
            self.metrics_history['energy_consumed'].append(random.uniform(1, 5))
            self.metrics_history['distance_error'].append(random.uniform(0.05, 0.3))
            self.metrics_history['success_rate'].append(random.uniform(0.6, 0.9))
            self.metrics_history['rl_rewards'].append(random.uniform(-1, 3))
    
    def setup_dashboard(self):
        """Setup the advanced 3D dashboard"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Advanced 3D AI Robotic Arm with Computer Vision Integration', 
                         fontsize=16, color='white', weight='bold')
        
        # Create sophisticated layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # Main 3D arm visualization
        self.axes['arm_3d'] = self.fig.add_subplot(gs[:2, :2], projection='3d')
        self.setup_3d_plot()
        
        # Computer vision feed
        self.axes['cv_feed'] = self.fig.add_subplot(gs[0, 2])
        self.axes['cv_feed'].set_title('Computer Vision Feed', color='white', fontsize=10)
        self.axes['cv_feed'].axis('off')
        
        # Joint states visualization
        self.axes['joints'] = self.fig.add_subplot(gs[1, 2])
        self.axes['joints'].set_title('Joint States & Torques', color='white', fontsize=10)
        
        # RL Performance metrics
        self.axes['rl_metrics'] = self.fig.add_subplot(gs[0, 3])
        self.axes['rl_metrics'].set_title('RL Performance', color='white', fontsize=10)
        
        # System performance
        self.axes['performance'] = self.fig.add_subplot(gs[1, 3])
        self.axes['performance'].set_title('System Metrics', color='white', fontsize=10)
        
        # Learning progress
        self.axes['learning'] = self.fig.add_subplot(gs[2, :])
        self.axes['learning'].set_title('Deep RL Learning Progress & Performance History', 
                                      color='white', fontsize=12)
        
        self.add_advanced_controls()
        self.initialize_plots()
        
    def setup_3d_plot(self):
        """Setup the 3D plot environment"""
        ax = self.axes['arm_3d']
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([0, 4])
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.set_title('3D Robotic Arm Workspace', color='white', fontsize=12, weight='bold')
        
        # Set dark background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)
        
    def add_advanced_controls(self):
        """Add advanced interactive controls"""
        # Control buttons row
        button_width, button_height = 0.08, 0.04
        button_y = 0.02
        
        # Start/Stop button
        ax_start = plt.axes([0.02, button_y, button_width, button_height])
        self.start_button = Button(ax_start, 'Start/Stop', color='lightblue')
        self.start_button.on_clicked(self.toggle_simulation)
        
        # AI Learning toggle
        ax_learning = plt.axes([0.12, button_y, button_width, button_height])
        self.learning_button = Button(ax_learning, 'Toggle RL', color='lightgreen')
        self.learning_button.on_clicked(self.toggle_learning)
        
        # Computer Vision toggle
        ax_cv = plt.axes([0.22, button_y, button_width, button_height])
        self.cv_button = Button(ax_cv, 'Toggle CV', color='yellow')
        self.cv_button.on_clicked(self.toggle_computer_vision)
        
        # Reset button
        ax_reset = plt.axes([0.32, button_y, button_width, button_height])
        self.reset_button = Button(ax_reset, 'Reset Arm', color='orange')
        self.reset_button.on_clicked(self.reset_arm)
        
        # 3D Target position sliders
        slider_width, slider_height = 0.15, 0.02
        slider_x = 0.45
        
        ax_x = plt.axes([slider_x, 0.04, slider_width, slider_height])
        self.slider_x = Slider(ax_x, 'Target X', -2.5, 2.5, 
                              valinit=self.target_position[0], color='red')
        
        ax_y = plt.axes([slider_x, 0.02, slider_width, slider_height])
        self.slider_y = Slider(ax_y, 'Target Y', -2.5, 2.5, 
                              valinit=self.target_position[1], color='green')
        
        ax_z = plt.axes([slider_x, 0.00, slider_width, slider_height])
        self.slider_z = Slider(ax_z, 'Target Z', 0.5, 3.5, 
                              valinit=self.target_position[2], color='blue')
        
        # Connect slider events
        self.slider_x.on_changed(self.update_target)
        self.slider_y.on_changed(self.update_target)
        self.slider_z.on_changed(self.update_target)
        
    def toggle_simulation(self, event):
        self.running = not self.running
        print(f"Simulation: {'ON' if self.running else 'OFF'}")
        
    def toggle_learning(self, event):
        self.optimizer.learning_enabled = not self.optimizer.learning_enabled
        print(f"Deep RL Learning: {'ON' if self.optimizer.learning_enabled else 'OFF'}")
        
    def toggle_computer_vision(self, event):
        self.cv_enabled = not self.cv_enabled
        if self.cv_enabled:
            if self.vision_system.initialize_camera():
                self.vision_system.start_detection()
                print("Computer Vision: ON")
            else:
                print("Computer Vision: Failed to start (using simulation)")
                self.cv_enabled = True  # Use simulated targets
                self.vision_system.start_detection()
        else:
            self.vision_system.stop_detection()
            print("Computer Vision: OFF")
            
    def reset_arm(self, event):
        for joint in self.arm.joint_states:
            joint.angle = 0.0
            joint.velocity = 0.0
            joint.torque = 0.0
        print("Arm reset to home position")
        
    def update_target(self, val):
        if not self.cv_enabled:  # Only update from sliders if CV is off
            self.target_position = [self.slider_x.val, self.slider_y.val, self.slider_z.val]
        
    def initialize_plots(self):
        """Initialize all plots with default data"""
        # Set arm to a reasonable starting position
        for i, joint in enumerate(self.arm.joint_states):
            joint.angle = np.sin(i * 0.3) * 0.4
            
        self.update_3d_arm_plot()
        self.update_joint_plot()
        self.update_cv_feed()
        self.update_performance_plots()
        
    def start_animation(self):
        """Start the real-time animation"""
        def animate(frame):
            return self.update_display(frame)
            
        self.animation = animation.FuncAnimation(
            self.fig, animate, interval=50, cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()
        
        # Cleanup
        if self.vision_system.cap:
            self.vision_system.stop_detection()
    
    def update_display(self, frame):
        """Update all dashboard components"""
        if not self.running:
            return []
            
        self.frame_count += 1
        
        # Update target from computer vision or sliders
        if self.cv_enabled and self.vision_system.target_detected:
            self.target_position = self.vision_system.target_position.copy()
        else:
            self.target_position = [self.slider_x.val, self.slider_y.val, self.slider_z.val]
        
        # Execute AI-powered movement
        start_time = time.time()
        old_state = self.optimizer.get_state(self.target_position)
        
        # Get optimal path using Deep RL
        optimal_angles = self.optimizer.optimize_path_rl(self.target_position)
        
        # Execute movement
        self.arm.move_to_angles(optimal_angles, dt=0.05)
        
        # Evaluate and learn from experience
        reward, error, success = self.optimizer.evaluate_and_learn(
            old_state, optimal_angles, self.target_position
        )
        
        # Update metrics
        completion_time = time.time() - start_time
        energy = sum(abs(joint.torque) for joint in self.arm.joint_states)
        
        self.metrics_history['completion_time'].append(completion_time)
        self.metrics_history['energy_consumed'].append(energy)
        self.metrics_history['distance_error'].append(error)
        self.metrics_history['rl_rewards'].append(reward)
        
        # Update visualizations
        self.update_3d_arm_plot()
        self.update_joint_plot()
        self.update_cv_feed()
        self.update_performance_plots()
        
        return []
        
    def update_3d_arm_plot(self):
        """Update the 3D arm visualization"""
        ax = self.axes['arm_3d']
        ax.clear()
        self.setup_3d_plot()
        
        # Get 3D arm positions
        positions, end_pos = self.arm.forward_kinematics_3d()
        
        # Draw arm links with gradient colors
        colors = ['cyan', 'yellow', 'magenta', 'orange', 'lime', 'pink']
        
        for i in range(len(positions) - 1):
            x_vals = [positions[i][0], positions[i+1][0]]
            y_vals = [positions[i][1], positions[i+1][1]]
            z_vals = [positions[i][2], positions[i+1][2]]
            
            color = colors[i % len(colors)]
            ax.plot3D(x_vals, y_vals, z_vals, color=color, linewidth=6, alpha=0.8)
        
        # Draw joints as spheres
        for i, pos in enumerate(positions):
            if i == 0:
                ax.scatter(pos[0], pos[1], pos[2], color='red', s=200, alpha=0.9, label='Base')
            elif i == len(positions) - 1:
                ax.scatter(pos[0], pos[1], pos[2], color='green', s=150, alpha=0.9, label='End Effector')
            else:
                ax.scatter(pos[0], pos[1], pos[2], color='blue', s=100, alpha=0.8)
        
        # Draw target with pulsing effect
        pulse_size = 200 + 100 * np.sin(self.frame_count * 0.2)
        ax.scatter(self.target_position[0], self.target_position[1], self.target_position[2], 
                  color='red', s=pulse_size, alpha=0.8, marker='*', label='Target')
        
        # Draw workspace sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        workspace_radius = sum(self.arm.link_lengths) * 0.9
        
        x_sphere = workspace_radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = workspace_radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = workspace_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + workspace_radius
        
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='white')
        
        # Draw trajectory
        if len(self.arm.position_history) > 1:
            history = list(self.arm.position_history)
            x_hist = [pos[0] for pos in history]
            y_hist = [pos[1] for pos in history]
            z_hist = [pos[2] for pos in history]
            ax.plot3D(x_hist, y_hist, z_hist, 'w--', alpha=0.6, linewidth=2, label='Trajectory')
        
        # Add distance and performance info
        distance = np.linalg.norm(np.array(self.target_position) - np.array(end_pos))
        info_text = f"Distance to Target: {distance:.3f}m\n"
        info_text += f"RL Epsilon: {self.optimizer.dqn.epsilon:.3f}\n"
        info_text += f"Success Rate: {self.optimizer.success_count}/{self.optimizer.episode_count}\n"
        info_text += f"CV Mode: {'ON' if self.cv_enabled else 'OFF'}"
        
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        ax.legend(loc='upper right')
        
        # Store position history
        self.arm.position_history.append(end_pos)
        
    def update_joint_plot(self):
        """Update joint states and torque visualization"""
        ax = self.axes['joints']
        ax.clear()
        
        joint_angles = [joint.angle for joint in self.arm.joint_states]
        joint_velocities = [joint.velocity for joint in self.arm.joint_states]
        joint_torques = [joint.torque for joint in self.arm.joint_states]
        
        x = np.arange(len(joint_angles))
        width = 0.25
        
        # Create grouped bar chart
        bars1 = ax.bar(x - width, joint_angles, width, label='Angles (rad)', 
                      color='skyblue', alpha=0.8)
        bars2 = ax.bar(x, joint_velocities, width, label='Velocities (rad/s)', 
                      color='orange', alpha=0.8)
        bars3 = ax.bar(x + width, [t/10 for t in joint_torques], width, 
                      label='Torques/10 (Nm)', color='red', alpha=0.8)
        
        ax.set_title('Joint States & Torques', color='white', fontsize=10)
        ax.set_ylabel('Values', color='white')
        ax.set_xticks(x)
        ax.set_xticklabels([f'J{i+1}' for i in range(len(joint_angles))])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add joint limit indicators
        for i, (min_angle, max_angle) in enumerate(self.arm.joint_limits):
            ax.axhline(y=min_angle, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=max_angle, color='red', linestyle='--', alpha=0.5)
        
    def update_cv_feed(self):
        """Update computer vision feed"""
        ax = self.axes['cv_feed']
        ax.clear()
        ax.set_title('Computer Vision Feed', color='white', fontsize=10)
        ax.axis('off')
        
        if self.cv_enabled:
            frame = self.vision_system.get_frame_for_display()
            if frame is not None and frame.size > 0:
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(frame_rgb)
                
                # Add CV status info
                status_text = f"Target Detection: {'ON' if self.vision_system.target_detected else 'OFF'}\n"
                status_text += f"Color Filter: {self.vision_system.current_color.upper()}\n"
                status_text += f"3D Position: ({self.target_position[0]:.2f}, {self.target_position[1]:.2f}, {self.target_position[2]:.2f})"
                
                ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'No Camera Feed\n(Using Simulated Targets)', 
                       ha='center', va='center', color='yellow', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Computer Vision\nDISABLED', 
                   ha='center', va='center', color='white', fontsize=14)
        
    def update_performance_plots(self):
        """Update RL performance and system metrics"""
        # RL Performance metrics
        ax_rl = self.axes['rl_metrics']
        ax_rl.clear()
        
        if len(self.optimizer.performance_metrics['episode_rewards']) > 0:
            recent_rewards = list(self.optimizer.performance_metrics['episode_rewards'])[-20:]
            recent_errors = list(self.optimizer.performance_metrics['average_error'])[-20:]
            
            x = range(len(recent_rewards))
            ax_rl.plot(x, recent_rewards, 'purple', linewidth=2, label='RL Rewards', alpha=0.8)
            ax_rl.plot(x, [-e*5 for e in recent_errors], 'red', linewidth=2, 
                      label='Error*(-5)', alpha=0.8)
            
            ax_rl.set_title('RL Performance (Recent)', color='white', fontsize=10)
            ax_rl.set_ylabel('Values', color='white')
            ax_rl.legend(fontsize=8)
            ax_rl.grid(True, alpha=0.3)
            
            # Add performance stats
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            avg_error = np.mean(recent_errors) if recent_errors else 0
            
            stats_text = f"Avg Reward: {avg_reward:.2f}\nAvg Error: {avg_error:.3f}\nEpsilon: {self.optimizer.dqn.epsilon:.3f}"
            ax_rl.text(0.02, 0.98, stats_text, transform=ax_rl.transAxes, fontsize=8,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # System performance metrics
        ax_perf = self.axes['performance']
        ax_perf.clear()
        
        if len(self.metrics_history['completion_time']) > 0:
            metrics = ['Time*100', 'Energy/10', 'Accuracy', 'RL Reward']
            
            recent_time = list(self.metrics_history['completion_time'])[-1] * 100
            recent_energy = list(self.metrics_history['energy_consumed'])[-1] / 10
            recent_error = list(self.metrics_history['distance_error'])[-1]
            recent_reward = list(self.metrics_history['rl_rewards'])[-1]
            
            accuracy = max(0, 1.0 - recent_error)
            values = [recent_time, recent_energy, accuracy, recent_reward]
            colors = ['blue', 'red', 'green', 'purple']
            
            bars = ax_perf.bar(metrics, values, color=colors, alpha=0.7)
            ax_perf.set_title('Current System Metrics', color='white', fontsize=10)
            ax_perf.set_ylabel('Normalized Values', color='white')
            ax_perf.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax_perf.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Learning progress and history
        ax_learning = self.axes['learning']
        ax_learning.clear()
        
        if len(self.optimizer.performance_metrics['learning_progress']) > 10:
            progress_data = list(self.optimizer.performance_metrics['learning_progress'])
            x = range(len(progress_data))
            
            # Plot multiple metrics
            ax_learning.plot(x, progress_data, 'purple', linewidth=2, 
                           label='RL Rewards', alpha=0.8)
            
            if len(self.metrics_history['distance_error']) > 1:
                error_data = list(self.metrics_history['distance_error'])
                error_x = np.linspace(0, len(progress_data)-1, len(error_data))
                ax_learning.plot(error_x, [-e*3 for e in error_data], 'red', 
                               linewidth=2, label='Distance Error*(-3)', alpha=0.8)
            
            if len(self.metrics_history['energy_consumed']) > 1:
                energy_data = list(self.metrics_history['energy_consumed'])
                energy_x = np.linspace(0, len(progress_data)-1, len(energy_data))
                ax_learning.plot(energy_x, [e/10 for e in energy_data], 'orange', 
                               linewidth=2, label='Energy/10', alpha=0.8)
            
            ax_learning.set_title('Deep RL Learning Progress & Performance History', 
                                color='white', fontsize=12)
            ax_learning.set_ylabel('Values', color='white')
            ax_learning.set_xlabel('Training Steps', color='white')
            ax_learning.legend()
            ax_learning.grid(True, alpha=0.3)
            
            # Add comprehensive statistics
            recent_window = 50
            if len(progress_data) >= recent_window:
                recent_rewards = progress_data[-recent_window:]
                avg_recent_reward = np.mean(recent_rewards)
                reward_trend = np.mean(recent_rewards[-10:]) - np.mean(recent_rewards[:10])
                
                stats_text = f"Training Steps: {len(progress_data)}\n"
                stats_text += f"Avg Recent Reward: {avg_recent_reward:.3f}\n"
                stats_text += f"Reward Trend: {reward_trend:+.3f}\n"
                stats_text += f"Exploration Rate: {self.optimizer.dqn.epsilon:.3f}\n"
                stats_text += f"Success Rate: {self.optimizer.success_count}/{self.optimizer.episode_count}\n"
                stats_text += f"Learning: {'ON' if self.optimizer.learning_enabled else 'OFF'}"
                
                ax_learning.text(0.02, 0.98, stats_text, transform=ax_learning.transAxes,
                               verticalalignment='top', fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

def main():
    """Main function to run the enhanced 3D robotic arm system"""
    print("="*80)
    print("🤖 ADVANCED 3D AI ROBOTIC ARM SYSTEM 🤖")
    print("="*80)
    print("Features:")
    print("✅ 3D Robotic Arm Simulation with Forward/Inverse Kinematics")
    print("✅ Deep Reinforcement Learning (DQN) for Path Optimization")
    print("✅ Computer Vision Integration with Real-time Target Detection")
    print("✅ Advanced Performance Metrics and Learning Analytics")
    print("✅ Interactive 3D Visualization Dashboard")
    print("="*80)
    
    try:
        print("🔧 Initializing system components...")
        
        # Create enhanced system components
        arm = RoboticArm3D(num_joints=6)
        optimizer = AdvancedPathOptimizer(arm)
        dashboard = Advanced3DDashboard(arm, optimizer)
        
        print("🎮 Setting up interactive dashboard...")
        dashboard.setup_dashboard()
        
        print("🚀 System ready! Available controls:")
        print("┌─────────────────────────────────────────────────────────┐")
        print("│ 🎯 Start/Stop: Begin/pause simulation                  │")
        print("│ 🧠 Toggle RL: Enable/disable Deep Reinforcement Learning│")
        print("│ 📷 Toggle CV: Enable/disable Computer Vision detection  │")
        print("│ 🔄 Reset Arm: Return arm to home position             │")
        print("│ 🎚️  Sliders: Manually adjust 3D target position (X,Y,Z)│")
        print("└─────────────────────────────────────────────────────────┘")
        print()
        print("🔬 Advanced Features:")
        print("• Deep Q-Network learns optimal movement strategies")
        print("• Computer vision detects colored objects as targets")
        print("• 3D forward/inverse kinematics with DH parameters")
        print("• Real-time performance analytics and learning curves")
        print("• Multi-objective reward function (accuracy, energy, smoothness)")
        print()
        print("🎥 Computer Vision Setup:")
        print("• Place a RED colored object in front of your webcam")
        print("• Toggle CV to start real-time target detection")
        print("• System will track object position in 3D space")
        print()
        print("📊 Watch the AI learn and improve over time!")
        print("Close the plot window to exit the system.")
        print("="*80)
        
        # Start the advanced visualization
        dashboard.start_animation()
        
    except KeyboardInterrupt:
        print("\n🛑 System shutdown requested...")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔌 Cleaning up resources...")
        try:
            if 'dashboard' in locals():
                dashboard.vision_system.stop_detection()
        except:
            pass
        print("✅ System shutdown complete!")

if __name__ == "__main__":
    main()