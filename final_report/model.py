import numpy as np
import matplotlib.pyplot as plt
import parameters as param 
from matplotlib.animation import FuncAnimation, PillowWriter
import os
class WormModel:
	def __init__(self, name, length, n_segments, dim=2, fixedDOFs=[], stretch_fraction=0.1):
		self.name = name
		self.n = n_segments
		self.length = length
		self.deltaL = length / n_segments
		
		self.dim = dim  # Set to 2 or 3 for 2D/3D
		self.q = np.zeros((n_segments*3+1, self.dim))
		self.q[0] = np.zeros(self.dim)
		
		# Springs: simple parallel arrays for easy vectorized computation
		# Each segment has 5 springs: 1 horizontal spring + 4 links
		n_springs = n_segments * 5
		self.springs = np.zeros((n_springs, 2), dtype=np.int32)  # [p1_idx, p2_idx] per spring
		self.spring_k = np.zeros(n_springs)   # spring constants
		self.spring_l0 = np.zeros(n_springs)  # rest lengths
		
		for i in range(1, n_segments+1):
			node_index = 3*i
			self.q[node_index] = [i * self.deltaL , 0.0, 0.0][:self.dim] # node position

		for i in range(n_segments):
			node1_index = 3*i
			node2_index = 3*(i+1)
			top_connector_index = 3*i + 1
			bot_connector_index = 3*i + 2

			x_pos = (self.q[node1_index][0] + self.q[node2_index][0]) / 2
			y_pos = np.sqrt((self.deltaL/2)**2 - (self.deltaL/4)**2)
			self.q[top_connector_index] = [x_pos, y_pos, 0.0][:self.dim] # top connector position
			self.q[bot_connector_index] = [x_pos, -y_pos, 0.0][:self.dim] # bottom connector position
	
			#springs
			self.springs[5*i] = [node1_index, node2_index] # horizontal spring
			self.springs[5*i+1] = [node1_index, top_connector_index] # top left link
			self.springs[5*i+2] = [top_connector_index, node2_index] # top right link
			self.springs[5*i+3] = [node1_index, bot_connector_index] # bottom left link
			self.springs[5*i+4] = [bot_connector_index, node2_index] # bottom right link
			self.spring_k[5*i] = 13 # spring constant
			self.spring_k[5*i+1] = 1e4 # link constant
			self.spring_k[5*i+2] = 1e4 # link constant
			self.spring_k[5*i+3] = 1e4 # link constant
			self.spring_k[5*i+4] = 1e4 # link constant
			self.spring_l0[5*i] = self.deltaL # rest length
			self.spring_l0[5*i+1] = np.sqrt( (self.deltaL/2)**2 + (y_pos)**2 ) # rest length
			self.spring_l0[5*i+2] = np.sqrt( (self.deltaL/2)**2 + (y_pos)**2 ) # rest length
			self.spring_l0[5*i+3] = np.sqrt( (self.deltaL/2)**2 + (y_pos)**2 ) # rest length
			self.spring_l0[5*i+4] = np.sqrt( (self.deltaL/2)**2 + (y_pos)**2 ) # rest length

		self.q0 = self.q.copy() # initial position
		self.u0 = np.zeros_like(self.q0) # initial velocity
		self.nv = self.q.shape[0]
		self.ndof = self.dim * self.nv
		self.ne = len(self.springs)

		# ============ NODE TYPE CLASSIFICATION ============
		# Identify which nodes are main nodes vs connectors
		# Main nodes: indices 0, 3, 6, 9, ... (every 3rd starting from 0)
		# Top connectors: indices 1, 4, 7, 10, ... (every 3rd starting from 1)
		# Bot connectors: indices 2, 5, 8, 11, ... (every 3rd starting from 2)
		self.is_main_node = np.zeros(self.nv, dtype=bool)
		self.is_connector = np.zeros(self.nv, dtype=bool)
		self.main_node_indices = []
		self.connector_indices = []
		
		for k in range(self.nv):
			if k % 3 == 0:  # Main nodes at 0, 3, 6, ...
				self.is_main_node[k] = True
				self.main_node_indices.append(k)
			else:  # Connectors at 1, 2, 4, 5, 7, 8, ...
				self.is_connector[k] = True
				self.connector_indices.append(k)
		
		self.main_node_indices = np.array(self.main_node_indices)
		self.connector_indices = np.array(self.connector_indices)
		
		# ============ TOGGLE FLAGS ============
		# Set these to True/False to include/exclude connectors from physics
		self.connectors_have_mass = False      # Set True to give connectors mass
		self.connectors_have_gravity = False   # Set True to apply gravity to connectors
		self.connectors_have_damping = False   # Set True to apply damping to connectors

		# Boundary Conditions # Set of all DOFs
		self.freeIndex = np.setdiff1d(np.arange(self.ndof), fixedDOFs) # All the DOFs are free except the fixed ones
		self.fixedIndex = fixedDOFs # Fixed DOFs
		self.isFixed = np.zeros(self.ndof)
		self.groundPosition = np.min(self.q[:, 1]) - 0.10 # Location of ground along y axis
		
		#-----------Forces------------#

		"""Calculate safe Fmax based on geometry."""
		deltaL = self.deltaL
		h = np.sqrt((deltaL/2)**2 - (deltaL/4)**2)
		cot_theta = (deltaL/2) / h  # ≈ 1.15
		k_spring = self.spring_k[0]  # horizontal spring stiffness
		Fmax = k_spring * stretch_fraction * deltaL / (4 * cot_theta)
		self.Fmax = Fmax

		# Radii of spheres
		R = np.zeros(self.nv)
		R_main = self.deltaL / 1000      # Radius for main nodes
		R_connector = self.deltaL / 10000  # Radius for connectors (can set to 0 or small)
		for k in range(self.nv):
			if self.is_main_node[k]:
				R[k] = R_main
			else:
				R[k] = R_connector if self.connectors_have_mass else 1e-10  # tiny if no mass
		self.R = R

		# Mass vector and matrix
		m = np.zeros(self.dim * self.nv)
		for k in range(self.nv):
			if self.is_main_node[k] or self.connectors_have_mass:
				node_mass = 4/3 * np.pi * R[k]**3 * param.rho_metal
			else:
				node_mass = 1e-10  # tiny mass for connectors (avoid division by zero)
			m[self.dim*k] = node_mass
			m[self.dim*k + 1] = node_mass
		self.m = m
		self.mMat = np.diag(m)
		
		# Gravity (external force)
		W = np.zeros(self.dim * self.nv)
		g = np.array([0, -9.8, 0])[:self.dim]  # m/s^2
		for k in range(self.nv):
			if self.is_main_node[k] or self.connectors_have_gravity:
				for d in range(self.dim):
					W[self.dim*k + d] = m[self.dim*k + d] * g[d]
			# else: W stays 0 for connectors
		self.W = W

		# Viscous damping (external force)
		C = np.zeros((2 * self.nv, 2 * self.nv))
		for k in range(self.nv):
			if self.is_main_node[k] or self.connectors_have_damping:
				C[2*k, 2*k] = 6.0 * np.pi * param.visc * R[k]
				C[2*k+1, 2*k+1] = 6.0 * np.pi * param.visc * R[k]
		self.c = C
		
	def update_internal_state(self, q):
		self.q[0] = q[0:3]
		for i in range(1, self.n+1):
			# Update node positions
			self.q[3*i] = q[3*i: 3*i+2]
			self.q[3*i+1] = q[3*i+1: 3*i+3]
			self.q[3*i+2] = q[3*i+2: 3*i+4]
		
	def plot_objects(self, ax=None, artists=None):
		# Create axes if not given (for static plot)
		if ax is None:
			fig, ax = plt.subplots(figsize=(8,4))

		# Extract current worm geometry
		q_x = self.q[::3, 0]
		q_y = self.q[::3, 1]

		# connectors (top/bottom)
		c_x = []
		c_y = []
		for i in range(self.n):
			c_x.append(self.q[3*i+1, 0])
			c_y.append(self.q[3*i+1, 1])
			c_x.append(self.q[3*i+2, 0])
			c_y.append(self.q[3*i+2, 1])

		# spring endpoints
		p1 = self.q[self.springs[:, 0]]
		p2 = self.q[self.springs[:, 1]]

		spring_mask = np.zeros(len(self.springs), dtype=bool)
		spring_mask[::5] = True
		link_mask = ~spring_mask

		# =====================================================
		#  INIT MODE → CREATE ARTISTS
		# =====================================================
		if artists is None:
			artists = {}

			artists["nodes"], = ax.plot(q_x, q_y, 'o', color='black', markersize=4)
			artists["conn"], = ax.plot(c_x, c_y, 'o', color='green', markersize=4)

			# Springs
			artists["springs"] = [
				ax.plot([], [], '-', color='red')[0] 
				for _ in np.where(spring_mask)[0]
			]

			# Links
			artists["links"] = [
				ax.plot([], [], '-', color='black')[0] 
				for _ in np.where(link_mask)[0]
			]

			# Ellipses
			artists["top"] = [ax.plot([], [], '-', linewidth=1)[0] for _ in range(self.n)]
			artists["bot"] = [ax.plot([], [], '-', linewidth=1)[0] for _ in range(self.n)]

			# also update them immediately for static plot
			self.plot_objects(ax=ax, artists=artists)
			return artists

		# =====================================================
		#  UPDATE MODE → UPDATE ARTISTS
		# =====================================================

		# Update nodes & connectors
		artists["nodes"].set_data(q_x, q_y)
		artists["conn"].set_data(c_x, c_y)

		# update springs
		spring_indices = np.where(spring_mask)[0]
		for j, i in enumerate(spring_indices):
			artists["springs"][j].set_data([p1[i, 0], p2[i, 0]],
										[p1[i, 1], p2[i, 1]])

		# update links
		link_indices = np.where(link_mask)[0]
		for j, i in enumerate(link_indices):
			artists["links"][j].set_data([p1[i, 0], p2[i, 0]],
										[p1[i, 1], p2[i, 1]])

		# update ellipses
		for i in range(self.n):
			a = (q_x[i] - q_x[i+1]) / 2
			theta = np.linspace(0, np.pi, 200)

			# top ellipse
			b_top = (c_y[2*i] - q_y[i]) / 2
			h = (q_x[i+1] + q_x[i])/2
			k = (c_y[2*i] + q_y[i])/2
			x = h + a*np.cos(theta)
			y = k + b_top*np.sin(theta)
			artists["top"][i].set_data(x, y)

			# bottom ellipse
			b_bot = (q_y[i] - c_y[2*i+1]) / 2
			theta2 = np.linspace(np.pi, 2*np.pi, 200)
			k2 = (q_y[i] + c_y[2*i+1])/2
			x = h + a*np.cos(theta2)
			y = k2 + b_bot*np.sin(theta2)
			artists["bot"][i].set_data(x, y)

		return artists

	
	def plot(self, ctime=0.0):
		fig, ax = plt.subplots(figsize=(8,4))
		self.plot_objects(ax=ax)  # init + update
		ax.set_title(f'Worm Configuration, Time: {ctime:.2f} s')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.axis('equal')
		plt.show()

	def plot_springs(worm, t):
		plt.figure()
		plt.title(f"Spring Network (Time: {t:.2f} s)")
		for _, ind in enumerate(worm.springs):
			p1 = int(ind[0])
			p2 = int(ind[1])
			#print(p1, p2)
			plt.plot([worm.q[p1, 0], worm.q[p2, 0]], [worm.q[p1, 1], worm.q[p2, 1]], 'bo-')
			plt.xlabel("x (m)")
			plt.ylabel("y (m)")
			plt.axis("equal")
		plt.grid(True)
		plt.show()


	def animate_worm(self, frames, times, save_dir="animations", target_fps=20, speedup=1.0):
		"""
		Parameters:
		- frames: list of q arrays from the solver
		- times: corresponding time array from the solver
		- save_dir: subdirectory to save GIF
		- target_fps: target frames per second for GIF (default 20, good for GIFs)
		- speedup: playback speed multiplier (1.0 = real-time, 2.0 = 2x speed)
		"""

		# Ensure save directory exists
		os.makedirs(save_dir, exist_ok=True)

		# Generate unique filename
		fname = self.name + "_anim.gif"
		counter = 1
		while os.path.exists(os.path.join(save_dir, fname)):
			fname = f"{self.name}({counter}).gif"
			counter += 1
		filepath = os.path.join(save_dir, fname)

		# === SUBSAMPLE FRAMES ===
		dt_sim = np.mean(np.diff(times))
		total_time = times[-1] - times[0]
		
		# Calculate frame skip to achieve target_fps at given speedup
		# We want: (num_output_frames / target_fps) = total_time / speedup
		# So: num_output_frames = target_fps * total_time / speedup
		num_output_frames = int(target_fps * total_time / speedup)
		num_output_frames = max(2, min(num_output_frames, len(frames)))  # clamp
		
		# Subsample indices
		frame_indices = np.linspace(0, len(frames)-1, num_output_frames, dtype=int)
		frames_subset = [frames[i] for i in frame_indices]
		
		print(f"Animation: {len(frames)} total frames → {len(frames_subset)} output frames")
		print(f"Playback: {total_time:.2f}s sim time at {speedup}x speed = {total_time/speedup:.2f}s GIF")
		# ========================

		# Figure setup
		fig, ax = plt.subplots(figsize=(8,4))
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_xlim([-1, 2])
		ax.set_ylim([-1, 1])
		
		# Plot objects
		artists = self.plot_objects(ax=ax)

		def flatten_artists():
			return [artists["nodes"], artists["conn"]] \
				+ artists["springs"] \
				+ artists["links"] \
				+ artists["top"] \
				+ artists["bot"]
		
		def init():
			return flatten_artists()

		def update(frame_idx):
			q = frames_subset[frame_idx]
			self.q = q
			self.plot_objects(ax=ax, artists=artists)
			return flatten_artists()

		ani = FuncAnimation(fig, update, frames=len(frames_subset),
							init_func=init, blit=True, interval=1000/target_fps)

		# Save as GIF
		ani.save(filepath, writer=PillowWriter(fps=target_fps))
		print(f"Animation saved to {filepath}")
		
		return ani
