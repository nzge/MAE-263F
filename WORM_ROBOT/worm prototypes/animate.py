def save_animation_gif(ani, subdir="animations", base_name="worm_animation", fps=30):
    """
    Save a Matplotlib animation as a GIF to a subdirectory.
    Automatically increments the filename if it already exists.
    
    Parameters
    ----------
    ani : matplotlib.animation.FuncAnimation
        The animation object
    subdir : str
        Subdirectory to save the GIF
    base_name : str
        Base name of the file
    fps : int
        Frames per second
    """
    # Ensure the directory exists
    os.makedirs(subdir, exist_ok=True)
    
    # Find a unique filename
    idx = 0
    while True:
        if idx == 0:
            filename = f"{base_name}.gif"
        else:
            filename = f"{base_name}({idx}).gif"
        filepath = os.path.join(subdir, filename)
        if not os.path.exists(filepath):
            break
        idx += 1
    
		# Save animation
    ani.save(filepath, writer='pillow', fps=fps)
    print(f"Animation saved to: {filepath}")


	def plot_objects(self):

		# Plot objects
		node_pts, = ax.plot([], [], 'o', color='black', markersize=4)
		conn_pts, = ax.plot([], [], 'o', color='green', markersize=4)
		spring_lines = [ax.plot([], [], '-', color='red')[0] for _ in range(self.ne)]  # horizontal springs in red
		link_lines = [ax.plot([], [], '-', color='black')[0] for _ in range(self.ne)]  # all links
		body_top = [ax.plot([], [], 'k-', linewidth=1)[0] for _ in range(self.n)] # Worm body ellipses top
		body_bot = [ax.plot([], [], 'k-', linewidth=1)[0] for _ in range(self.n)] # Worm body ellipses bottom
		
		# Nodes
		q_x = self.q[::3, 0]
		q_y = self.q[::3, 1]

		# Connectors
		# All connectors (top and bottom interleaved)
		c_x = []
		c_y = []
		for i in range(self.n):
			c_x.append(self.q[3*i+1, 0])
			c_y.append(self.q[3*i+1, 1])
			c_x.append(self.q[3*i+2, 0])
			c_y.append(self.q[3*i+2, 1])

		# print(c_x, c_y)
		# print(q_x, q_y)

		# Extract all spring/link endpoints using vectorized indexing
		p1 = self.q[self.springs[:, 0]]  # shape: (n_springs, dim)
		p2 = self.q[self.springs[:, 1]]  # shape: (n_springs, dim)
		
		# Separate horizontal springs (index 5*i) from links (index 5*i+1 to 5*i+4)
		spring_mask = np.zeros(len(self.springs), dtype=bool)
		spring_mask[::5] = True  # every 5th starting at 0 is a horizontal spring
		link_mask = ~spring_mask
		
		plt.plot(q_x, q_y, 'o', color='black')  # Nodes
		plt.plot(c_x, c_y, 'o', color='green')  # Connectors
		
		# Plot horizontal springs (red)
		for i in np.where(spring_mask)[0]:
			plt.plot([p1[i, 0], p2[i, 0]], [p1[i, 1], p2[i, 1]], '-', color='red')
		
		# Plot links (black)
		for i in np.where(link_mask)[0]:
			plt.plot([p1[i, 0], p2[i, 0]], [p1[i, 1], p2[i, 1]], '-', color='black')
		# -------------------
		
		# Worm Profile (ellipses)
		for i in range(len(q_x)-1):
			# Top half-ellipse
			a = (q_x[i] - q_x[i+1]) / 2
			b = (c_y[i] - q_y[i]) / 2
			h, k = (q_x[i+1] + q_x[i])/2, (c_y[i] + q_y[i])/2
			theta = np.linspace(0, np.pi, 300)
			x = h + a * np.cos(theta)
			y = k + b * np.sin(theta)
			plt.plot(x, y)

			# Bottom half-ellipse
			b = (q_y[i] - c_y[i+1]) / 2
			k = (c_y[i+1] + q_y[i])/2
			theta = np.linspace(np.pi, 2*np.pi, 300)
			x = h + a * np.cos(theta)
			y = k + b * np.sin(theta)
			plt.plot(x, y)
	
	def plot(self, ctime=0.0):
		plt.figure(figsize=(8,4))
		self.plot_objects()
		plt.title(f'Worm Configuration, Time: {ctime:.2f} s')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.axis('equal')
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


	def animate_worm(self, frames, times, save_dir="animations", base_name="worm_anim"):
		"""
		Parameters:
		- worm: WormModel instance
		- frames: list of q arrays from the solver
		- times: corresponding time array from the solver
		- save_dir: subdirectory to save GIF
		- base_name: base filename (will append (1), (2), etc. if exists)
		"""

		# Ensure save directory exists
		os.makedirs(save_dir, exist_ok=True)

		# Generate unique filename
		fname = base_name + ".gif"
		counter = 1
		while os.path.exists(os.path.join(save_dir, fname)):
			fname = f"{base_name}({counter}).gif"
			counter += 1
		filepath = os.path.join(save_dir, fname)

		# Figure setup
		fig, ax = plt.subplots(figsize=(8,4))
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_xlim([-1, 2])
		ax.set_ylim([-1, 1])
		
		# Plot objects
		artists = self.plot_objects(ax=ax)

		# Function to initialize animation
		def init():
			artists = self.plot_objects(ax=ax)  # INIT MODE

		# Animation update function
		def update(k):
			self.q = frames[k].reshape(self.q.shape)
			self.plot_objects(ax=ax, artists=artists)  # UPDATE MODE
			return sum([
				[artists["nodes"], artists["conn"]],
				artists["springs"],
				artists["links"],
				artists["top"],
				artists["bot"]
			], [])

		# Calculate fps based on simulation times
		dt_sim = np.mean(np.diff(times))
		fps = max(1, int(1/dt_sim))

		ani = FuncAnimation(fig, update, frames=len(frames),
							init_func=init, blit=True, interval=1000/fps)

		# Save as GIF
		ani.save(filepath, writer=PillowWriter(fps=fps))
		print(f"Animation saved to {filepath}")
		
		return ani