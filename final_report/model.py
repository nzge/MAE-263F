import numpy as np
import matplotlib.pyplot as plt
import parameters as param 

class WormModel:
	def __init__(self, length, n_segments):
		self.n = n_segments
		self.length = length
		self.deltaL = length / n_segments
		
		#Geometry#
		# Nodes
		self.nodes = np.zeros((n_segments+1, 2))
		self.nodes[0] = [0.0, 0.0]
		for i in range(1, n_segments+1):
				self.nodes[i, 0] = i * self.deltaL
				self.nodes[i, 1] = 0.0
		
		# Springs
		dt = np.dtype([("p1", "f8", (2,)), ("p2", "f8", (2,)), ("k", "f8"), ("l0", "f8")])
		self.springs = np.zeros(n_segments, dtype=dt)
		k_s = 13 # segment spring stiffness
		for i in range(n_segments):
				self.springs["p1"][i] = self.nodes[i]
				self.springs["p2"][i] = self.nodes[i+1]
				self.springs["k"][i] = k_s
				self.springs["l0"][i] = self.deltaL
		
		# Connectors & links
		self.connectors = np.zeros((n_segments*2, 2))
		self.links = np.zeros(n_segments*4, dtype=dt)
		k_l = 1e4
		for i in range(n_segments):
				x_pos = (self.nodes[i][0] + self.nodes[i+1][0]) / 2
				y_pos = np.sqrt((self.deltaL/2)**2 - (self.deltaL/4)**2)
				n1, n2 = self.nodes[i], self.nodes[i+1]
				top_connector = [x_pos, y_pos]
				bottom_connector = [x_pos, -y_pos]
				self.connectors[2*i] = top_connector
				self.connectors[2*i+1] = bottom_connector
				l0 = np.sqrt( (self.deltaL/2)**2 + (y_pos)**2 )

				self.links[4*i] = (n1, top_connector, k_l, l0)
				self.links[4*i+1] = (top_connector, n2, k_l, l0)
				self.links[4*i+2] = (n1, bottom_connector, k_l, l0)
				self.links[4*i+3] = (bottom_connector, n2, k_l, l0)
		

		self.nv = self.nodes.shape[0] + self.connectors.shape[0]  # nodes + connectors
		self.ne = self.springs.shape[0] + self.links.shape[0]      # springs + links
		self.ndof = 2 * self.nv                                    # 2D, x and y per vertex

		# Initial conditions
		q0, _ , _ , _ = self.get_internal_state()
		self.q0 = q0 # initial position
		u0 = np.zeros(2 * self.nv) # old velocity
		self.u0 = u0 # initial velocity

		# Boundary Conditions
		all_DOFs = np.arange(self.ndof) # Set of all DOFs
		fixed_index = np.array([ ]) # Fixed DOFs
		free_index = np.setdiff1d(all_DOFs, fixed_index) # All the DOFs are free except the fixed ones
		self.freeIndex = free_index
		self.fixedIndex = fixed_index

		isFixed = np.zeros(self.nv)
		self.isFixed = isFixed

		#-----------Forces------------#
		# Radii of spheres (given)
		R = np.zeros(self.nv)
		for k in range(self.nv):
			R[k] = self.deltaL/10 # meter

		# Mass vector and matrix
		m = np.zeros( 2 * self.nv )
		for k in range(0, self.nv):
			m[2*k] = 4/3 * np.pi * R[k]**3 * param.rho_metal # mass of k-th node along x
			m[2*k + 1] = 4/3 * np.pi * R[k]**3 * param.rho_metal # mass of k-th node along y
		self.m = m
		self.mMat = np.diag(m)
		
		# Gravity (external force)
		W = np.zeros( 2 * self.nv)
		g = np.array([0, -9.8]) # m/s^2
		for k in range(0, self.nv):
			W[2*k] = m[2*k] * g[0] # Weight along x
			W[2*k+1] = m[2*k+1] * g[1] # Weight along y
		# Gradient of W = 0
		self.W = W

		# Viscous damping (external force)
		C = np.zeros((2 * self.nv, 2 * self.nv))
		for k in range(0, self.nv):
			C[2*k, 2*k] = 6.0 * np.pi * param.visc * R[k] # Damping along x for k-th node
			C[2*k+1, 2*k+1] = 6.0 * np.pi * param.visc * R[k] # Damping along y for k-th node
		self.c = C
		
	def update_internal_state(self, q):
		for i in range(self.n):
			# Update node positions
				n1 = q[3*i: 3*i+2]
				n2 = q[3*(i+1): 3*(i+1)+2]
				self.nodes[i] = n1
				self.nodes[i+1] = n2
			# Update connector positions
				top_connector = q[3*i + 2 : 3*i + 4]
				bottom_connector = q[3*i + 4 : 3*i + 6]
				self.connectors[2*i] = top_connector
				self.connectors[2*i+1] = bottom_connector
			#Update springs 
				self.springs["p1"][i] = n1
				self.springs["p2"][i] = n2
			#Update links
				self.links[4*i]["p1"] = n1
				self.links[4*i]["p2"] = top_connector
				self.links[4*i+1]["p1"] = top_connector
				self.links[4*i+1]["p2"] = n2
				self.links[4*i+2]["p1"] = n1
				self.links[4*i+2]["p2"] = bottom_connector
				self.links[4*i+3]["p1"] = bottom_connector
				self.links[4*i+3]["p2"] = n2

	def get_internal_state(self, k=None):
		q = np.zeros(self.ndof)
		stiffness_matrix = np.zeros((self.ne))
		spring_index_matrix = np.zeros((self.ne, 4))
		l0 = np.zeros(self.ne)
			
		for i in range(self.n):
			# Get node positions
			n1 = self.nodes[i]
			n2 = self.nodes[i+1] 
			q[3*(i+1): 3*(i+1)+2] = n1
			q[3*i: 3*i+2] = n2
			# get connector positions
			top_connector = self.connectors[2*i]
			bottom_connector = self.connectors[2*i+1]
			q[3*i + 2 : 3*i + 4] = top_connector 
			q[3*i + 4 : 3*i + 6] = bottom_connector
			# get spring info
			stiffness_matrix[5*i] = self.springs["k"][i]
			spring_index_matrix[5*i] = np.hstack([self.springs["p1"][i], self.springs["p2"][i]])
			l0[5*i] = self.springs["l0"][i]
			# get link info
			stiffness_matrix[5*i+1] = self.links["k"][4*i]
			spring_index_matrix[5*i+1] = np.hstack([self.links["p1"][4*i], self.links["p2"][4*i]])
			l0[5*i+1] = self.links["l0"][4*i]
			stiffness_matrix[5*i+2] = self.links["k"][4*i+1]
			spring_index_matrix[5*i+2] = np.hstack([self.links["p1"][4*i+1], self.links["p2"][4*i+1]])
			l0[5*i+2] = self.links["l0"][4*i+1]
			stiffness_matrix[5*i+3] = self.links["k"][4*i+2]
			spring_index_matrix[5*i+3] = np.hstack([self.links["p1"][4*i+2], self.links["p2"][4*i+2]])
			l0[5*i+3] = self.links["l0"][4*i+2]
			stiffness_matrix[5*i+4] = self.links["k"][4*i+3]
			spring_index_matrix[5*i+4] = np.hstack([self.links["p1"][4*i+3], self.links["p2"][4*i+3]])
			l0[5*i+4] = self.links["l0"][4*i+3]

		if k is not None:
			return spring_index_matrix
		return q, stiffness_matrix, spring_index_matrix, l0

	def plot(self):
		# -------------------
		# Nodes
		q_x = self.nodes[:, 0]
		q_y = self.nodes[:, 1]

		# Connectors
		c_x = self.connectors[:, 0]
		c_y = self.connectors[:, 1]

		# Springs: extract endpoints
		s_lines = np.array([[self.springs[i]["p1"], self.springs[i]["p2"]] for i in range(len(self.springs))])
		s_x = s_lines[:, :, 0]
		s_y = s_lines[:, :, 1]

		# Links: same as springs
		l_lines = np.array([[self.links[i][0], self.links[i][1]] for i in range(len(self.links))])
		l_x = l_lines[:, :, 0]
		l_y = l_lines[:, :, 1]
		
		plt.figure(figsize=(8,4))
		plt.plot(q_x, q_y, 'o', color='black')  # Nodes
		plt.plot(c_x, c_y, 'o', color='green')  # Connectors
		for i in range(len(s_lines)):
				plt.plot(s_x[i], s_y[i], '-', color='red')  # Springs
		for i in range(len(l_lines)):
				plt.plot(l_x[i], l_y[i], '-', color='black')  # Links
		# -------------------
		
		# Worm Profile (ellipses)
		for i in range(len(q_x)-1):
				# Top half-ellipse
				a = (q_x[i] - q_x[i+1]) / 2
				b = (c_y[2*i] - q_y[i]) / 2
				h, k = (q_x[i+1] + q_x[i])/2, (c_y[2*i] + q_y[i])/2
				theta = np.linspace(0, np.pi, 300)
				x = h + a * np.cos(theta)
				y = k + b * np.sin(theta)
				plt.plot(x, y)

				# Bottom half-ellipse
				b = (q_y[i] - c_y[2*i+1]) / 2
				k = (c_y[2*i+1] + q_y[i])/2
				theta = np.linspace(np.pi, 2*np.pi, 300)
				x = h + a * np.cos(theta)
				y = k + b * np.sin(theta)
				plt.plot(x, y)
		
		plt.title('Worm Configuration')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.axis('equal')
		plt.show()
