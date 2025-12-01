import numpy as np
import matplotlib.pyplot as plt
import importlib

import parameters as param
importlib.reload(param)

def getContract(worm, t=None, T_wave=2.0, wavelength=1.0, wave_type='traveling'):
    """
    Return external contractile force vector for peristaltic locomotion.
    
    Parameters:
    -----------
    worm : WormModel
        The worm model instance
    t : float
        Current simulation time (default 0.0)
    Fmax : float
        Maximum contractile force magnitude
    T_wave : float
        Wave period (time for one complete wave cycle)
    wavelength : float
        Spatial wavelength as fraction of body length (1.0 = one wave spans entire body)
    wave_type : str
        'traveling' - wave propagates head to tail (peristaltic)
        'standing' - stationary wave with oscillating amplitude
        'retrograde' - wave propagates tail to head
    
    Returns:
    --------
    F_contract : ndarray
        Force vector of size worm.ndof
    """

    if t is None:
        t = 0.0

    F_contract = np.zeros(worm.ndof)
    n_seg = worm.n
    
    # Angular frequency
    omega = 2.0 * np.pi / T_wave
    # Wave number (spatial frequency)
    k = 2.0 * np.pi / wavelength
    
    for i in range(n_seg):
        # Normalized position along body (0 = head, 1 = tail)
        x_norm = (i + 0.5) / n_seg  # center of segment
        
        # Compute wave phase based on wave type
        if wave_type == 'traveling':
            # Wave travels from head (x=0) to tail (x=1)
            phase = k * x_norm - omega * t
            amplitude = np.sin(phase)
        elif wave_type == 'retrograde':
            # Wave travels from tail to head
            phase = k * x_norm + omega * t
            amplitude = np.sin(phase)
        elif wave_type == 'standing':
            # Standing wave: sin(kx) * cos(ωt)
            amplitude = np.sin(k * x_norm) * np.cos(omega * t)
        else:
            raise ValueError(f"Unknown wave_type: {wave_type}")
        
        # Convert to contraction force (only contract, don't expand beyond rest)
        # Use rectified sine for unidirectional contraction
        F_seg = worm.Fmax * max(0, amplitude)
        
        # Alternative: bidirectional (both squeeze and expand)
        # F_seg = Fmax * amplitude
        
        # Get DOF indices for this segment's connectors
        top_conn_idx = 3 * i + 1  # node index
        bot_conn_idx = 3 * i + 2  # node index
        
        # Y-DOF indices (assuming 2D: DOF = 2*node_idx for x, 2*node_idx+1 for y)
        top_y_dof = 2 * top_conn_idx + 1
        bot_y_dof = 2 * bot_conn_idx + 1
        
        # Apply contractile forces (squeeze the segment)
        # Top connector: force DOWN (negative y)
        # Bottom connector: force UP (positive y)
        if top_y_dof < worm.ndof and bot_y_dof < worm.ndof:
            F_contract[top_y_dof] = -F_seg  # push top down
            F_contract[bot_y_dof] = +F_seg  # push bottom up

    return F_contract

def getContract_single_segment(worm, t=None):
    """ Original simple single-segment sinusoidal contraction (kept for reference). """
    if t is None:
        t = 0.0
    F = 0.8 * np.sin(2.0 * np.pi * (t/2))
    F_contract = np.zeros(worm.ndof)
    
    try:
        F_contract[worm.dim * 1 + 1] = -F
        F_contract[worm.dim * 2 + 1] = F
    except Exception:
        pass
    return F_contract

class ContractionEngine_segmentdriven:
    def __init__(self, n_segments, T_contraction, T_wave, Fmax, mu, sigma, pulse_type):
        self.n = n_segments
        self.T_contraction = T_contraction
        self.T_wave = T_wave

        # Contraction parameters
        self.Fmax = Fmax
        self.mu = mu
        self.sigma = sigma
        self.pulse_type = pulse_type

        # State arrays
        self.phase = np.zeros(n_segments, dtype=float)
        self.active = np.zeros(n_segments, dtype=bool)
        self.activated_this_wave = np.zeros(n_segments, dtype=bool)

    # Step 1: activate segments based on wave
    def set_activation_wave(self, t):
        scale = int(round(1 / param.dt))
        t_int, T_wave_int = int(round(t * scale)), int(round(self.T_wave * scale))
        phase_norm = np.mod(t_int, T_wave_int) / T_wave_int
        segment_pos = np.arange(self.n, dtype=float) / self.n

        for i, seg in enumerate(segment_pos):
            if (phase_norm >= seg) and (not self.activated_this_wave[i]):
                self.active[i] = True
                self.activated_this_wave[i] = True
                break  # only first segment

    # Step 2: increment phase and deactivate over-threshold segments
    def update_phase(self):
        self.phase[self.active] += param.dt
        self.phase[~self.active] = 0.0
        over_threshold = self.phase >= self.T_contraction
        self.phase[over_threshold] = 0.0
        self.active[over_threshold] = False

    # Step 3: pulse generator
    def pulse(self):
        if self.pulse_type == 'gaussian':
            return self.Fmax * np.exp(-0.5 * ((self.phase - self.mu) / self.sigma)**2)
        elif self.pulse_type == 'square':
            norm_phase = self.phase / self.T_contraction
            pulse_vals = np.zeros_like(self.phase)
            pulse_vals[norm_phase <= 0.2] = self.Fmax  # first 20% of contraction
            return pulse_vals
        elif self.pulse_type == 'dirac':
            pulse_vals = np.zeros_like(self.phase)
            pulse_vals[self.phase == 0.0] = self.Fmax  # instantaneous at start
            return pulse_vals
        else:
            raise ValueError("Unknown pulse type")

    # Step 4: compute forces for all segments
    def compute_forces(self):
        pulse_vals = self.pulse()
        pulse_vals[~self.active] = 0.0

        Fg = np.zeros(self.n * 2)
        Fg[0::2] = pulse_vals  # negative force
        Fg[1::2] = -pulse_vals   # positive force
        return Fg
