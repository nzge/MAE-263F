import numpy as np
import matplotlib.pyplot as plt
import importlib

import parameters as param
importlib.reload(param)

def getFexternal(t=None, ndof=None):
    """
    Return external contractile force vector.
    Allows calling either getFexternal(t, m) or getFexternal(m).
    """
    # Support call with single argument getFexternal(m)
    if t is None:
        t = 0.0

    F = 1.25 * np.sin(2.0 * np.pi * (t/5))  # Contractile force varying with time
    F_contract = np.zeros(ndof)

    # Apply forces to the y-DOFs of node 0 and node 3 if indices exist
    try:
        F_contract[2 * 1 + 1] = -F
        F_contract[2 * 3 + 1] = F
    except Exception:
        # If vector is too small, silently skip assignment
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
