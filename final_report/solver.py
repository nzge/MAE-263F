import numpy as np
import matplotlib.pyplot as plt

class TimeStepper:
    def __init__(self, q0, u0, worm, param, mMat, obj, dt, totalTime, groundPosition):
        self.q = q0.copy()
        self.u = u0.copy()
        self.worm = worm
        self.param = param
        self.mMat = mMat
        self.obj = obj
        self.dt = dt
        self.totalTime = totalTime
        self.time = 0.0
        self.Nsteps = round(totalTime / dt)
        self.groundPosition = groundPosition

        # Constraints / contact state
        self.isFixed = np.zeros_like(q0[::2])  # or your existing version
        self.free_index = getFreeIndex(self.isFixed, worm)

    def predictor_corrector(self):
        # 1. Predictor
        q_guess = self.q.copy()
        q_new, error, reactionForce = self.obj.objfun(
            q_guess, self.q, self.u, self.dt, self.param.tol, 
            self.param.maximum_iter, self.param.m, self.mMat, self.param.EI,
            self.param.EA, self.param.W, self.param.C, 
            self.worm.deltaL, self.free_index
        )
        if error < 0:
            raise RuntimeError("Predictor failed to converge")

        # 2. Corrector logic (ground/contact)
        needCorrector = False
        for k in range(2, len(self.isFixed)):
            yk = q_new[2*k + 1]
            fk = reactionForce[2*k + 1]

            # Contact condition
            if self.isFixed[k] == 0 and yk < self.groundPosition:
                needCorrector = True
                self.isFixed[k] = 1
                q_guess[2*k + 1] = self.groundPosition

            # Release condition
            elif self.isFixed[k] == 1 and fk < 0:
                needCorrector = True
                self.isFixed[k] = 0

        # 3. If needed, re-solve
        if needCorrector:
            self.free_index = getFreeIndex(self.isFixed, self.worm)
            q_new, error, reactionForce = self.obj.objfun(
                q_guess, self.q, self.u, self.dt, self.param.tol,
                self.param.maximum_iter, self.param.m, self.mMat, self.param.EI,
                self.param.EA, self.param.W, self.param.C, 
                self.worm.deltaL, self.free_index
            )
        return q_new

    def step(self):
        q_new = self.predictor_corrector()
        u_new = (q_new - self.q) / self.dt

        # Update state
        self.q = q_new
        self.u = u_new
        self.time += self.dt

    def run(self, plotStep=20):
        for step in range(self.Nsteps):
            self.step()
            if step % plotStep == 0:
                self.plot()

    def plot(self):
        x_arr = self.q[::2]
        y_arr = self.q[1::2]

        plt.clf()
        plt.plot(x_arr, y_arr, "ko-")
        plt.axis("equal")
        plt.title(f"t = {self.time:.4f}s")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.001)

def getFreeIndex(isFixed, worm):
  # isFixed is a 0 or 1 vector of size nv
  # free_index is the output of size ndof = 2 * nv
  worm.nv = len(isFixed) # Number of vertices
  ndof = 2 * worm.nv
  all_DOFs = np.zeros(worm.ndof) # Set of all DOFs -- all DOFs are free

  # Hard code the clamp condition
  all_DOFs[0:4] = 1 # Fix the x-coordinate (left wall)

  for k in range(worm.nv):
    if isFixed[k] == 1:
      # all_DOFs[2*k] = 1
      all_DOFs[2*k+1] = 1 # Fix the y-coordinate (ground)
  free_index = np.where(all_DOFs == 0)[0]
  return free_index