from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

class Lorenz(ABC):
    def __init__(self, tmax, X0, num_samples):
        self.tmax = tmax
        self.X0 = X0
        self.num_samples = num_samples
        self.trajectory = self._integrate() # Shape (*, num_samples)
    
    @abstractmethod
    def _model(self, t, X):
        """Define Lorenz model"""
        ...
    
    def _integrate(self):
        soln = solve_ivp(self._model, (0, self.tmax), self.X0, dense_output=True)
        t = np.linspace(0, self.tmax, self.num_samples)
        return soln.sol(t)

    @abstractmethod
    def plot(self):
        """Method to plot trajectory of Lorenz system"""
        ...

class Lorenz63(Lorenz):
    def __init__(self, tmax, X0, num_samples):
        self._sigma = 10.
        self._beta = 8/3
        self._rho = 28
        super().__init__(tmax, X0, num_samples)

    def _model(self, t, X):
        """Lorenz '63 model
        Args:
            t: float
            X: array of shape (3,)
        """
        x, y, z = X
        dxdt = self._sigma*(y - x)
        dydt = x*(self._rho - z) - y
        dzdt = x*y - self._beta*z
        return dxdt, dydt, dzdt

    def plot(self):
        x, y, z = self.trajectory
        fig = plt.figure(figsize=(9, 6.5))
        ax = fig.gca(projection='3d')
        ax.plot(x,y,z)
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel('z', fontsize=20)
        ax.set_title('The Lorenz Attractor', fontsize=20)

class Lorenz96(Lorenz):
    ...


