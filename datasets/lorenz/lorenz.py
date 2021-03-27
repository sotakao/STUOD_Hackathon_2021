import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from copy import deepcopy
from tqdm import tqdm

class Lorenz63():
    def __init__(self, tmax, X0, num_samples):
        self._sigma = 10.
        self._beta = 8/3
        self._rho = 28.
        self.params = (self._sigma, self._beta, self._rho)
        self.num_samples = num_samples
        self.trajectory = self._integrate(tmax, X0).transpose() #Size (num_samples, 3)

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

    def _integrate(self, tmax, X0):
        soln = solve_ivp(self._model, (0, tmax), X0, dense_output=True)
        t = np.linspace(0, tmax, self.num_samples)
        return soln.sol(t)

    def plot(self):
        x, y, z = self.trajectory.transpose()
        fig = plt.figure(figsize=(9, 6.5))
        ax = fig.gca(projection='3d')
        ax.plot(x,y,z)
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel('z', fontsize=20)
        ax.set_title('The Lorenz Attractor', fontsize=20)
        plt.show()

class Lorenz96():
    def __init__(self, tmax, X0, Y0, num_samples):
        self._h = 1
        self._c = 10
        self._b = 10
        self._F = 10
        self._K = X0.shape[0]
        self._J = round(Y0.shape[0]/self._K)
        self._X = deepcopy(X0)
        self._Y = deepcopy(Y0)
        self.num_samples = num_samples
        self.tmax = tmax
        self.dt = tmax/num_samples
        self.params = (self._h, self._c, self._b, self._F)
        self.trajectory = {'X': X0[None], 'Y': Y0[None]}
        self._integrate()
    
    def _model(self, X, Y):
        """Lorenz '96 model
        Args:
            t: float
            X: array of shape (K,)
            Y: array of shape (J*K,)
        """
        h, c, b, F, K, J = *self.params, self._K, self._J
        dXdt = -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) - X + F - h * c * Y.reshape(K, J).mean(1)
        dYdt = c * (b * np.roll(Y, 1) * (np.roll(Y, 2) - np.roll(Y, -1)) - Y + (h / J) * np.repeat(X, J))
        return (dXdt, dYdt)

    
    def step(self):
        """One time step of RK4 scheme"""
        k1_X, k1_Y = self._model(self._X, self._Y)
        k2_X, k2_Y = self._model(self._X + self.dt * k1_X / 2, self._Y + self.dt * k1_Y / 2)
        k3_X, k3_Y = self._model(self._X + self.dt * k2_X / 2, self._Y + self.dt * k2_Y / 2)
        k4_X, k4_Y = self._model(self._X + self.dt * k3_X, self._Y + self.dt * k3_Y)

        self._X += (1 / 6) * self.dt * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        self._Y += (1 / 6) * self.dt * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        self.trajectory['X'] = np.append(self.trajectory['X'], self._X[None], axis=0)
        self.trajectory['Y'] = np.append(self.trajectory['Y'], self._Y[None], axis=0)
    
    def _integrate(self):
        for _ in tqdm(range(self.num_samples), disable=False):
            self.step()

    def plot(self):
        """Plot the first 5 modes of X and Y"""
        X = self.trajectory['X'][:,:5]
        Y = self.trajectory['Y'][:,:5]
        timestamps = np.arange(0, self.tmax + self.dt, self.dt)
        fig = plt.figure(figsize=(17, 3))
        ax = plt.subplot(1, 2, 1)
        for i in range(5):
            plt.plot(timestamps, X[:,i], label=f'X{i+1}')
            ax.set_xlabel('time', fontsize=15)
            ax.set_ylabel('x', fontsize=15)
            ax.set_title('Slow modes (X)', fontsize=15)
            ax.legend()
        ax = plt.subplot(1, 2, 2)
        for i in range(5):
            plt.plot(timestamps, Y[:,i], label=f'Y{i+1}')
            ax.set_xlabel('time', fontsize=15)
            ax.set_ylabel('y', fontsize=15)
            ax.set_title('Fast modes (Y)', fontsize=15)
            ax.legend()
        plt.show()


