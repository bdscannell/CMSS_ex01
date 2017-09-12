import numpy as np # numerical methods library
import matplotlib.pyplot as plt # plotting library

# function defining the initial analytic solution
def initialBell(x):
    return np.where(x%1. < 0.5, np.power(np.sin(2*x*np.pi), 2), 0.)

# alternative forcing function
def initialSin(x):
    return np.where(x%1. < 0.5, np.sin(4*x*np.pi), 0.)

# loop over space function
def loopSpace(n,courant,phiN,phiNminus = None):
    phiNplus = np.zeros_like(phiN)
    if phiNminus is None:
        for j in xrange(1, n):  # FTCS scheme
            phiNplus[j] = phiN[j] - 0.5 * c * (phiN[j + 1] - phiN[j - 1])
        # boundary condition
        phiNplus[0] = phiN[0] - 0.5 * c * (phiN[1] - phiN[n - 1])
        phiNplus[n] = phiNplus[0]
    else:
        for j in xrange(1, n):  # CTCS scheme
            phiNplus[j] = phiNminus[j] - courant * (phiN[j + 1] - phiN[j - 1])
        # boundary condition
        phiNplus[0] = phiNminus[0] - courant * (phiN[1] - phiN[n - 1])
        phiNplus[n] = phiNplus[0]
    return(phiNplus)

def main:
    '''Program to solve the linear advection equation using a choice of numerical schemes'''


# Setup, space, initial phi profile and Courant number
nx = 100     # number of points in space
c = 0.2     # Courant number
nt = 200    # number of time steps
x = np.linspace(0.0, 1.0, nx+1) # spatial variable going from zero to one inclusive

# initial conditions for variable phi at timestep 0
phi_old = initialSin(x)
# update phi to timestep 1 using FTCS
phi = loopSpace(nx,c,phi_old)
# loop over remaining timesteps using CTCS
for n in xrange(1,nt):
    phi_new = loopSpace(nx,c,phi,phi_old)
    phi_old = phi
    phi = phi_new


# derived quantities
u = 1.
dx = 1./nx
dt = c*dx/u
t = nt*dt

# Plot the solution in comparison with the analytic solution
plt.plot(x, initialSin(x - u*t), '*k', label='analytic')
plt.plot(x, initialSin(x), 'r', label='analytic, t=0')
plt.plot(x, phi, 'b', label='CTCS')
plt.legend(loc='best')
plt.xlabel('$x$')
plt.ylabel('$\phi$')
plt.axhline(0, linestyle=':', color='black')
plt.show()
