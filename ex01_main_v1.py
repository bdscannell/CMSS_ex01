import numpy as np # numerical methods library
import matplotlib.pyplot as plt # plotting library

# function defining the initial analytic solution
def initialBell(x):
    return np.where(x%1. < 0.5, np.power(np.sin(2*x*np.pi), 2), 0.)

# alternative forcing function
def initialSin(x):
    return np.where(x%1. < 0.5, np.sin(2*x*np.pi), 0.)

# loop over space function
def loopSpace(nx,courant,phiN,phiNminus = None):
    phiNplus = np.zeros_like(phiN)
    if phiNminus:
        for j in xrange(1,nx): # CTCS scheme
            phiNplus[j] = phiNminus[j] - courant * (phiN[j + 1] - phiN[j - 1])
        # boundary condition
        phiNplus[0] = phiNminus[0] - courant * (phiN[1] - phiN[nx - 1])
        phiNplus[nx] = phiNplus[0]
    else:
        for j in xrange(1, nx):  # FTCS scheme
            phiNplus[j] = phiN[j] - 0.5 * c * (phiN[j + 1] - phiN[j - 1])
        # boundary condition
        phiNplus[0] = phiN[0] - 0.5 * c * (phiN[1] - phiN[nx - 1])
        phiNplus[nx] = phiNplus[0]
    return(phiNplus)

# Setup, space, initial phi profile and Courant number
nx = 100     # number of points in space
c = 0.2     # Courant number
# spatial variable going from zero to one inclusive
x = np.linspace(0.0, 1.0, nx+1)
# three time levels of the dependent variable, phi
phi = initialBell(x)
phiNew = phi.copy()
phiOld = phi.copy()

# # FTCS for the first time-step
# # phi = loopSpace(nx,c,phi)
# # loop over space
# for j in xrange(1,nx): # don't forget that indices are (inclusive, exclusive) so 1 to nx-1
#     phi[j] = phiOld[j] - 0.5*c*(phiOld[j+1] - phiOld[j-1])
#
# # apply periodic boundary condition
# phi[0] = phiOld[0] - 0.5*c*(phiOld[1] - phiOld[nx-1])
# phi[nx] = phi[0]
#
# # Loop over remaining time steps (nt) using CTCS
# nt = 600
# for n in xrange(1,nt):
#     # loop over space
#     for j in xrange(1,nx):
#         phiNew[j] = phiOld[j] - c*(phi[j+1] - phi[j-1])
#     # apply periodic boundary condition
#     phiNew[0] = phiOld[0] - c*(phi[1] - phi[nx - 1])
#     phiNew[nx] = phiNew[0]
#
#     #update phi for the next time-step
#     phiOld = phi.copy()
#     phi = phiNew.copy()
#
# # derived quantities
# u = 1.
# dx = 1./nx
# dt = c*dx/u
# t = nt*dt
#
# # Plot the solution in comparison with the analytic solution
# plt.plot(x, initialBell(x - u*t), '*k', label='analytic')
# plt.plot(x, initialBell(x), 'r', label='analytic, t=0')
# plt.plot(x, phi, 'b', label='CTCS')
# plt.legend(loc='best')
# plt.xlabel('x')
# plt.ylabel('$\phi$')
# plt.axhline(0, linestyle=':', color='black')
# plt.show()
plt.plot(x,phi)
plt.show()