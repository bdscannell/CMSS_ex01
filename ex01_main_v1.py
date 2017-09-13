import numpy as np  # numerical methods library
import matplotlib.pyplot as plt  # plotting library


# find nearest function
def find_nearest(array, value):
    """Function to return index of element in array which is numerically closest to value"""
    idx = (np.abs(array - value)).argmin()
    return idx


# initial condition definition
def initialBell_old(x):
    return np.where(x % 1. < 0.5, np.power(np.sin(2 * x * np.pi), 2), 0.)


# initial condition definition
def initialBell(x, width=None):
    if width is None:
        initialCond = np.where(x % 1. < 0.5, np.power(np.sin(2 * x * np.pi), 2), 0.)
        return initialCond
    else:
        x = x % 1.
        initialCond = np.zeros(np.shape(x))
        if width > 1:
            width = width % 1.
        # end
        if width <= 0.5:
            ix1 = find_nearest(x, 0.5 - width)
            ix2 = find_nearest(x, 0.5)
            initialCond[ix1:ix2 + 1] = np.power(np.sin(np.arange(0, ix2 - ix1 + 1, 1.) * (np.pi / (ix2 - ix1))), 2)
        else:
            ix1 = find_nearest(x, 0.5 - (width / 2))
            ix2 = find_nearest(x, x[ix1] + width)
            initialCond[ix1:ix2 + 1] = np.power(np.sin(np.arange(0, ix2 - ix1 + 1, 1.) * (np.pi / (ix2 - ix1))), 2)
        return initialCond


# alternative initial condition definition
def initialSin3(x):
    return np.where(x % 1. < 0.5, np.power(np.sin(4 * x * np.pi), 3), 0.)


# loop over space function
def adv_diff(c, d, f, fm=None):
    fp = np.zeros_like(f)
    n = np.shape(f)[0] - 1
    if fm is None:
        for j in xrange(1, n):  # FTCS scheme
            fp[j] = f[j] - 0.5*c*(f[j+1]-f[j-1]) + 0.5*d*(f[j+1]-2*f[j]+f[j-1])
        # boundary condition
        fp[0] = f[0] - 0.5*c*(f[1]-f[n-1]) + 0.5*d*(f[1]-2*f[0]+f[n-1])
        fp[n] = fp[0]
    else:
        for j in xrange(1, n):  # CTCS scheme
            fp[j] = fm[j] - c*(f[j+1]-f[j-1]) + d*(fm[j+1]-2*fm[j]+fm[j-1])
        # boundary condition
        fp[0] = fm[0] - c*(f[1]-f[n-1]) + d*(fm[1]-2*fm[0]+fm[n-1])
        fp[n] = fp[0]
    return (fp)


def main():
    """Program written as an exercise durign the NCAS Climate Modelling Summer School 2017. /
    Program solves the linear advection and diffusion equation and evaluates the impacts of /
    varying the Courant number on the integrated quantity after a standard run time./

    Brian Scannell September 2017 """

    # Setup, space, initial phi profile and Courant number
    nx = 100  # number of points in space
    c = 0.2  # Courant number = u delta(t) / delta(x)
    D = 0.2  # non-dimensional diffusivity coefficient = 2.k.delta(t)/delta(x)^2
    nt = 200  # number of time steps
    x = np.linspace(0.0, 1.0, nx + 1)  # spatial variable going from zero to one inclusive

    # initial conditions for variable phi at timestep 0
    phi_old = initialBell(x, 0.25)
    # update phi to timestep 1 using FTCS
    phi = adv_diff(c,D,phi_old)
    # loop over remaining timesteps using CTCS
    for n in xrange(1, nt):
        phi_new = adv_diff(c, D, phi, phi_old)
        phi_old = phi
        phi = phi_new
    # end of time loop

    # derived quantities
    u = 1.
    dx = 1. / nx
    dt = c * dx / u
    t = nt * dt

    # Plot the solution in comparison with the analytic solution
    #plt.plot(x, initialBell(x - u * t, 0.25), '*k', label='analytic')
    plt.plot(x, initialBell(x, 0.25), 'r', label='analytic, t=0')
    plt.plot(x, phi, 'b', label='CTCS')
    plt.legend(loc='best')
    plt.xlabel('$x$')
    plt.ylabel('$\phi$')
    plt.axhline(0, linestyle=':', color='black')
    plt.show()


main()
