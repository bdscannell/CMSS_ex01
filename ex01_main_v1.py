import numpy as np  # numerical methods library
import matplotlib.pyplot as plt  # plotting library


# find nearest function
def find_nearest(array, value):
    """Function to return index of element in <array> which is numerically closest to <value>"""
    idx = (np.abs(array - value)).argmin()
    return idx

# initial condition definition
def initialBell(x, width=0.5):
    """Function to create initial conditions for an advection-diffusion scheme. \
     x is an evenly-spaced, ascending vector of spatial coordinates from 0 to 1. \
     width [optional, default = 0.5] defines the fraction of x defined with a non-zero initial condition.\
     Inital values based on a sin^2 curve of amplitude 1 over range 0 to pi mapped over width.\
     For width <= 0.5, width mapped to x = 0.5-width : 0.5\
     For width > 0.5, width centred on x = 0.5.\
     \
     Brian Scannell, CMSS September 2017"""
    if width == 0.5:
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


# advection-diffusion scheme with FTCS/CTCS using timestep n-1 for diffusion
def adv_diff(c, d, f, fm=None):
    fp = np.zeros_like(f)
    n = np.shape(f)[0] - 1
    if fm is None:
        for j in xrange(1, n):  # FTCS scheme
            fp[j] = f[j] - 0.5*c*(f[j+1]-f[j-1]) + d*(f[j+1]-2*f[j]+f[j-1])
        # boundary condition
        fp[0] = f[0] - 0.5*c*(f[1]-f[n-1]) + d*(f[1]-2*f[0]+f[n-1])
        fp[n] = fp[0]
    else:
        for j in xrange(1, n):  # CTCS scheme
            fp[j] = fm[j] - c*(f[j+1]-f[j-1]) + 2*d*(fm[j+1]-2*fm[j]+fm[j-1])
        # boundary condition
        fp[0] = fm[0] - c*(f[1]-f[n-1]) + 2*d*(fm[1]-2*fm[0]+fm[n-1])
        fp[n] = fp[0]
    return (fp)

def adv_diff_scenario(T, U, DT, DX, K):
    """Code to execute the advection-diffusion solution for a specific scenario.\

    Brian Scannell, CMSS September 2017"""

    # Derived parameters based on configuration
    t = np.append(np.arange(0, T, T / 5), T)  # Time steps at which to record results
    nx = 1. / DX  # number of points in space
    c = U * DT / DX  # Courant number = u delta(t) / delta(x)
    d = K * DT / np.power(DX, 2)  # non-dimensional diffusivity coefficient = 2.k.delta(t)/delta(x)^2
    nt = int(T / DT)  # number of time steps
    x = np.linspace(0.0, 1.0, int(nx) + 1)  # spatial grid point positions

    # define output arrays
    phi_results = np.zeros((np.size(t), np.size(x)))

    # initial conditions for variable phi at timestep 0
    phi_old = initialBell(x, 0.25)
    phi_results[0, :] = phi_old
    # update phi to timestep 1 using FTCS
    phi = adv_diff(c, d, phi_old)
    # loop over remaining timesteps using CTCS
    for n in xrange(2, nt + 1):
        tn = n * DT
        phi_new = adv_diff(c, d, phi, phi_old)
        phi_old = phi
        phi = phi_new
        # determine whether to save results
        if any(abs(tn - t) < DT):
            ix = find_nearest(t, tn)
            phi_results[ix, :] = phi
    # end of time loop

    return [x, t, c, d, phi_results]

def main():
    """Program written as an exercise durign the NCAS Climate Modelling Summer School 2017.\
    Program solves the linear advection and diffusion equation and evaluates the impacts of\
    varying the Courant number on the integrated quantity after a standard run time.\

    Brian Scannell, CMSS September 2017"""

    # User configuration
    T = 10.         # total model run time in seconds
    U = 0.125        # advection velocity in m/s
    DT = 1./48.    # time step size
    DX = 1./100.    # spatial grid spacing - ring world is only 1m around
    K =  1.14e-3      # diffusivity

    # Run current scenario
    output = adv_diff_scenario(T, U, DT, DX, K)
    x = output[0]
    t = output[1]
    c = output[2]
    d = output[3]
    phi_results = output[4]

    # scale results
    phi_sum = phi_results.sum(axis=1)
    phi_sum = phi_sum/phi_sum[0]

    # Plot the solution in comparison with the analytic solution
    for ix in range(np.size(phi_sum)):
        plt.plot(x,phi_results[ix,:], label = '$t=%.0fs,\ \ \Sigma\phi=%.2f$' % (t[ix],phi_sum[ix]))

#    plt.plot(x, initialBell(x, 0.25), 'r', label='t=0')
#    plt.plot(x, phi, 'b', label='CTCS')
    legend = plt.legend(loc='upper right', fontsize=11)
    legend.get_frame().set_edgecolor('none')
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$\phi$', fontsize=16)
    plt.title('$c=%.3f,\ \ D=%.3f,\ \ c^2+4d=%.3f$' % (c, d, c**2+4*d), fontsize=18)
    plt.axhline(0, linestyle=':', color='black')
    plt.grid()
    plt.show()

# run the program
main()
