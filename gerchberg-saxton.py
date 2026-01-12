import numpy as np
import matplotlib.pyplot as plt
import math

#========================
# spatial grid
#========================

N = 2**10          # grid resolution
dx = 8e-6          # pixel size (meters)

x = np.linspace(-N/2*dx, N/2*dx, N)
y = np.linspace(-N/2*dx, N/2*dx, N)
X, Y = np.meshgrid(x, y)

Rho = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)

R_ap = (N/2) * dx
rho = Rho / R_ap
mask = rho <= 1

#========================
# Zernike functions
#========================

def R_nm(n, m, rho):
    """Zernike radial polynomial."""
    R = np.zeros_like(rho)
    for k in range((n - m)//2 + 1):
        num = (-1)**k * math.factorial(n - k)
        den = (math.factorial(k) *
               math.factorial((n + m)//2 - k) *
               math.factorial((n - m)//2 - k))
        R += num / den * rho**(n - 2*k)
    return R

def Z_nm(n, m, rho, phi):
    """Full Zernike mode"""
    Z = np.zeros_like(rho)
    if m > 0:
        Z[mask] = R_nm(n, m, rho[mask]) * np.cos(m * phi[mask])
    elif m < 0:
        Z[mask] = R_nm(n, -m, rho[mask]) * np.sin(-m * phi[mask])
    else:
        Z[mask] = R_nm(n, 0, rho[mask])
    return Z

#========================
# simulate Gaussian field with Zernike phase
#========================

sigma = 9e-4
U = np.exp(-(X**2 + Y**2) / sigma**2)

Z = Z_nm(n=3, m=1, rho=rho, phi=Phi)
alpha = 1*np.pi
phase = alpha * Z
U_complex = (U * mask) * np.exp(1j * phase)

# Measurement planes
A1 = np.abs(U_complex)
U2 = np.fft.fft2(U_complex, norm="ortho")
A2 = np.abs(U2)

# Save simulated measurements
np.save("./Simulations/A1.npy", A1)
np.save("./Simulations/A2.npy", A2)
np.save("./Simulations/phi_true.npy", np.angle(U_complex))


#========================
# Gerchberg Saxto
#========================

def gerchberg_saxton(A1, A2, n_iters=200, seed=0, beta=0.8, use_hio=False, gamma=0.9):
    """
    gerchberg saxton
    """
    rng = np.random.default_rng(seed)
    phi0 = rng.uniform(-np.pi, np.pi, A1.shape)
    U1 = A1 * np.exp(1j * phi0)

    U1_old = U1.copy()
    errors = []

    for i in range(n_iters):

        # Forward propagation
        U2_tmp = np.fft.fft2(U1, norm="ortho")

        # Relaxed amplitude constraint in plane 2
        mag = beta*A2 + (1-beta)*np.abs(U2_tmp)
        U2 = mag * np.exp(1j * np.angle(U2_tmp))

        # Back propagation
        U1_tmp = np.fft.ifft2(U2, norm="ortho")

        # Enforce amplitude at plane 1
        U1_new = A1 * np.exp(1j * np.angle(U1_tmp))

        if use_hio:
            U1 = U1_new + gamma * (U1_new - U1_old)
        else:
            U1 = U1_new

        U1_old = U1_new.copy()

        # Error metric
        U2_check = np.fft.fft2(U1, norm="ortho")
        err = np.mean((np.abs(U2_check) - A2)**2)
        errors.append(err)

    return U1, np.array(errors)


#========================
# Field eval
#========================

def evaluate_fields(U_true, U_est, dx):
    amp_true = np.abs(U_true)
    amp_est = np.abs(U_est)

    phi_true = np.angle(U_true)
    phi_est = np.angle(U_est)

    phase_err = np.angle(np.exp(1j * (phi_est - phi_true)))

    amp_mse = np.mean((amp_true - amp_est)**2)

    inner = np.sum(U_true * np.conj(U_est)) * dx * dx
    F = np.abs(inner)**2 / (np.sum(amp_true**2) * np.sum(amp_est**2))

    return {
        "inner": inner,
        "F": F,
        "amp_mse": amp_mse,
        "phase_err": phase_err,
        "amp_true": amp_true,
        "amp_est": amp_est,
        "phi_true": phi_true,
        "phi_est": phi_est
    }


#========================
# plot
#========================

def plot_reconstruction_results(results):
    amp_true = results["amp_true"]
    amp_est = results["amp_est"]
    phi_true = results["phi_true"]
    phi_est = results["phi_est"]
    phase_err = results["phase_err"]

    plt.figure(figsize=(14,8))

    plt.subplot(2,3,1)
    plt.imshow(amp_true, cmap="gray")
    plt.title("True amplitude |U|")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(phi_true, cmap="twilight")
    plt.title("True phase arg(U)")
    plt.axis("off")
    plt.colorbar()

    plt.subplot(2,3,3)
    plt.imshow(amp_true**2, cmap="gray")
    plt.title("True intensity |U|^2")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.imshow(amp_est, cmap="gray")
    plt.title("Recovered amplitude |U_est|")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.imshow(phi_est, cmap="twilight")
    plt.title("Recovered phase arg(U_est)")
    plt.axis("off")
    plt.colorbar()

    plt.subplot(2,3,6)
    plt.imshow(phase_err, cmap="coolwarm")
    plt.title("Wrapped phase error")
    plt.axis("off")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


#========================
# Run
#========================

if __name__ == "__main__":

    print("Loading measurement data...")
    A1 = np.load("./Simulations/A1.npy")
    A2 = np.load("./Simulations/A2.npy")

    print("Running Gerchberg–Saxton...")
    U1_est, errors = gerchberg_saxton(
        A1, A2,
        n_iters=200,
        seed=5,
        beta=0.8,
        use_hio=True,      # HIO recommended
        gamma=0.9
    )

    print("Evaluating reconstruction...")
    results = evaluate_fields(U_complex, U1_est, dx)

    print("Fidelity:", results["F"])
    print("Amplitude MSE:", results["amp_mse"])

    print("Plotting results")
    plot_reconstruction_results(results)

