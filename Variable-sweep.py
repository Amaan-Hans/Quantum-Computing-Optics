import matplotlib
matplotlib.use("Agg")  # Headless mode

import matplotlib.pyplot as plt
import numpy as np
import os, math, json
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ============================================================
# Make unique run directory
# ============================================================
def make_run_dir(base="Runs"):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"{base}/run_{ts}"
    os.makedirs(path, exist_ok=True)
    return path

# ============================================================
# Zernike functions
# ============================================================
def R_nm(n, m, rho):
    R = np.zeros_like(rho)
    for k in range((n - m)//2 + 1):
        num = (-1)**k * math.factorial(n - k)
        den = (math.factorial(k) *
               math.factorial((n+m)//2 - k) *
               math.factorial((n-m)//2 - k))
        R += (num / den) * rho**(n - 2*k)
    return R

def Z_nm(n, m, rho, phi, mask):
    Z = np.zeros_like(rho)
    inside = mask.copy()
    if m > 0:
        Z[inside] = R_nm(n, m, rho[inside]) * np.cos(m * phi[inside])
    elif m < 0:
        Z[inside] = R_nm(n, -m, rho[inside]) * np.sin(-m * phi[inside])
    else:
        Z[inside] = R_nm(n, 0, rho[inside])
    return Z

# ============================================================
# Field generator
# ============================================================
def generate_field(N, dx, sigma, alpha, n, m):
    x = np.linspace(-N/2 * dx, N/2 * dx, N)
    y = np.linspace(-N/2 * dx, N/2 * dx, N)
    X, Y = np.meshgrid(x, y)

    R_ap = (N/2)*dx
    rho  = np.sqrt(X**2 + Y**2) / R_ap
    phi  = np.arctan2(Y, X)
    mask = rho <= 1

    amplitude = np.exp(-(X**2 + Y**2) / sigma**2) * mask
    Z = Z_nm(n, m, rho, phi, mask)
    phase = alpha * Z

    U_complex = amplitude * np.exp(1j * phase)

    return U_complex, amplitude, phase, mask

# ============================================================
# CPU Gerchberg–Saxton
# ============================================================
def GS_cpu(A1, A2, mask, beta=0.8, n_iters=250):
    phase0 = np.random.uniform(-np.pi, np.pi, A1.shape)
    U1 = A1 * np.exp(1j * phase0)

    for _ in range(n_iters):
        U2_tmp = np.fft.fft2(U1, norm="ortho")
        U2 = (beta*A2 + (1-beta)*np.abs(U2_tmp)) * np.exp(1j * np.angle(U2_tmp))

        U2 = A2 * np.exp(1j * np.angle(U2))
        U1 = np.fft.ifft2(U2, norm="ortho")

        U1 = mask * (A1 * np.exp(1j * np.angle(U1)))

    return U1

# ============================================================
# Composite image saver
# ============================================================
def save_composite(path, amp, phase, inten, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(amp, cmap="gray")
    axes[0].set_title("Amplitude")
    axes[0].axis("off")

    axes[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Phase")
    axes[1].axis("off")

    axes[2].imshow(inten, cmap="gray")
    axes[2].set_title("Intensity")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ============================================================
# Per-experiment worker (parallel safe)
# ============================================================
def run_single_experiment(args):

    (root, alpha, sigma, beta, n, m, N, dx, n_iters) = args

    folder = f"{root}/alpha{alpha:.2f}_sigma{sigma:.1e}_beta{beta:.2f}_n{n}_m{m}"
    os.makedirs(folder, exist_ok=True)

    # Generate true field
    U_true, A1, phase, mask = generate_field(N, dx, sigma, alpha, n, m)
    A2 = np.abs(np.fft.fft2(U_true, norm="ortho"))

    # Run GS
    U_est = GS_cpu(A1, A2, mask, beta=beta, n_iters=n_iters)

    # Extract results
    amp_est = np.abs(U_est)
    phase_est = np.angle(U_est)
    inten_est = amp_est**2
    inten_true = A1**2

    # Composite: true field
    save_composite(f"{folder}/true_field.png",
                   A1, phase, inten_true,
                   "True Field")

    # Composite: recovered
    save_composite(f"{folder}/recovered_field.png",
                   amp_est, phase_est, inten_est,
                   "Recovered Field")

    # Zernike phase only
    plt.figure(figsize=(5,5))
    plt.imshow(phase, cmap="coolwarm")
    plt.title("Zernike Phase")
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{folder}/zernike_phase.png", dpi=150)
    plt.close()

    # Metadata
    metadata = {
        "alpha": float(alpha),
        "sigma": float(sigma),
        "beta": float(beta),
        "n": n,
        "m": m,
        "N": N,
        "dx": dx,
        "iterations": n_iters,
        "timestamp": datetime.now().isoformat(),
        "cpu_only": True
    }
    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

# ============================================================
# Sweep controller
# ============================================================
def run_sweep():

    root = make_run_dir()
    print("Saving results to:", root)

    # Sweep ranges
    alphas = np.linspace(1*np.pi, 5*np.pi, 5)
    sigmas = np.linspace(2e-4, 8e-4, 5)
    betas  = np.linspace(0.5, 0.9, 4)
    modes  = [(2,2),(3,1),(3,3)]

    N = 1024
    dx = 8e-6
    n_iters = 250

    tasks = []
    for n, m in modes:
        for alpha in alphas:
            for sigma in sigmas:
                for beta in betas:
                    tasks.append((root, alpha, sigma, beta, n, m, N, dx, n_iters))

    print(f"Total experiments: {len(tasks)}")

    # CPU parallel execution
    with Pool(cpu_count()) as pool:
        pool.map(run_single_experiment, tasks)

    print("Sweep complete.")

# ============================================================
if __name__ == "__main__":
    run_sweep()
 