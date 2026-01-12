import streamlit as st
import os
import json
from PIL import Image
import numpy as np

def load_experiments(base_dir="Runs"):
    experiments = []
    for root, dirs, files in os.walk(base_dir):
        if "metadata.json" in files:
            folder = root
            meta_path = os.path.join(root, "metadata.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Attach folder path
            meta["folder"] = folder
            experiments.append(meta)
    return experiments

def select_experiment(experiments, alpha, sigma, beta, n, m):
    # Find closest match
    best = None
    best_dist = 1e9

    for exp in experiments:
        dist = (
            abs(exp["alpha"] - alpha) +
            abs(exp["sigma"] - sigma) +
            abs(exp["beta"]  - beta ) +
            abs(exp["n"] - n) +
            abs(exp["m"] - m)
        )
        if dist < best_dist:
            best_dist = dist
            best = exp

    return best

def show_images(folder):
    true_img      = Image.open(os.path.join(folder, "true_field.png"))
    recovered_img = Image.open(os.path.join(folder, "recovered_field.png"))
    zernike_img   = Image.open(os.path.join(folder, "zernike_phase.png"))

    st.write("### True Field (Amplitude | Phase | Intensity)")
    st.image(true_img, use_column_width=True)

    st.write("### Recovered Field")
    st.image(recovered_img, use_column_width=True)

    st.write("### Zernike Polynomial Phase")
    st.image(zernike_img, use_column_width=True)

st.title("Gerchberg–Saxton Sweep Explorer")
st.write("Interactively browse your high-dimensional sweep results.")

experiments = load_experiments()

if len(experiments) == 0:
    st.error("No experiments found in 'Runs/' folder.")
    st.stop()

alphas = sorted(set([exp["alpha"] for exp in experiments]))
sigmas = sorted(set([exp["sigma"] for exp in experiments]))
betas  = sorted(set([exp["beta"]  for exp in experiments]))
ns     = sorted(set([exp["n"]     for exp in experiments]))
ms     = sorted(set([exp["m"]     for exp in experiments]))

st.sidebar.header("Select Parameters")

alpha = st.sidebar.slider("alpha", float(min(alphas)), float(max(alphas)), float(alphas[0]), step=0.01)
sigma = st.sidebar.slider("sigma", float(min(sigmas)), float(max(sigmas)), float(sigmas[0]), step=0.00001)
beta  = st.sidebar.slider("beta",  float(min(betas)),  float(max(betas)),  float(betas[0]), step=0.01)
n     = st.sidebar.selectbox("n (Zernike index)", ns)
m     = st.sidebar.selectbox("m (Zernike index)", ms)

chosen_exp = select_experiment(experiments, alpha, sigma, beta, n, m)

st.subheader("Metadata")
st.json(chosen_exp)

# ------------------------
# Display images
# ------------------------
show_images(chosen_exp["folder"])
