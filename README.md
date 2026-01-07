Learning Phase Retrieval with Neural Gerchberg–Saxton
====================================================

This project investigates whether machine learning can learn to solve the optical phase retrieval problem faster and more robustly than the classical Gerchberg–Saxton (GS) algorithm.

Given intensity measurements in two optical planes, the objective is to reconstruct the unknown phase of the complex optical field using a learned, iterative model.

----------------------------------------------------
Problem Motivation
----------------------------------------------------
In optical systems, sensors typically measure only intensity, not phase. However, the full complex field is required to understand and control wave propagation.

The classical Gerchberg–Saxton algorithm alternates between two constraint planes (for example, spatial domain and Fourier domain), enforcing known magnitudes while iteratively updating phase.

Limitations of GS:
- Requires many iterations
- Can stagnate
- Sensitive to noise
- Does not exploit learned structure or priors

This project investigates whether a neural model can:
- directly predict phase, or
- accelerate or guide GS-style iteration

----------------------------------------------------
Problem Definition
----------------------------------------------------
Two intensity fields are considered:

Plane 1 intensity:
I1(x) = |U1(x)|^2

Plane 2 intensity:
I2(k) = |U2(k)|^2

The goal is to recover the unknown phase of the complex field:

U1(x) = A1(x) * exp(i * phi(x))

Only amplitudes are known; phase must be reconstructed.

----------------------------------------------------
Approach
----------------------------------------------------
1. Data Generation
- Simulate random complex optical fields
- Propagate between planes using Fourier-based propagation
- Store intensities and ground-truth phase

2. Baseline
- Implement classical Gerchberg–Saxton

3. Neural Model
- Takes intensity fields as input
- Predicts or refines phase
- Supports physics-informed and unrolled architectures

4. Training
Possible losses:
- phase reconstruction loss
- real + imaginary complex field loss
- forward propagation consistency

5. Evaluation
Compare machine learning approach against GS for:
- reconstruction accuracy
- computation time
- robustness to noise
- number of iterations

----------------------------------------------------
Repository Structure
----------------------------------------------------
data/
  raw/
  processed/
src/
  optics/
    propagation.py
    gs_baseline.py
  models/
    neural_gs.py
  datasets.py
  train.py
  evaluate.py
notebooks/
requirements.txt
README.md
LICENSE

----------------------------------------------------
Usage Overview
----------------------------------------------------
1. Generate Data
python -m src.generate_data --num-samples 10000 --output-dir data/processed

2. Train Model
python -m src.train --data-dir data/processed --epochs 50 --batch-size 32 --lr 1e-4

3. Evaluate
python -m src.evaluate --data-dir data/processed --model-checkpoint checkpoints/best_model.pt --num-gs-iterations 100

----------------------------------------------------
Roadmap
----------------------------------------------------
- Implement GS baseline
- Build dataset generation
- Implement neural phase retrieval
- Benchmark against GS
- Noise robustness experiments
- Optional: apply to real optical data

----------------------------------------------------
Background Concepts
----------------------------------------------------
- Fourier optics
- Phase retrieval
- Gerchberg–Saxton algorithm
- Iterative reconstruction
- Machine learning based inverse methods
  
