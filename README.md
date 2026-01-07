# Learning Phase Retrieval with Neural Gerchberg–Saxton

This project investigates whether machine learning can learn to solve the optical phase retrieval problem faster and more robustly than the classical Gerchberg–Saxton (GS) algorithm.

Given intensity measurements in two optical planes, the objective is to reconstruct the unknown phase of the complex optical field using a learned, iterative model.

## Problem Motivation

In optical systems, sensors typically measure only intensity, not phase. However, the full complex field is required to understand and control wave propagation.

The classical Gerchberg–Saxton algorithm alternates between two constraint planes (for example, spatial domain and Fourier domain), enforcing known magnitudes while iteratively updating phase. While effective, GS has several limitations:

Can require many iterations

Can stagnate

Can be sensitive to noise

Does not exploit data priors or learned structure

This project investigates whether a neural model can learn this mapping and either:

directly predict phase, or

accelerate/guide GS-style iteration.

## Problem Definition

We consider two intensity fields:

# Plane 1 intensity:

𝐼
1
(
𝑥
)
=
∣
𝑈
1
(
𝑥
)
∣
2
I
1
	​

(x)=∣U
1
	​

(x)∣
2

# Plane 2 intensity (for example Fourier plane):

𝐼
2
(
𝑘
)
=
∣
𝑈
2
(
𝑘
)
∣
2
I
2
	​

(k)=∣U
2
	​

(k)∣
2

The goal is to recover the unknown phase of the complex field:

𝑈
1
(
𝑥
)
=
𝐴
1
(
𝑥
)
𝑒
𝑖
𝜙
(
𝑥
)
U
1
	​

(x)=A
1
	​

(x)e
iϕ(x)

where only the magnitudes are known.

## Approach

Data Generation

Simulate random complex optical fields

Propagate between planes using Fourier-based propagation

Store:

intensity measurements in both planes

ground-truth complex phase

Baseline

Implement and evaluate the classical Gerchberg–Saxton algorithm

Neural Model

A model that takes two intensity fields as input

Either outputs phase directly, or refines an iterative estimate

Potentially unrolled / physics-informed architectures

Training

Train using supervised learning

Possible loss functions include:

phase reconstruction error

complex field error (real and imaginary components)

forward consistency loss against measured intensities

Evaluation
Compare machine learning model against Gerchberg–Saxton in terms of:

reconstruction accuracy

number of iterations required

computational speed

robustness to noise
