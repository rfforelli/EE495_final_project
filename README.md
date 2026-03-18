# Scientific Machine Learning for SHO Spectral Fitting in BE-PFM

Final project code for EE 495: Scientific Machine Learning. We studied the inverse problem of recovering simple harmonic oscillator (SHO) parameters from band-excitation piezoresponse force microscopy (BE-PFM) spectra. The project began as an extension of an existing research effort on physics-constrained neural networks for fast spectroscopic fitting, and we expanded it into a comparative study of four modeling paradigms:

- **Physics-Constrained Neural Network**
- **Pure Neural Network**
- **Physics-Informed Neural Network (PINN-style baseline)**
- **Neural ODE baseline**

We aim to answer the question:

> When the governing physics is known, what is the most effective way to incorporate it into a learning system for parameter estimation and signal reconstruction?


## Getting Started

- Clone the repo ```git clone https://github.com/rfforelli/EE495_final_project.git```

- Install ```sho-project``` conda environment with ```conda env create -f environment.yml```

- ```conda activate sho-project```

- ```ipython kernel install --user --name=sho-project```

- Run through the entire ```EE495_final.ipynb``` notebook


---

## 1. Problem Statement

In BE-PFM, the measured cantilever response is a complex-valued frequency-domain signal. For each spectrum, one often wants to recover the parameters of an SHO model: amplitude, resonance frequency, quality factor, and phase. Classical nonlinear least-squares fitting can do this, but it is often slow, sensitive to initialization, and vulnerable to noise.

- **Input:** a complex spectrum sampled across frequency
- **Output:** physical parameters of an SHO model

This sits between two classical methods from a sciml perspective:

1. **Pure machine learning:** learn a direct map from spectra to parameters
2. **Physics-based inference:** fit spectra using a known analytical forward model

This project compares both strategies, and also two hybrid strategies, on the same data.

---

## 2. Scientific Background

### 2.1 The Simple Harmonic Oscillator Forward Model

The complex SHO response is modeled as

<!-- \[
H(\omega; \theta) = \frac{A e^{i\phi} \omega_0^2}{\omega^2 - i \omega \omega_0 / Q - \omega_0^2},
\] -->

H(ω; θ) = (A e^(iφ) ω₀²) / (ω² - i ω ω₀ / Q - ω₀²)

where the parameter vector is

<!-- \[
\theta = (A, \omega_0, Q, \phi).
\] -->

θ = (A, ω₀, Q, φ)

Here:

- A is the amplitude
- ω₀ is the resonance frequency
- Q is the quality factor
- φ is the phase

The measured spectrum is complex-valued, so the learning problem is based on the real and imaginary parts:

x(ω) = Re(H(ω; θ)) + i·Im(H(ω; θ))

In practice, the code uses the real and imaginary channels as a two-channel input tensor.

The forward map

θ ↦ H(ω; θ)

is explicit and known. The inverse map

H(ω; θ) ↦ θ

is the difficult object. It is not guaranteed to be well-conditioned, because multiple parameter combinations can generate similar spectra, especially under noise, limited frequency resolution, or phase wrapping. This is one of the reasons why physically informed regularization matters here. The goal is not only to minimize a numerical loss, but also to preserve physically meaningful structure in the outputs.

---

## 3. Dataset and Preprocessing

The data comes in HDF5 format, containing BE-PFM spectra.

### 3.1 Raw Data Structure

The raw data are complex spectra collected on a spatial grid over multiple voltage steps. The code reads

- the complex raw signal
- the frequency bins
- the existing SHO fit parameter dataset

The original spectra are stored with more frequency bins and are resampled to 80 bins for efficiency and consistency across models.

### 3.2 Real and Imaginary Representation

The complex signal is split into real and imaginary parts:

x_real = Re(x), x_imag = Im(x)

Each sample is represented as a tensor of shape

\[
(80, 2),
\]

where the last dimension contains the real and imaginary channels.

### 3.3 Standardization

The repository uses two types of scaling:

#### Global standardization for input spectra

For each channel,

<!-- \[
\tilde{x} = \frac{x - \mu}{\sigma},
\] -->

x̃ = (x − μ) / σ

where \(\mu\) and \(\sigma\) are computed globally across the entire training tensor for that channel.

#### Standardization for output parameters

The parameter vectors are standardized using `StandardScaler`:

<!-- \[
\tilde{\theta}_j = \frac{\theta_j - \mu_j}{\sigma_j}.
\] -->

θ̃ⱼ = (θⱼ − μⱼ) / σⱼ

This improves optimization stability and makes parameter losses numerically comparable across dimensions.

### 3.4 Train/Test Split

We use a fixed random train/test split:

- **70% test**
- **30% train**
---

## 4. Model Families

This repository compares four models, each corresponding to a different way of treating physics.

---

### 4.1 Physics-Constrained Neural Network

A neural network predicts the unscaled parameter vector

<!-- \[
\hat{\theta} = f_\psi(x),
\] -->


θ̂ = f_ψ(x)

and the final reconstructed spectrum is not produced by a learned decoder, but by the analytical SHO model:
<!-- 
\[
\hat{x} = H(\omega; \hat{\theta}).
\] -->


x̂ = H(ω; θ̂)

This means the network is constrained to reconstruct spectra that lie on the manifold of valid SHO responses.

#### Architecture

The model uses:

- 1D convolutional blocks on the two-channel spectral input
- pooling and dense layers
- a final embedding layer that predicts the four SHO parameters

It contains a skip-connection appending the convolutional and dense features before the final parameter prediction. This gives the model both local spectral pattern extraction and compact global feature aggregation.

#### Loss

The model is trained only on reconstruction loss:

<!-- \[
\mathcal{L}_{\text{PCNN}} = \frac{1}{N}\sum_{i=1}^{N} \|x_i - \hat{x}_i\|_2^2.
\] -->

𝓛_PCNN = (1/N) Σᵢ₌₁ᴺ ‖xᵢ − x̂ᵢ‖₂²

This is a hard-physics architecture: the decoder is the physical law itself.


This model does not directly minimize parameter error. Instead, it minimizes the mismatch between observed and physically reconstructed spectra. That distinction turns out to matter significantly.

---

### 4.2 Pure Neural Network Baseline

A standard feedforward network learns the inverse mapping directly:
<!-- 
\[
\hat{\theta} = f_\psi(x).
\] -->

θ̂ = f_ψ(x)

The predicted parameters are then inserted into the SHO equation only at evaluation time to reconstruct the signal.

#### Architecture

The Pure NN uses a fully connected multilayer perceptron:

- flattened input of dimension \(80 \times 2 = 160\)
- hidden layers of sizes 256, 128, 64
- ReLU nonlinearities
- final linear layer to 4 outputs

#### Loss

It is trained with supervised parameter loss:

<!-- \[
\mathcal{L}_{\text{Pure}} = \frac{1}{N}\sum_{i=1}^{N}\|\theta_i - \hat{\theta}_i\|_2^2.
\] -->

𝓛_Pure = (1/N) Σᵢ₌₁ᴺ ‖θᵢ − θ̂ᵢ‖₂²

This model tells us how we can go ignoring the physical model during training and simply learning the inverse map directly.

---

### 4.3 PINN-Style

This model combines supervision and physics-based reconstruction. The network predicts parameters as in the Pure NN baseline,
<!-- 
\[
\hat{\theta} = f_\psi(x),
\] -->


θ̂ = f_ψ(x)

but is trained with a hybrid objective that penalizes both parameter mismatch and reconstruction mismatch:

<!-- \[
\mathcal{L}_{\text{PINN}} =
\lambda_{\text{param}} \|\theta - \hat{\theta}\|_2^2
+
\lambda_{\text{phys}} \|x - H(\omega;\hat{\theta})\|_2^2.
\] -->

𝓛_PINN = λ_param ‖θ − θ̂‖₂² + λ_phys ‖x − H(ω; θ̂)‖₂²

In the code, both weights are set to 1.0:
<!-- 
\[
\lambda_{\text{param}} = \lambda_{\text{phys}} = 1.
\] -->


λ_param = λ_phys = 1

PINNs often use residual terms such as

<!-- \[
\mathcal{R}(u) = u_t + \mathcal{N}[u]
\] -->


𝓡(u) = u_t + 𝒩[u]

inside the loss. This work does not solve a PDE directly but uses the known forward physics operator as a soft regularizer in the loss. Conceptually, the model belongs to the same family of methods that enforce physics by penalizing deviation from physical consistency during training.

---

### 4.4 Neural ODE Baseline

This model introduces continuous-depth dynamics in latent space. The input is first encoded into a latent representation \(z_0\), then evolved through a learned ODE:

<!-- \[
\frac{dz}{dt} = f_\theta(z,t),
\] -->

dz/dt = f_θ(z, t)

and finally decoded into the predicted parameters:

<!-- \[
\hat{\theta} = g(z(T)).
\] -->

θ̂ = g(z(T))


#### Architecture

The Neural ODE baseline contains:

- an encoder \(x \mapsto z_0\)
- an ODE block defined by a neural vector field
- a decoder \(z(T) \mapsto \hat{\theta}\)

The latent dynamics are integrated numerically using `torchdiffeq.odeint`.

#### Loss

Like the Pure NN, it is trained with supervised parameter loss:

<!-- \[
\mathcal{L}_{\text{NODE}} = \frac{1}{N}\sum_{i=1}^{N}\|\theta_i - \hat{\theta}_i\|_2^2.
\] -->

𝓛_NODE = (1/N) Σᵢ₌₁ᴺ ‖θᵢ − θ̂ᵢ‖₂²

---

## 5. Optimization

The Pure NN, PINN-style baseline, and Neural ODE baseline are trained with Adam:

<!-- \[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,
\]
\[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
\]
\[
\theta_{t+1} = \theta_t - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}.
\] -->


m_t = β₁ m_{t−1} + (1 − β₁) g_t

v_t = β₂ v_{t−1} + (1 − β₂) g_t²

θ_(t+1) = θ_t − α m_t / (√v_t + ε)

This is appropriate for their smooth supervised or hybrid losses.


The Physics-Constrained NN uses AdaHessian, which approximates second-order curvature information. This is useful because training with a hard physics decoder can produce a more difficult loss landscape than direct regression.


---

## 6. Evaluation Metrics


After prediction, the parameters are mapped through the SHO model to produce a reconstructed spectrum. Reconstruction MSE is then computed on the scaled real and imaginary channels:

<!-- \[
\text{MSE}_{\text{recon}} =
\frac{1}{N}\sum_{i=1}^{N}\|x_i - \hat{x}_i\|_2^2.
\] -->


MSE_recon = (1/N) Σᵢ₌₁ᴺ ‖xᵢ − x̂ᵢ‖₂²

This is the most natural metric for the physics-constrained model and, arguably, the most scientifically meaningful one, since it measures whether the predicted parameters generate the observed signal.


Parameter error is also computed:

<!-- \[
\text{MSE}_{\text{param},j}
=
\frac{1}{N}\sum_{i=1}^{N} (\theta_{ij} - \hat{\theta}_{ij})^2,
\] -->

MSE_param,j = (1/N) Σᵢ₌₁ᴺ (θᵢⱼ − θ̂ᵢⱼ)²


for each of the four parameters.

For inverse problems, low parameter error does not automatically imply better signal-level fidelity, especially when the inverse mapping is non-unique or ill-conditioned.

---

## 7. Results Summary

The main results are summarized below.

### Reconstruction MSE

| Model | Avg Reconstruction MSE |
|---|---:|
| Pure NN | \(1.6506 \times 10^{-1}\) |
| PINN | \(6.3986 \times 10^{-2}\) |
| Neural ODE | \(1.3784 \times 10^{-1}\) |
| Physics-Constrained NN | **\(4.5914 \times 10^{-2}\)** |

### Parameter MSE

#### Physics-Constrained NN
- Amplitude: \(7.9470 \times 10^{-9}\)
- Frequency: \(7.3416 \times 10^{6}\)
- Quality Factor: \(5.4671 \times 10^{3}\)
- Phase: \(4.9509 \times 10^{1}\)

#### Pure NN
- Amplitude: \(4.1930 \times 10^{-12}\)
- Frequency: \(7.7675 \times 10^{5}\)
- Quality Factor: \(8.3523 \times 10^{2}\)
- Phase: \(1.0040\)

#### PINN
- Amplitude: \(1.0823 \times 10^{-11}\)
- Frequency: \(6.5413 \times 10^{5}\)
- Quality Factor: \(7.4528 \times 10^{2}\)
- Phase: \(2.1506\)

#### Neural ODE
- Amplitude: \(5.3104 \times 10^{-12}\)
- Frequency: \(5.6782 \times 10^{5}\)
- Quality Factor: \(6.7407 \times 10^{2}\)
- Phase: \(9.9670 \times 10^{-1}\)

---

The key finding is that the Physics-Constrained Neural Network delivers the most accurate signal reconstruction, even though it does not achieve the lowest parameter MSE.

This model is trained to generate spectra that adhere to the SHO constraints. As a result, it learns to remain within the space of physically valid spectral responses, which significantly enhances fidelity at the signal level.

In contrast, the Pure Neural Network and Neural ODE focus solely on minimizing parameter error. While they may produce parameter estimates that are numerically closer to the ground truth, these estimates do not necessarily yield the most accurate reconstruction of the observed spectra.

Across all models, resonance frequency is recovered relatively well. This makes sense because the resonance peak location is highly informative and strongly visible in the magnitude response.

Phase and quality factor are much more difficult which can be because phase wrapping makes angular regression non-Euclidean, quality factor controls peak sharpness and damping, which can be harder to identify under finite resolution and noise, and the inverse mapping is not equally identifiable in all coordinates.

The PINN-style model improves reconstruction substantially over the Pure NN, while preserving good parameter accuracy. The soft physics term regularizes the inverse map without fully constraining it.

The Neural ODE trains much more slowly and does not beat the PINN or Physics-Constrained NN in reconstruction.

See the Jupyter notebook for all training results.

---

NOTE: The values presently in the Jupiter Notebook may be slightly different than the values reported here and in the two page report since we have ran it again since writing this but they should be substantially similar.

Ryan Forelli and Ethan Gindlesperger