# Physics-Informed Variational Autoencoder (PIVA) for 3-Compartment Prostate MRI
## Theory, Architecture, and Empirical Analysis

---

## 1. The Inverse Problem

### 1.1 Biological Setup

Prostate tissue is not uniform. At the microscopic level it consists of three distinct biological compartments:

| Compartment | Abbreviation | Physical role |
|---|---|---|
| Epithelium | ep | Glandular cell walls |
| Stroma | st | Fibromuscular connective tissue |
| Lumen | lu | Fluid-filled glandular spaces |

Each compartment has three measurable MRI parameters:

- $D_c$ — diffusivity $(\times 10^{-3}\ \text{mm}^2/\text{s})$: how fast water molecules diffuse through compartment $c$
- $T_{2,c}$ — transverse relaxation time (ms): how fast the MRI signal decays in compartment $c$
- $v_c$ — volume fraction: proportion of tissue occupied by compartment $c$, with $\sum_c v_c = 1$

The full parameter vector for one voxel is:

$$\boldsymbol{\theta} = [\underbrace{D_{ep},\ D_{st},\ D_{lu}}_{\text{diffusivities}},\ \underbrace{T_{2,ep},\ T_{2,st},\ T_{2,lu}}_{\text{relaxation times}},\ \underbrace{v_{ep},\ v_{st},\ v_{lu}}_{\text{volume fractions}}] \in \mathbb{R}^9$$

### 1.2 The MRI Signal Equation

The measured MRI signal at diffusion-weighting $b$ and echo time $\text{TE}$ is the sum of contributions from all three compartments:

$$S(b,\ \text{TE};\ \boldsymbol{\theta}) = S_0 \sum_{c \in \{ep,\ st,\ lu\}} v_c \cdot \exp\!\left(-\frac{b \cdot D_c}{1000}\right) \cdot \exp\!\left(-\frac{\text{TE}}{T_{2,c}}\right)$$

where:
- $b\ (\text{ms/mm}^2)$ is the diffusion-weighting gradient strength
- $\text{TE}\ (\text{ms})$ is the echo time
- $S_0$ is the baseline signal (set to 1000 in this implementation)
- The factor of 1000 in the denominator converts $D$ from $\times 10^{-3}\ \text{mm}^2/\text{s}$ to matching units with $b$

For $N_b = 4$ b-values and $N_{TE} = 4$ echo times, the acquisition yields $N_s = 16$ signal measurements per voxel:

$$\mathbf{x} = [S(b_1,\text{TE}_1),\ S(b_1,\text{TE}_2),\ \ldots,\ S(b_4,\text{TE}_4)]^\top \in \mathbb{R}^{16}$$

### 1.3 Why This is Hard

The forward problem — given $\boldsymbol{\theta}$, compute $\mathbf{x}$ — is straightforward. The **inverse problem** — given $\mathbf{x}$, recover $\boldsymbol{\theta}$ — is fundamentally ill-conditioned for three reasons.

**Underdetermination.** 16 measurements must uniquely determine 9 parameters. While the system is technically overdetermined in dimension count, the signal equation is highly nonlinear and many $\boldsymbol{\theta}$ combinations produce nearly identical $\mathbf{x}$.

**Identifiability.** Consider swapping the contribution of the epithelium and stroma compartments while adjusting volume fractions proportionally. The resulting signal can be nearly identical. The parameter space has approximate symmetries that the signal does not break.

**Noise amplification.** Clinical MRI operates at SNR ≈ 12:1 to 100:1. MRI magnitude images follow a **Rician distribution** arising from complex Gaussian noise in k-space:

$$x_{\text{noisy}} = \sqrt{(S_{\text{clean}} + \varepsilon_{\text{real}})^2 + \varepsilon_{\text{imag}}^2}, \qquad \varepsilon_{\text{real}},\ \varepsilon_{\text{imag}} \sim \mathcal{N}(0,\ \sigma^2)$$

At $\sigma = 0.05$ (clinical SNR ≈ 20:1, relative to unit-scale $S_0$), the noise corrupts the signal substantially enough that many $\boldsymbol{\theta}$ values are consistent with the observed $\mathbf{x}_{\text{noisy}}$. This **posterior uncertainty** is real, not a modelling artifact — it reflects genuine ambiguity in the physics.

---

## 2. Classical Approach: NLLS

### 2.1 How It Works

Non-linear least squares (NLLS) treats the inverse problem as a per-voxel optimization:

$$\hat{\boldsymbol{\theta}}_{\text{NLLS}} = \arg\min_{\boldsymbol{\theta} \in \Omega}\ \left\|\mathbf{x}_{\text{noisy}} - S(\cdot;\ \boldsymbol{\theta})\right\|_2^2$$

where $\Omega$ is the feasible parameter set (physiological bounds). Implementation uses `scipy.curve_fit` with the Trust Region Reflective (TRF) algorithm and bounded constraints.

The 8 free parameters (with $v_{lu} = 1 - v_{ep} - v_{st}$ by the simplex constraint) are:

$$[D_{ep},\ D_{st},\ D_{lu},\ T_{2,ep},\ T_{2,st},\ T_{2,lu},\ v_{ep},\ v_{st}]$$

### 2.2 Why It Fails

At clinical noise levels, the loss landscape $\|\mathbf{x}_{\text{noisy}} - S(\cdot;\boldsymbol{\theta})\|^2$ develops many local minima with similar objective values. The optimizer finds different local solutions depending on initialization, producing inconsistent parameter estimates across voxels.

From the notebook results at $\sigma = 0.05$:

$$R^2_v = 0.238, \qquad \text{T2-MAE} = 91.8\ \text{ms}, \qquad D\text{-MAE} = 0.225 \times 10^{-3}\ \text{mm}^2/\text{s}$$

At $\sigma = 0.08$ (highest clinical noise), volume fraction recovery collapses to $R^2_v = -0.137$ — worse than predicting the population mean. Some voxel fits fail entirely, requiring a validity mask that silently excludes the worst-performing cases from the reported metrics, making NLLS results optimistically biased.

---

## 3. Physics-Informed Autoencoder (PIA)

### 3.1 Key Idea

PIA replaces per-voxel optimization with a learned amortized inference function — a neural network $f_\phi$ trained to invert the signal equation:

$$f_\phi: \mathbb{R}^{16} \to \mathbb{R}^9, \qquad \hat{\boldsymbol{\theta}} = f_\phi(\mathbf{x}_{\text{noisy}})$$

The critical architectural choice is that the **decoder is not learned** — it is the fixed physics equation $S(\cdot; \boldsymbol{\theta})$. Only the encoder is trained.

### 3.2 Architecture

**Encoder:** A shared MLP maps the 16-dimensional noisy signal to a 512-dimensional feature representation:

$$\mathbf{h} = \text{MLP}(\mathbf{x}_{\text{noisy}}) \in \mathbb{R}^{512}$$

with layers $16 \to 64 \to 128 \to 256 \to 512 \to 512$, each followed by LeakyReLU.

**Predictor heads:** Three separate networks extract D, T2, and v from the shared feature:

$$D_{\text{raw}} = \text{Head}_D(\mathbf{h}) \in \mathbb{R}^3, \qquad T_{2,\text{raw}} = \text{Head}_{T2}(\mathbf{h}) \in \mathbb{R}^3, \qquad v_{\text{raw}} = \text{Head}_v(\mathbf{h}) \in \mathbb{R}^3$$

Each head is a 3-layer MLP ($512 \to 512 \to 512 \to 3$).

**Physiological constraints:** Raw outputs are mapped into plausible physiological ranges using tanh:

$$D_c = D_{\text{mean},c} + D_{\delta,c} \cdot \tanh(D_{\text{raw},c}), \qquad c \in \{ep,\ st,\ lu\}$$
$$T_{2,c} = T_{2,\text{mean},c} + T_{2,\delta,c} \cdot \tanh(T_{2,\text{raw},c})$$
$$\mathbf{v} = \text{softmax}(\mathbf{v}_{\text{raw}})$$

where $D_{\delta,c}$ and $T_{2,\delta,c}$ are the half-widths of the physiological bounds.

**Decoder:** The fixed physics equation maps $(\hat{D}, \hat{T}_2, \hat{v})$ to a reconstructed signal:

$$\hat{S}(b,\text{TE}) = S_0 \sum_c \hat{v}_c \cdot \exp\!\left(-\frac{b \hat{D}_c}{1000}\right) \cdot \exp\!\left(-\frac{\text{TE}}{\hat{T}_{2,c}}\right)$$

### 3.3 Training Objective

PIA minimizes the mean squared error between reconstructed and **clean** (noiseless) signal:

$$\mathcal{L}_{\text{PIA}}(\phi) = \mathbb{E}_{(\mathbf{x}_{\text{noisy}},\ \mathbf{x}_{\text{clean}})}\!\left[\left\|\hat{S}(\hat{\boldsymbol{\theta}}) - \mathbf{x}_{\text{clean}}\right\|_2^2\right]$$

The key insight is that training against the clean signal, not the noisy signal, forces the encoder to learn to **denoise** as part of the inversion task.

### 3.4 Results and Limitation

PIA substantially outperforms NLLS at $\sigma = 0.05$:

| Metric | NLLS | PIA | Improvement |
|---|---|---|---|
| $D$ MAE | 0.225 | 0.093 | 59% |
| $T_2$ MAE | 91.8 ms | 32.6 ms | 65% |
| $v$ $R^2$ | 0.238 | 0.572 | ×2.4 |

The fundamental limitation is that PIA is **fully deterministic** — it outputs a single point estimate with no uncertainty. Given that the inverse problem is genuinely ill-conditioned, this point estimate may be one of many equally plausible solutions. A clinician cannot tell from PIA's output how reliable the prediction is for a specific voxel.

---

## 4. Physics-Informed Variational Autoencoder (PIVA)

### 4.1 Motivation

The forward model $\mathbf{x} = S(\boldsymbol{\theta}) + \varepsilon$ is a many-to-one map at clinical noise levels. Given a noisy measurement $\mathbf{x}_{\text{noisy}}$, the set of plausible parameters is a distribution over $\boldsymbol{\theta}$, not a single point. PIVA's goal is to learn this posterior distribution:

$$p(\boldsymbol{\theta} \mid \mathbf{x}_{\text{noisy}})$$

A well-calibrated posterior means: across many voxels where the model assigns a 95% credible interval, the true parameter falls inside that interval 95% of the time.

### 4.2 Latent Variable Model

PIVA treats the tissue parameters $\boldsymbol{\theta}$ as a deterministic function of a **latent variable** $\mathbf{z} \in \mathbb{R}^9$:

$$\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\ \mathbf{I})$$
$$\boldsymbol{\theta} = g(\mathbf{z}) = [D(\mathbf{z}),\ T_2(\mathbf{z}),\ v(\mathbf{z})]$$
$$\mathbf{x} = S(\boldsymbol{\theta}) + \varepsilon$$

The 9 latent dimensions map directly to the 9 tissue parameters: $z_{0:3} \to D$, $z_{3:6} \to T_2$, $z_{6:9} \to v$.

The generative direction is $p(\mathbf{z}) \to g(\mathbf{z}) \to S(g(\mathbf{z}))$. The inference direction — what PIVA learns — is the approximate posterior:

$$q_\phi(\mathbf{z} \mid \mathbf{x}) \approx p(\mathbf{z} \mid \mathbf{x})$$

### 4.3 Evidence Lower Bound (ELBO)

Exact posterior inference requires computing $p(\mathbf{x})$ by integrating over $\mathbf{z}$, which is intractable. Variational inference instead maximizes a lower bound on the log-likelihood.

Starting from:

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x} \mid \mathbf{z})\ p(\mathbf{z})\ d\mathbf{z}$$

Using Jensen's inequality with any distribution $q(\mathbf{z})$:

$$\log p(\mathbf{x}) = \log \mathbb{E}_{q(\mathbf{z})}\!\left[\frac{p(\mathbf{x} \mid \mathbf{z})\ p(\mathbf{z})}{q(\mathbf{z})}\right] \geq \mathbb{E}_{q(\mathbf{z})}\!\left[\log \frac{p(\mathbf{x} \mid \mathbf{z})\ p(\mathbf{z})}{q(\mathbf{z})}\right]$$

This yields the ELBO:

$$\log p(\mathbf{x}) \geq \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\!\left[\log p(\mathbf{x} \mid \mathbf{z})\right]}_{\text{reconstruction term}} - \underbrace{D_{\text{KL}}\!\left[q_\phi(\mathbf{z} \mid \mathbf{x})\ \|\ p(\mathbf{z})\right]}_{\text{regularisation term}}$$

**Reconstruction term:** How well does sampling $\mathbf{z}$ from the posterior and decoding recover $\mathbf{x}$? Under Gaussian likelihood with fixed variance, this is proportional to the negative signal MSE.

**KL term:** How far is the learned posterior from the prior? This penalises the encoder for making predictions that deviate too far from the $\mathcal{N}(0, I)$ prior — regularising the latent space.

### 4.4 KL Divergence — Closed Form

With $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$D_{\text{KL}}\!\left[q\ \|\ p\right] = -\frac{1}{2} \sum_{j=1}^{9} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

where the sum is over all 9 latent dimensions. Decomposing per dimension:

$$\text{KL}_j = -\frac{1}{2}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

Key observations:
- $\text{KL}_j = 0$ when $\mu_j = 0$ and $\sigma_j = 1$ (posterior equals prior — encoder learned nothing)
- $\text{KL}_j \to \infty$ when $\sigma_j \to 0$ (posterior collapses to a point — encoder is overconfident)
- $\text{KL}_j \approx \frac{1}{2}\mu_j^2$ when $\sigma_j \ll 1$ (large KL driven by large $|\mu|$, not by small $\sigma$)

This last point is critical and is directly responsible for the failures observed in the experiment.

### 4.5 Reparameterization Trick

The reconstruction loss requires backpropagating through a sample $\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$. Sampling is not differentiable. The reparameterization trick rewrites the sample as a deterministic function of the parameters plus independent noise:

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\ \mathbf{I})$$

Now $\partial \mathbf{z} / \partial \boldsymbol{\mu} = \mathbf{I}$ and $\partial \mathbf{z} / \partial \boldsymbol{\sigma} = \text{diag}(\boldsymbol{\varepsilon})$ — gradients flow through the sample to the encoder parameters.

In practice, the encoder predicts $\log \sigma^2$ (log-variance) for numerical stability:

$$\boldsymbol{\sigma} = \exp\!\left(\frac{1}{2} \log \boldsymbol{\sigma}^2\right), \qquad \log \boldsymbol{\sigma}^2\ \text{clamped to}\ [-8,\ 4]$$

### 4.6 PIVA Architecture

**Encoder:** Identical depth to PIA — shared MLP $16 \to 64 \to 128 \to 256 \to 512 \to 512$.

**Variational heads:** Instead of three separate D/T2/v predictor heads, PIVA has two heads producing the posterior parameters:

$$\boldsymbol{\mu} = \text{Head}_\mu(\mathbf{h}) \in \mathbb{R}^9, \qquad \log \boldsymbol{\sigma}^2 = \text{Head}_{\log\sigma^2}(\mathbf{h}) \in \mathbb{R}^9$$

Each head has architecture $512 \to 512 \to 512 \to 9$ (predictor depth = 3).

**Reparameterization:** $\mathbf{z} = \boldsymbol{\mu} + \exp(0.5 \cdot \log \boldsymbol{\sigma}^2) \odot \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$.

**Parameter decoding (`params_from_z`):** The 9-dimensional $\mathbf{z}$ is split and mapped to physiological parameters:

$$D_c = D_{\text{mean},c} + D_{\delta,c} \cdot \tanh(z_c), \qquad c \in \{ep,\ st,\ lu\},\ z_c = z_{c}$$
$$T_{2,c} = T_{2,\text{mean},c} + T_{2,\delta,c} \cdot \tanh(z_{3+c})$$
$$\mathbf{v} = \text{softmax}(\mathbf{z}_{6:9})$$

**Physics decoder:** Identical to PIA — the fixed signal equation applied to $(\hat{D}, \hat{T}_2, \hat{v})$.

### 4.7 Training Objective

The PIVA loss is the negative ELBO scaled by signal amplitude:

$$\mathcal{L}_{\text{PIVA}} = \underbrace{\mathbb{E}_{\boldsymbol{\varepsilon}}\!\left[\left\|\hat{S}(\mathbf{z}) - \mathbf{x}_{\text{clean}}\right\|_2^2\right]}_{\text{signal MSE}} + \beta \cdot D_{\text{KL}}\!\left[q_\phi(\mathbf{z}|\mathbf{x})\ \|\ \mathcal{N}(\mathbf{0},\mathbf{I})\right]$$

The $\beta$ coefficient (from $\beta$-VAE) controls the trade-off:
- $\beta \to 0$: PIVA degenerates to PIA (no KL pressure, $\sigma \to 0$)
- $\beta \to \infty$: posterior forced to prior ($\mu \to 0$, $\sigma \to 1$), reconstruction collapses

**KL warm-up:** $\beta$ is linearly ramped from 0 to $\beta_{\max}$ over the first `warmup_epochs` epochs. This allows the reconstruction term to establish a good encoder before KL pressure is applied. Without warm-up, the encoder can get trapped in a degenerate solution where KL is small but reconstruction is poor.

$$\beta(t) = \beta_{\max} \cdot \min\!\left(1,\ \frac{t}{T_{\text{warm}}}\right)$$

### 4.8 Inference: Monte Carlo Posterior

At test time, PIVA runs $N_{\text{MC}}$ stochastic forward passes through the model (sampling a different $\boldsymbol{\varepsilon}$ each time) and aggregates:

$$\hat{\boldsymbol{\theta}} = \frac{1}{N_{\text{MC}}} \sum_{k=1}^{N_{\text{MC}}} g(\boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}^{(k)}) \qquad \text{(posterior mean)}$$

$$\hat{\boldsymbol{\sigma}}_\theta = \text{std}_{k}\!\left[g(\boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}^{(k)})\right] \qquad \text{(posterior standard deviation)}$$

The 95% credible interval for parameter $\theta_j$ is:

$$\text{CI}_{95} = \left[\hat{\theta}_j - 1.96\,\hat{\sigma}_{\theta_j},\quad \hat{\theta}_j + 1.96\,\hat{\sigma}_{\theta_j}\right]$$

### 4.9 What PIVA Should Theoretically Achieve

A well-functioning PIVA should satisfy:

**Coverage:** Across all test voxels,
$$\Pr\!\left(\theta_j^{\text{true}} \in \left[\hat{\theta}_j - 1.96\,\hat{\sigma}_{\theta_j},\ \hat{\theta}_j + 1.96\,\hat{\sigma}_{\theta_j}\right]\right) \approx 0.95$$

**Calibration:** The posterior standard deviation should predict actual error:
$$\mathbb{E}\!\left[|\hat{\theta}_j - \theta_j^{\text{true}}|\right] \approx \hat{\sigma}_{\theta_j}$$

In other words, Pearson $r(\hat{\sigma}_{\theta_j},\ |\hat{\theta}_j - \theta_j^{\text{true}}|) > 0$, ideally approaching 1.

**Posterior quality:** The per-voxel $\sigma_z$ (in latent space) should be meaningfully above zero:
$$\sigma_{z,j} \in [0.2,\ 0.8] \quad \text{for all } j = 1,\ldots,9$$

If $\sigma_z \approx 0$: posterior = point mass, PIVA reduces to a noisy PIA. If $\sigma_z \approx 1$: posterior = prior, encoder learned nothing.

**Accuracy not sacrificed:** The posterior mean should match or exceed PIA in MAE and Pearson $r$ across all 9 parameters, since the mean of a well-calibrated distribution is at least as good as a point estimate.

---

## 5. Results Analysis

### 5.1 NLLS Baseline

| $\sigma$ | $D\ R^2$ | $T_2\ R^2$ | $v\ R^2$ | $D$ MAE | $T_2$ MAE | $v$ MAE |
|---|---|---|---|---|---|---|
| 0.01 | 0.942 | 0.834 | 0.432 | 0.189 | 80.1 | 0.102 |
| 0.03 | 0.932 | 0.817 | 0.456 | 0.210 | 88.3 | 0.106 |
| 0.05 | 0.923 | 0.810 | 0.238 | 0.225 | 91.8 | 0.123 |
| 0.08 | 0.916 | 0.812 | **−0.137** | 0.240 | 93.2 | 0.153 |

**Reading:** D recovery is acceptable. T2 MAE is large — 91.8 ms at clinical SNR, which is ~12% of the lumen T2 range. Volume fraction recovery is the critical weakness: $R^2_v = -0.137$ at $\sigma = 0.08$ means NLLS predictions of $v$ are worse than simply predicting the population mean for every voxel. This is not a failure of the algorithm — it is a consequence of the fundamental ill-conditionedness of the inverse problem at high noise.

### 5.2 PIA

| $\sigma$ | $D\ R^2$ | $T_2\ R^2$ | $v\ R^2$ | $D$ MAE | $T_2$ MAE | $v$ MAE |
|---|---|---|---|---|---|---|
| 0.01 | 0.988 | 0.960 | 0.813 | 0.079 | 34.9 | 0.062 |
| 0.03 | 0.986 | 0.961 | 0.696 | 0.083 | 34.8 | 0.078 |
| 0.05 | 0.983 | 0.968 | 0.572 | 0.093 | 32.6 | 0.093 |
| 0.08 | 0.978 | 0.959 | 0.480 | 0.103 | 36.7 | 0.106 |

**Reading:** PIA comprehensively outperforms NLLS on every metric at every noise level. T2 MAE improves by 65% (91.8 → 32.6 ms). Volume fraction R² improves by 140% (0.238 → 0.572) at $\sigma = 0.05$. The model is also noise-robust: D R² drops only from 0.988 to 0.978 across the full SNR range (vs NLLS: 0.942 to 0.916).

However, note that PIA's D and T2 MAE **plateau and slightly creep** from epoch 30 onward during training, while signal MSE keeps decreasing. This is the identifiability signature — the encoder finds signal-consistent parameter combinations that are not the ground truth.

### 5.3 PIVA Variants — Original Implementation (Shallow Encoder, tanh)

At $\sigma = 0.05$ (clinical reference):

| Model | $D\ R^2$ | $T_2\ R^2$ | $v\ R^2$ | $D$ MAE | $T_2$ MAE | $v$ MAE |
|---|---|---|---|---|---|---|
| PIA | **0.983** | **0.968** | **0.572** | **0.093** | **32.6** | **0.093** |
| PIVA-Tight | 0.952 | 0.788 | 0.254 | 0.173 | 94.9 | 0.128 |
| PIVA-Wide | 0.819 | 0.797 | 0.111 | 0.342 | 85.9 | 0.139 |
| PIVA-Free | −1.483 | −0.658 | −2.231 | 1.345 | 303.9 | 0.285 |

**PIVA-Tight:** All metrics are worse than PIA. Critically, T2 R² = 0.788 and T2 MAE = 94.9 ms — both **worse than NLLS** (0.810 and 91.8 ms). The original PIVA-Tight is the worst-performing model for T2 and $v$ recovery.

**PIVA-Wide:** Doubling the tanh half-widths worsens every metric further. The optimization landscape becomes harder — the encoder must learn to hit targets that require more extreme tanh values, leading to more saturation and poorer gradients.

**PIVA-Free:** Complete failure. KL = 330–350 throughout training (vs ~30 for PIVA-Tight). The Softplus decoder has no upper bound, so z grows without constraint. The KL term — computing distance from N(0,I) — explodes as z drifts arbitrarily far. D MAE = 1.3 and T2 MAE = 303 ms are non-functional. The reconstruction term never manages to drive useful learning.

### 5.4 PIVA β-Sweep — Improved Architecture (Deep Encoder, Predictor Heads)

After fixing the encoder depth to match PIA ($16 \to 64 \to 128 \to 256 \to 512 \to 512$) and adding predictor-style mu/logvar heads ($512 \to 512 \to 512 \to 9$):

| $\beta_{\max}$ | $D\ R^2$ | $T_2\ R^2$ | $v\ R^2$ | $D$ MAE | $T_2$ MAE | $v$ MAE |
|---|---|---|---|---|---|---|
| 0.01 | **0.988** | **0.968** | 0.568 | **0.076** | 32.8 | 0.098 |
| 0.10 | 0.987 | 0.968 | **0.580** | 0.079 | 33.1 | 0.096 |
| 0.30 | 0.987 | 0.907 | 0.553 | 0.084 | 57.0 | 0.100 |
| 1.00 | 0.984 | 0.882 | 0.571 | 0.089 | 65.6 | 0.099 |
| 3.00 | 0.939 | 0.915 | 0.143 | 0.179 | 57.6 | 0.132 |

The architectural fix closes the entire gap with PIA at $\beta \in \{0.01, 0.1\}$. PIVA-Tight at $\beta = 0.01$ achieves D MAE = 0.076 — actually **better than PIA** (0.093). T2 R² = 0.968 matches PIA exactly.

The T2 MAE deterioration at $\beta \geq 0.3$ (32 → 57 → 65 ms) traces directly to the tanh saturation mechanism analyzed in Section 6.

### 5.5 Uncertainty Quantification Results

**Posterior σ magnitude:**

| $\beta_{\max}$ | $D\ \bar{\sigma}$ | $T_2\ \bar{\sigma}$ (ms) | $v\ \bar{\sigma}$ | Mean CV ($\bar{\sigma}/|\bar{\mu}|$) |
|---|---|---|---|---|
| 0.01 | 0.00523 | 1.76 | 0.00456 | ~0.007 |
| 0.10 | 0.00844 | 4.42 | 0.00451 | ~0.011 |
| 0.30 | 0.01387 | 8.47 | 0.00456 | ~0.018 |
| 1.00 | 0.02264 | 13.15 | 0.00591 | ~0.030 |
| 3.00 | 0.04412 | 23.20 | 0.00941 | ~0.060 |

The posterior σ increases monotonically with $\beta$ — more KL pressure does widen the intervals. But even at $\beta = 3.0$, D posterior σ = 0.044 × 10⁻³ mm²/s. The mean D MAE at $\beta = 3.0$ is 0.179. So the posterior σ is still 4× smaller than the mean error.

**95% CI Coverage:**

| $\beta_{\max}$ | D coverage | $T_2$ coverage | $v$ coverage |
|---|---|---|---|
| 0.01 | 0.086 | 0.082 | 0.067 |
| 0.10 | 0.151 | 0.104 | 0.068 |
| 0.30 | 0.210 | 0.101 | 0.047 |
| 1.00 | 0.307 | 0.131 | 0.061 |
| 3.00 | 0.349 | 0.219 | 0.090 |
| **Target** | **0.950** | **0.950** | **0.950** |

Best achieved coverage is D = 0.349 at $\beta = 3.0$ — at which point D R² has dropped to 0.939 and v R² has collapsed to 0.143. There is no $\beta$ that simultaneously achieves good coverage and acceptable accuracy.

**Calibration Pearson $r$:**

All D calibration Pearson r values are **negative** across all $\beta$:

| $\beta_{\max}$ | D cal $r$ | $T_2$ cal $r$ | $v$ cal $r$ |
|---|---|---|---|
| 0.01 | −0.276 | −0.106 | +0.255 |
| 0.10 | −0.121 | −0.040 | +0.267 |
| 3.00 | −0.074 | −0.328 | +0.087 |

Negative calibration means voxels where the model reports higher posterior σ actually have **lower prediction error**. The uncertainty is anti-correlated with the actual difficulty of each voxel.

---

## 6. Diagnosis: What is Wrong

### 6.1 The Tanh Jacobian Collapse

This is the primary failure mode. The parameter decoding is:

$$D_c = D_{\text{mean},c} + D_{\delta,c} \cdot \tanh(z_c)$$

The Jacobian of this mapping with respect to $z_c$ is:

$$\frac{\partial D_c}{\partial z_c} = D_{\delta,c} \cdot \underbrace{\left(1 - \tanh^2(z_c)\right)}_{\to\ 0\ \text{when}\ |z_c| \gg 1}$$

The encoder learns to produce large $|\mu_c|$ values for confident predictions (which is optimal for reconstruction). Once $|\mu_c| \geq 3$, $\tanh(\mu_c) \approx \pm 0.995$ and the Jacobian is $\approx 0.01 \cdot D_{\delta,c}$. The uncertainty $\sigma_{z,c}$ is then **compressed by a factor of 100** when passed through the tanh to parameter space:

$$\sigma_{D_c} \approx D_{\delta,c} \cdot \left(1 - \tanh^2(\mu_c)\right) \cdot \sigma_{z,c} \approx 0$$

This explains why CI coverage is near zero despite KL ≈ 30 during training. The KL is large because $\mu$ is large (not because $\sigma_z$ is large), and the large $\mu$ saturates the tanh, killing all uncertainty in parameter space.

### 6.2 KL Decomposition — μ vs σ Contribution

The per-dimension KL can be written as:

$$\text{KL}_j = \underbrace{\frac{1}{2}\mu_j^2}_{\text{from}\ \mu} + \underbrace{\frac{1}{2}\left(\sigma_j^2 - 1 - \log \sigma_j^2\right)}_{\text{from}\ \sigma}$$

The second term is minimized at $\sigma_j = 1$ (where it equals 0). If $\sigma_j \ll 1$, the $\sigma$-term is:

$$\frac{1}{2}(\sigma_j^2 - 1 - \log \sigma_j^2) \approx \frac{1}{2}(-\log \sigma_j^2) = -\log \sigma_j \to \infty \text{ as } \sigma_j \to 0$$

So very small $\sigma_j$ should be penalized. But in practice, the optimizer balances $\mu_j^2/2$ against reconstruction gain, and finds it easier to increase $|\mu_j|$ while keeping $\sigma_j$ small. The reconstruction improvement from confident predictions (large $|\mu|$) outweighs the KL penalty from small $\sigma$ — especially at low $\beta$.

With $\text{KL} \approx 30$ and 9 dimensions: $\text{KL}/\text{dim} \approx 3.3$ nats. If this comes entirely from $\mu$: $\mu_j \approx \sqrt{2 \times 3.3} \approx 2.6$. At $|\mu| = 2.6$, $\tanh^2(2.6) = 0.987$, so Jacobian $= 1 - 0.987 = 0.013$. Even if $\sigma_{z} = 0.5$, the parameter-space $\sigma_D \approx D_\delta \times 0.013 \times 0.5 \approx 0.001$ — still near-zero coverage.

### 6.3 No Floor on Per-Dimension σ

Nothing in the current training objective prevents any individual $\sigma_{z,j}$ from collapsing to zero. The optimizer can simultaneously:
- Increase $|\mu_j|$ to improve reconstruction (reducing signal MSE)
- Decrease $\sigma_{z,j}$ toward zero (slightly increasing KL, but less than the reconstruction gain)

There is no lower bound on $\sigma_{z,j}$. Free bits (a minimum KL per dimension) would impose such a bound.

### 6.4 T2 Collapse at High β

The T2_MAE deterioration at $\beta \geq 0.3$ (32 → 57 → 65 ms) is a consequence of the same mechanism in reverse: higher $\beta$ penalizes large $|\mu|$, pushing $\mu_{T2} \to 0$. Via the tanh mapping, $\tanh(0) = 0$ and $T_{2,c} \to T_{2,\text{mean},c}$. The model predicts the physiological mean T2 for every voxel — which has MAE equal to the standard deviation of T2 in the dataset. T2_lu has the largest standard deviation (~200 ms), so it suffers most.

This is the core tension: lower $\beta$ allows large $|\mu|$ (good reconstruction, zero uncertainty), higher $\beta$ forces small $|\mu|$ (poor reconstruction, but wider intervals — though these are still tiny in parameter space due to the Jacobian collapse).

### 6.5 Anti-Calibration — Why Is r Negative?

The negative Pearson r between $\hat{\sigma}_D$ and $|$error$|$ is a structural consequence of the tanh geometry. At the extremes of the tanh (high $|\mu|$):
- The model is most "confident" ($\sigma_D$ large because $|\mu|$ is large)... no wait.

Actually, consider two voxels:
- Voxel A: True D near the physiological centre. $\mu_D$ is moderate, tanh Jacobian is large, $\sigma_D$ is larger.
- Voxel B: True D near the physiological boundary. $\mu_D$ is large (tanh saturated), Jacobian is near zero, $\sigma_D \approx 0$.

But voxel A (centre) is easier to predict — MAE is lower. Voxel B (boundary) has near-zero $\sigma_D$ but may have higher MAE because parameter recovery at the physiological boundary is harder. This creates a negative correlation: high $\sigma_D$ (voxels near centre) → lower error; low $\sigma_D$ (voxels near boundary via tanh saturation) → higher error. The uncertainty estimate is measuring tanh geometry, not true posterior uncertainty.

---

## 7. The Path Forward

### 7.1 Structural Fix: Replace tanh with Linear + Clamp

$$D_c = \text{clamp}\!\left(D_{\text{mean},c} + D_{\delta,c} \cdot z_c,\ D_{\text{low},c},\ D_{\text{high},c}\right)$$

Jacobian = $D_{\delta,c}$ (constant, never zero). Uncertainty in $z_c$ propagates directly:

$$\sigma_{D_c} = D_{\delta,c} \cdot \sigma_{z,c}$$

With $D_{\delta,ep} = 0.2$ and $\sigma_{z} = 0.5$: $\sigma_{D_{ep}} = 0.10 \times 10^{-3}\ \text{mm}^2/\text{s}$ — a physically meaningful interval.

### 7.2 Free Bits: Force Non-Trivial Posterior

Floor the per-dimension KL at $\lambda_{\text{free}}$ nats:

$$D_{\text{KL,free}}[q\ \|\ p] = \sum_{j=1}^{9} \max\!\left(\lambda_{\text{free}},\ \text{KL}_j\right)$$

With $\lambda_{\text{free}} = 1.0$: each $\sigma_{z,j}$ is prevented from collapsing below $\approx 0.76$ (when $\mu_j = 0$). The optimizer cannot reduce $\sigma_{z,j}$ further because the gradient through $\max(\lambda, \text{KL}_j)$ is zero once KL$_j < \lambda$.

### 7.3 Post-Hoc Temperature Scaling

Even after the structural fix, some gap from 0.95 coverage will remain. Temperature scaling finds a scalar $T > 1$ such that:

$$\text{CI}_{95} = \left[\hat{\theta}_j - 1.96 \cdot T \cdot \hat{\sigma}_{\theta_j},\ \hat{\theta}_j + 1.96 \cdot T \cdot \hat{\sigma}_{\theta_j}\right]$$

achieves exactly 0.95 coverage on a held-out calibration set. This is a conformal prediction result — the calibrated CI has valid frequentist coverage guarantees. The temperature $T$ quantifies how much the model underestimates uncertainty (currently $T \approx 6$–$10$ for D based on observed coverage vs target).

---

## 8. Summary

| Aspect | NLLS | PIA | PIVA (current) | PIVA (target) |
|---|---|---|---|---|
| D accuracy | Good | Excellent | Excellent (deep) | Excellent |
| T2 accuracy | Moderate | Excellent | Excellent (low β) | Excellent |
| v accuracy | Poor | Good | Good (low β) | Good |
| Uncertainty | None | None | Near-zero CI | Calibrated CI |
| Coverage | N/A | N/A | 0.086–0.349 | ≥ 0.90 |
| Calibration | N/A | N/A | Negative (anti-correlated) | Positive, slope ≈ 1 |
| $\sigma_z$ | N/A | N/A | ~0.007 (collapsed) | 0.3–0.7 |

The notebook demonstrates that:

1. PIVA with the correct architecture (deep encoder, predictor heads) achieves equivalent point estimation accuracy to PIA.
2. The variational component currently adds no practical value — the posterior is functionally a point mass.
3. The failure is structural (tanh Jacobian) and objective-level (no KL floor), not a hyperparameter tuning problem.
4. The fix requires changing one function (`params_from_z`: tanh → linear + clamp) and one loss term (standard KL → free bits KL). Neither requires architectural changes to the encoder or the training loop.
5. Once fixed, PIVA's uncertainty estimates should be both meaningful (σ_z > 0) and calibrated (coverage → 0.95), which no deterministic method — PIA or NLLS — can provide.

---

*This document was produced as part of the PIVA development cycle for 3-compartment prostate MRI parameter estimation.*
