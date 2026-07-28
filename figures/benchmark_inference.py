"""
Inference Time Benchmark: CPU vs GPU for all methods.
Outputs a comparison table for the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import json

# ── Reconstruct model architectures from checkpoint inspection ──

class PIACNN(nn.Module):
    """CNN-PIA (Stage 1): 4-layer CNN encoder with 1x1 prediction heads."""
    def __init__(self, b_values=[0, 5, 50, 100, 200, 500, 800, 1000], device='cpu'):
        super().__init__()
        self.b_values = torch.tensor(b_values, dtype=torch.float32, device=device)
        self.device = device
        n_b = len(b_values)
        self.features = nn.Sequential(
            nn.Conv2d(n_b, 64, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(),
        )
        self.f_pred = nn.Conv2d(64, 1, 1)
        self.Dt_pred = nn.Conv2d(64, 1, 1)
        self.Dstar_pred = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        feat = self.features(x)
        f = torch.sigmoid(self.f_pred(feat)) * 0.3 + 0.05
        Dt = torch.sigmoid(self.Dt_pred(feat)) * 0.0008 + 0.0007
        Dstar = torch.sigmoid(self.Dstar_pred(feat)) * 0.05 + 0.005
        return f, Dt, Dstar


class ResBlock(nn.Module):
    """Residual block with GroupNorm used in the U-Net Refiner."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.leaky_relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.leaky_relu(x + residual)


class UNetRefiner(nn.Module):
    """Stage 2 U-Net Refiner: takes 3-channel parameter maps, outputs refined maps."""
    def __init__(self):
        super().__init__()
        self.enc1 = ResBlock(3, 64)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)
        self.bottleneck = ResBlock(256, 256)
        self.dec3 = ResBlock(512, 128)  # 256 + 256 skip
        self.dec2 = ResBlock(256, 64)   # 128 + 128 skip
        self.dec1 = ResBlock(128, 64)   # 64 + 64 skip
        self.final_conv = nn.Conv2d(64, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([F.interpolate(b, e3.shape[2:], mode='bilinear', align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False), e1], dim=1))
        return self.final_conv(d1)


# Import MLP-PIA from src
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from PIA import PIA
from utils import funcBiExp, fit_biExponential_model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_nlls_cpu(image, b_values, n_runs=3):
    """Benchmark NLLS (CPU-only, no GPU possible)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fit_biExponential_model(image, b_values)
        times.append(time.perf_counter() - t0)
    return np.median(times)


def benchmark_model_cpu(model, input_tensor, n_warmup=5, n_runs=20):
    """Benchmark a PyTorch model on CPU."""
    model = model.cpu().eval()
    x = input_tensor.cpu()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            times.append(time.perf_counter() - t0)
    return np.median(times)


def benchmark_model_gpu(model_gpu, input_tensor, n_warmup=10, n_runs=50):
    """Benchmark a PyTorch model on GPU with proper CUDA synchronization.
    model_gpu must already be on GPU (use device='cuda' at init for PIA)."""
    if not torch.cuda.is_available():
        return None
    model_gpu = model_gpu.eval()
    x = input_tensor.cuda()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_gpu(x)
            torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model_gpu(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return np.median(times)


def benchmark_pipeline_cpu(stage1, refiner, input_2d, input_4d, n_warmup=5, n_runs=20):
    """Benchmark full 2-stage pipeline on CPU."""
    stage1 = stage1.cpu().eval()
    refiner = refiner.cpu().eval()
    x_2d = input_2d.cpu()
    x_4d = input_4d.cpu()
    is_mlp = isinstance(stage1, PIA)

    with torch.no_grad():
        for _ in range(n_warmup):
            if is_mlp:
                _, _, f, Dt, Ds = stage1(x_2d)
                param_map = torch.stack([f.view(200, 200), Dt.view(200, 200), Ds.view(200, 200)]).unsqueeze(0)
            else:
                f, Dt, Ds = stage1(x_4d)
                param_map = torch.cat([f, Dt, Ds], dim=1)
            _ = refiner(param_map)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            if is_mlp:
                _, _, f, Dt, Ds = stage1(x_2d)
                param_map = torch.stack([f.view(200, 200), Dt.view(200, 200), Ds.view(200, 200)]).unsqueeze(0)
            else:
                f, Dt, Ds = stage1(x_4d)
                param_map = torch.cat([f, Dt, Ds], dim=1)
            _ = refiner(param_map)
            times.append(time.perf_counter() - t0)
    return np.median(times)


def benchmark_pipeline_gpu(stage1_gpu, refiner_gpu, input_2d, input_4d, n_warmup=10, n_runs=50):
    """Benchmark full 2-stage pipeline on GPU.
    stage1_gpu and refiner_gpu must already be on GPU."""
    if not torch.cuda.is_available():
        return None
    stage1_gpu = stage1_gpu.eval()
    refiner_gpu = refiner_gpu.eval()
    x_2d = input_2d.cuda()
    x_4d = input_4d.cuda()
    is_mlp = isinstance(stage1_gpu, PIA)

    with torch.no_grad():
        for _ in range(n_warmup):
            if is_mlp:
                _, _, f, Dt, Ds = stage1_gpu(x_2d)
                param_map = torch.stack([f.view(200, 200), Dt.view(200, 200), Ds.view(200, 200)]).unsqueeze(0)
            else:
                f, Dt, Ds = stage1_gpu(x_4d)
                param_map = torch.cat([f, Dt, Ds], dim=1)
            _ = refiner_gpu(param_map)
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if is_mlp:
                _, _, f, Dt, Ds = stage1_gpu(x_2d)
                param_map = torch.stack([f.view(200, 200), Dt.view(200, 200), Ds.view(200, 200)]).unsqueeze(0)
            else:
                f, Dt, Ds = stage1_gpu(x_4d)
                param_map = torch.cat([f, Dt, Ds], dim=1)
            _ = refiner_gpu(param_map)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return np.median(times)


if __name__ == '__main__':
    print("=" * 70)
    print("  PIA-IVIM Inference Time Benchmark")
    print("=" * 70)

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU: Not available (CPU-only benchmarks)")
    print()

    b = np.array([0, 5, 50, 100, 200, 500, 800, 1000], dtype=np.float64)

    # Create synthetic test input (one 200x200 patient slice)
    np.random.seed(42)
    dummy_image = np.random.rand(200, 200, 8).astype(np.float64) * 0.5 + 0.5
    input_2d = torch.randn(200 * 200, 8)           # MLP input: flattened voxels
    input_4d = torch.randn(1, 8, 200, 200)          # CNN input: image with b-value channels

    # ── Instantiate models ──
    mlp_pia = PIA(device='cpu')
    cnn_pia = PIACNN(device='cpu')
    refiner = UNetRefiner()

    # GPU model instances (PIA needs device='cuda' at init for b_values tensor)
    if has_gpu:
        mlp_pia_gpu = PIA(device='cuda')
        cnn_pia_gpu = PIACNN(device='cuda').cuda()
        refiner_gpu = UNetRefiner().cuda()
    else:
        mlp_pia_gpu = cnn_pia_gpu = refiner_gpu = None

    print(f"  Model parameters:")
    print(f"    MLP-PIA:      {count_params(mlp_pia):>10,}")
    print(f"    CNN-PIA:      {count_params(cnn_pia):>10,}")
    print(f"    U-Net Refiner:{count_params(refiner):>10,}")
    print()

    # ── Run benchmarks ──
    results = {}

    # NLLS (CPU only)
    print("  Benchmarking NLLS (CPU)...", flush=True)
    nlls_cpu = benchmark_nlls_cpu(dummy_image, b, n_runs=1)  # NLLS is slow, 1 run
    results['NLLS'] = {'cpu': nlls_cpu, 'gpu': None}
    print(f"    NLLS CPU: {nlls_cpu:.3f}s")

    # MLP-PIA (standalone)
    print("  Benchmarking MLP-PIA...", flush=True)
    mlp_cpu = benchmark_model_cpu(mlp_pia, input_2d)
    mlp_gpu = benchmark_model_gpu(mlp_pia_gpu, input_2d) if has_gpu else None
    results['MLP-PIA'] = {'cpu': mlp_cpu, 'gpu': mlp_gpu}
    print(f"    MLP-PIA CPU: {mlp_cpu:.4f}s  GPU: {mlp_gpu:.4f}s" if mlp_gpu else f"    MLP-PIA CPU: {mlp_cpu:.4f}s")

    # CNN-PIA (standalone)
    print("  Benchmarking CNN-PIA...", flush=True)
    cnn_cpu = benchmark_model_cpu(cnn_pia, input_4d)
    cnn_gpu = benchmark_model_gpu(cnn_pia_gpu, input_4d) if has_gpu else None
    results['CNN-PIA'] = {'cpu': cnn_cpu, 'gpu': cnn_gpu}
    print(f"    CNN-PIA CPU: {cnn_cpu:.4f}s  GPU: {cnn_gpu:.4f}s" if cnn_gpu else f"    CNN-PIA CPU: {cnn_cpu:.4f}s")

    # MLP-PIA + Refiner
    print("  Benchmarking MLP-PIA + Refiner...", flush=True)
    mlp_ref_cpu = benchmark_pipeline_cpu(mlp_pia, refiner, input_2d, input_4d)
    mlp_ref_gpu = benchmark_pipeline_gpu(mlp_pia_gpu, refiner_gpu, input_2d, input_4d) if has_gpu else None
    results['MLP-PIA + Refiner'] = {'cpu': mlp_ref_cpu, 'gpu': mlp_ref_gpu}

    # CNN-PIA + Refiner
    print("  Benchmarking CNN-PIA + Refiner...", flush=True)
    cnn_ref_cpu = benchmark_pipeline_cpu(cnn_pia, refiner, input_2d, input_4d)
    cnn_ref_gpu = benchmark_pipeline_gpu(cnn_pia_gpu, refiner_gpu, input_2d, input_4d) if has_gpu else None
    results['CNN-PIA + Refiner'] = {'cpu': cnn_ref_cpu, 'gpu': cnn_ref_gpu}

    # ── Print table ──
    print()
    print("=" * 70)
    print("  RESULTS: Inference Time per Patient (200×200 slice)")
    print("=" * 70)
    header = f"  {'Method':<25s} {'CPU (s)':>10s} {'GPU (s)':>10s} {'Speedup':>10s}"
    print(header)
    print("  " + "-" * 65)

    nlls_cpu_time = results['NLLS']['cpu']
    for method, times in results.items():
        cpu_str = f"{times['cpu']:.4f}" if times['cpu'] is not None else "N/A"
        gpu_str = f"{times['gpu']:.4f}" if times['gpu'] is not None else "N/A"
        # Speedup = NLLS_CPU / method's best time
        best = times['gpu'] if times['gpu'] is not None else times['cpu']
        speedup = f"{nlls_cpu_time / best:.0f}×" if best else "—"
        if method == 'NLLS':
            speedup = "1×"
        print(f"  {method:<25s} {cpu_str:>10s} {gpu_str:>10s} {speedup:>10s}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'inference_timing.json')
    save_data = {}
    for method, times in results.items():
        save_data[method] = {
            'cpu_seconds': times['cpu'],
            'gpu_seconds': times['gpu'],
        }
    if has_gpu:
        save_data['gpu_name'] = torch.cuda.get_device_name(0)
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
