import functools
import torch
import triton
import triton.language as tl
from triton.testing import perf_report, Benchmark, do_bench
import matplotlib.pyplot as plt
import numpy as np

# Forward kernel
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def layernorm_fwd_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    mean_ptr, rstd_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    xhat = xc * rstd

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = xhat * w + b

    tl.store(y_ptr + row * N + offs, y, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


# Backward kernel
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def layernorm_bwd_kernel(
    x_ptr, w_ptr,
    dy_ptr,
    dx_ptr, dw_ptr, db_ptr,
    mean_ptr, rstd_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    xhat = (x - mean) * rstd
    dyhat = dy * w

    sum_dy = tl.sum(dyhat, axis=0)
    sum_dy_xhat = tl.sum(dyhat * xhat, axis=0)

    dx = (dyhat - sum_dy / N - xhat * sum_dy_xhat / N) * rstd
    tl.store(dx_ptr + row * N + offs, dx, mask=mask)

    tl.atomic_add(dw_ptr + offs, dyhat * xhat, mask=mask)
    tl.atomic_add(db_ptr + offs, dyhat, mask=mask)


# Pytorch solution
def layernorm_forward_torch(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * rstd
    
    # 4. Применяем scale (weight) и shift (bias)
    output = x_hat * weight + bias
    
    return output

# Triton forward
def layernorm_forward(x, w, b, eps=1e-5):
    if not x.is_contiguous():
        x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)

    layernorm_fwd_kernel[(M,)](
        x, w, b, y, mean, rstd,
        N=N, eps=eps,
    )
    return y, (x, w, b, mean, rstd)

# Triton backward
def layernorm_backward(dy, ctx):
    x, w, b, mean, rstd = ctx
    if not dy.is_contiguous():
        dy = dy.contiguous()
    M, N = x.shape

    dx = torch.empty_like(x)
    dw = torch.zeros_like(w, dtype=torch.float32)
    db = torch.zeros_like(b, dtype=torch.float32)

    layernorm_bwd_kernel[(M,)](
        x, w, dy, dx, dw, db, mean, rstd,
        N=N,
    )
    return dx, dw, db


HIDDEN = 1024

def make_inputs(n_elements, device="cuda", dtype=torch.bfloat16):
    M = max(1, n_elements // HIDDEN)
    N = HIDDEN
    x = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=dtype)
    b = torch.randn(N, device=device, dtype=dtype)
    return x, w, b

_PROVIDER = {
    "Triton": lambda x, w, b: layernorm_forward(x, w, b)[0],
    "PyTorch": lambda x, w, b: layernorm_torch(x, w, b),
    "PyTorch (compile)": torch.compile(lambda x, w, b: layernorm_torch(x, w, b)),
}

@perf_report(
    Benchmark(
        x_names=["n_elements"],
        x_vals=[2**i for i in range(20, 26)],  # от ~1M до 32M элементов
        line_arg="provider",
        line_vals=list(_PROVIDER.keys()),
        line_names=list(_PROVIDER.keys()),
        styles=[("blue", "-"), ("red", "--"), ("green", "-.")],
        ylabel="Latency (ms)",
        plot_name="layernorm_forward_latency",
        args={},
    )
)
def benchmark(n_elements, provider):
    x, w, b = make_inputs(n_elements)
    fn = functools.partial(_PROVIDER[provider], x, w, b)
    ms, min_ms, max_ms = do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return ms, min_ms, max_ms


def run_benchmark():
    n_elements_list = [2**i for i in range(20, 26)]
    results = {name: [] for name in _PROVIDER}

    for n in n_elements_list:
        x, w, b = make_inputs(n)
        for name, fn in _PROVIDER.items():
            fn_partial = functools.partial(fn, x, w, b)
            ms, _, _ = do_bench(fn_partial, quantiles=[0.5])
            results[name].append(ms)

    # Вывод ускорения
    for i, n in enumerate(n_elements_list):
        speedup = results["PyTorch"][i] / results["Triton"][i]
        print(f"n_elements={n:8d} : Triton speedup over PyTorch = {speedup:.2f}x")


if __name__ == "__main__":
    test_correctness(dtype=torch.float32)
    test_correctness(dtype=torch.bfloat16)

    run_benchmark()
