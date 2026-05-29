import functools
import torch
import triton
import triton.language as tl
from triton.testing import do_bench
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


# Triton fwd + bwd
class LayerNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps=1e-5):
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
        ctx.save_for_backward(x, w, b, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, mean, rstd = ctx.saved_tensors
        eps = ctx.eps
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
        return dx, dw.to(w.dtype), db.to(b.dtype), None


# PyTorch solution
def layernorm_forward_torch(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * rstd
    return x_hat * weight + bias


# Correctness test
def test_correctness(dtype=torch.float32, atol=1e-5, rtol=1e-5):
    M, N = 32, 1024
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    w = torch.randn(N, device='cuda', dtype=dtype)
    b = torch.randn(N, device='cuda', dtype=dtype)
    eps = 1e-5

    # Forward
    y_torch = layernorm_forward_torch(x, w, b, eps)
    y_triton = LayerNormTriton.apply(x, w, b, eps)

    torch.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
    print("Forward OK!")

    # Backward
    x.requires_grad_(True)
    w.requires_grad_(True)
    b.requires_grad_(True)

    y_torch = layernorm_forward_torch(x, w, b, eps)
    y_triton = LayerNormTriton.apply(x, w, b, eps)

    grad_out = torch.randn_like(y_torch)

    y_torch.backward(grad_out, retain_graph=True)
    grad_x_torch, grad_w_torch, grad_b_torch = x.grad, w.grad, b.grad

    x.grad, w.grad, b.grad = None, None, None
    y_triton.backward(grad_out)
    grad_x_triton, grad_w_triton, grad_b_triton = x.grad, w.grad, b.grad

    if dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(grad_x_triton, grad_x_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(grad_w_triton, grad_w_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(grad_b_triton, grad_b_torch, atol=atol, rtol=rtol)
    print("Backward OK!")

# Benchmark ig
HIDDEN = 1024

def make_inputs(n_elements, device="cuda", dtype=torch.bfloat16):
    M = max(1, n_elements // HIDDEN)
    N = HIDDEN
    x = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=dtype)
    b = torch.randn(N, device=device, dtype=dtype)
    return x, w, b


def run_benchmark():
    providers = {
        "Triton": lambda x, w, b: LayerNormTriton.apply(x, w, b, eps=1e-5),
        "PyTorch": lambda x, w, b: layernorm_forward_torch(x, w, b, eps=1e-5),
        "PyTorch (compile)": torch.compile(lambda x, w, b: layernorm_forward_torch(x, w, b, eps=1e-5)),
    }

    n_elements_list = [2**i for i in range(20, 26)]
    results = {name: [] for name in providers}

    for n in n_elements_list:
        x, w, b = make_inputs(n, dtype=torch.bfloat16)
        for name, fn in providers.items():
            fn_partial = functools.partial(fn, x, w, b)
            ms = do_bench(fn_partial, quantiles=[0.5])[0]
            results[name].append(ms)
        speedup = results["PyTorch"][-1] / results["Triton"][-1]
        print(f"n_elements={n:8d} : Triton speedup over PyTorch = {speedup:.2f}x")


if __name__ == "__main__":
    test_correctness(dtype=torch.float32)
    test_correctness(dtype=torch.bfloat16, atol=1e-2, rtol=1e-2)
    run_benchmark()
