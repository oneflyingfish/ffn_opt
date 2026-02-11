from torch import Tensor
import torch
import triton
import triton.language as tl
from utils_inference.bench.cuda_bench import test_time_cuda


@triton.jit
def add_kernel_triton(x_ptr, y_ptr, o_ptr, elements, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0)

    data_offset = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data_mask = tl.where(data_offset < elements, True, False)

    x = tl.load(x_ptr + data_offset, mask=data_mask)
    y = tl.load(y_ptr + data_offset, mask=data_mask)
    output = x + y
    tl.store(o_ptr + data_offset, output, mask=data_mask)


def add(x: Tensor, y: Tensor) -> Tensor:
    output = torch.empty_like(x)

    assert x.is_cuda and y.is_cuda and output.is_cuda

    elements = x.numel()

    grid = lambda meta: (triton.cdiv(elements, meta["BLOCK_SIZE"]),)
    add_kernel_triton[grid](x, y, output, elements, BLOCK_SIZE=1024)
    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[
            2**i for i in range(10, 30, 1)
        ],  # Different possible values for `x_name`. `x_name` 的不同可能值。
        x_log=True,  # x axis is logarithmic. x 轴为对数。
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot. 参数名称，其值对应于绘图中的不同线条。
        line_vals=[
            "triton",
            "torch",
        ],  # Possible values for `line_arg`. `line_arg` 的可能值。
        line_names=["Triton", "Torch"],  # Label name for the lines. 线条的标签名称。
        styles=[("blue", "-"), ("green", "-")],  # Line styles. 线条样式。
        xlabel="data size",
        ylabel="GB/s",
        plot_name="benchmark add",
        args={},  # Values for function arguments not in `x_names` and `y_name`. 不在 `x_names` 和 `y_name` 中的函数参数值。
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (10, 98432)
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(output_torch - output_triton))}"
    )

    benchmark.run(print_data=True, save_path="data/benchmark")
