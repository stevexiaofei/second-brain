---
title: PyTorch 构建与运行
type: experience
status: evergreen
tags: [pytorch, build, devops]
created: 2026-08-11
updated: 2026-08-11
---

# PyTorch 构建与运行

## Problem

从源码构建 PyTorch 涉及多个后端（CPU/CUDA/ROCm/XPU/MPS）、大量第三方子模块、代码生成（torchgen）和 C++17 编译要求，初次构建容易踩坑。本文记录前置条件、构建步骤、关键环境变量、Docker 用法、测试入口与文档构建的实践经验。

## Environment

- **Python**：3.9 或更高（`setup.py` 的 `python_min_version = (3, 9, 0)`）
- **编译器**：完全支持 C++17 的编译器（clang 或 gcc；Linux 上 gcc 9.4.0+）；Windows 上 Visual Studio 或 Visual Studio Build Tool
- **构建工具**：CMake、Ninja（推荐，比 Make 显著更快）
- **版本**：仓库 `pytorch-main`，`version.txt` 记为 `2.9.0a0`

## Solution

### 1. 获取源码（含子模块）

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

子模块（`gloo`、`cpuinfo`、`onnx`、`fbgemm`、`cutlass` 等）是编译时必需的，`--recursive` 不能漏。

### 2. 安装依赖

```bash
# 通用
conda install cmake ninja
pip install -r requirements.txt

# Linux
pip install mkl-static mkl-include
# CUDA LAPACK（可选）
.ci/docker/common/install_magma_conda.sh 12.4
# Inductor/Triton（可选）
make triton

# macOS（intel x86 机器）
pip install mkl-static mkl-include
# 分布式
conda install pkg-config libuv

# Windows
pip install mkl-static mkl-include
conda install -c conda-forge libuv
```

### 3. 安装 PyTorch（可编辑构建）

```bash
# Linux / macOS
python -m pip install --no-build-isolation -v -e .

# AMD ROCm 先运行
python tools/amd_build/build_amd.py
```

`--no-build-isolation` 是关键：它让构建使用当前环境的 CMake/Ninja，而不是 pip 隔离环境里重新装一遍。`-e` 做可编辑安装，修改 Python 代码立即生效；C++ 改动需要重新构建扩展模块。

### 4. 关键环境变量

构建行为通过环境变量控制（完整列表见 `setup.py` 顶部注释）：

| 变量 | 作用 |
| ---- | ---- |
| `USE_CUDA=0` | 禁用 CUDA 构建 |
| `USE_ROCM=0` | 禁用 ROCm |
| `USE_XPU=0` | 禁用 Intel GPU |
| `USE_DISTRIBUTED=0` | 禁用分布式 |
| `DEBUG` | `-O0 -g` 调试构建 |
| `REL_WITH_DEB_INFO` | 优化 + 调试符号 |
| `MAX_JOBS` | 最大编译作业数 |
| `CMAKE_FRESH=1` | 强制重新运行 cmake 配置 |
| `CMAKE_ONLY=1` | 只运行 cmake 不构建 |
| `TORCH_CUDA_ARCH_LIST` | 指定 CUDA 架构（如 `"6.0;7.0"`） |
| `PYTORCH_ROCM_ARCH` | 指定 AMD GPU 目标（如 `"gfx900;gfx906"`） |
| `TORCH_XPU_ARCH_LIST` | 指定 XPU 架构 |
| `USE_SYSTEM_LIBS` | 使用系统提供的库 |

实践经验：
- **纯 CPU 调试**：`USE_CUDA=0 DEBUG=1 python -m pip install --no-build-isolation -v -e .`，构建快、体积小，适合改 C++ 逻辑。
- **限制并发**：内存不足时设 `MAX_JOBS=4`，否则 ninja 会吃满内存被 OOM kill。
- **指定 CUDA 架构**：不设 `TORCH_CUDA_ARCH_LIST` 会为所有支持的架构编译，非常慢；只设当前 GPU 的架构可大幅提速。
- **改了 CMake 配置不生效**：用 `CMAKE_FRESH=1` 强制重新配置。

### 5. Docker

```bash
# 使用预构建镜像
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest

# 自行构建
make -f docker.Makefile
```

注意：PyTorch 用共享内存跨进程共享数据（DataLoader 的多进程、分布式训练），必须用 `--ipc=host` 或 `--shm-size` 增加共享内存，否则多进程会因 `shm` 不足崩溃。

### 6. 构建文档

```bash
cd docs/
pip install -r requirements.txt
make html
make serve

# PDF
make latexpdf
```

需要 Sphinx 与 `pytorch_sphinx_theme2`；若遇 katex 错误运行 `npm install katex`。

### 7. 测试

测试位于 `test/`。主要入口：

| 入口 | 范围 |
| ---- | ---- |
| `test/run_test.py` | 测试运行器 |
| `test/test_torch.py` | 核心张量算子测试 |
| `test/test_nn.py` | nn 模块测试 |
| `test/test_autograd.py` | autograd 测试 |
| `test/test_cuda.py` | CUDA 测试 |
| `test/test_jit.py` | TorchScript 测试 |
| `test/test_fx.py` | FX 测试 |
| `test/dynamo/` | Dynamo 测试 |
| `test/cpp/` | C++ 测试（需 `BUILD_TEST=ON`） |
| `test/run_doctests.sh` | doctest 运行 |

### 8. torchrun 入口

`setup.py` 的 `entry_points` 注册了 `torchrun` 控制台脚本，指向 `torch.distributed.run:main`，用于启动分布式训练：

```bash
torchrun --nproc_per_node=4 train_script.py
```

## Lessons Learned

- **子模块必须完整拉取**：`git submodule update --init --recursive` 是构建前提，缺一个子模块都会在 CMake 阶段报错。
- **`--no-build-isolation` 不可省**：否则 pip 会创建隔离环境，重新装 CMake/Ninja，且可能与你装好的 CUDA toolkit 不匹配。
- **Ninja 比 Make 快很多**：PyTorch 源文件上万，Make 的串行调度会浪费大量时间；`conda install ninja` 后 CMake 会自动选用。
- **CUDA 架构列表是构建时间的最大变量**：默认编译所有架构可能要数小时，只编译当前 GPU 架构可降到几十分钟。
- **共享内存是分布式/DataLoader 的隐形坑**：Docker 里不加 `--ipc=host` 会在多进程 DataLoader 时报 `Bus error` 或 `RuntimeError: DataLoader worker ... is killed`。
- **改 C++ 后要重新构建 `_C` 扩展**：可编辑安装只对 Python 代码热生效；C++ 改动需重新跑 `pip install -e .` 或对应的构建命令。
- **构建产物在 `torch/lib/`**：编译出的 `libc10.so`、`libtorch_cpu.so` 等放在这里，`torch._C` 在运行时加载它们。

## Related

- [依赖关系](./pytorch-dependencies/) — `USE_*` 选项对应的第三方库与可选后端
- [代码生成 torchgen](./pytorch-torchgen/) — 构建时由 `caffe2/CMakeLists.txt` 与 `tools/setup_helpers/generate_code.py` 触发
- [整体架构](./pytorch-architecture/) — 构建产出的库如何映射到五层架构

## References

- 构建入口：`setup.py`、`CMakeLists.txt`、`pyproject.toml`
- 构建编排：`caffe2/CMakeLists.txt`、`tools/setup_helpers/generate_code.py`
- 环境变量：`setup.py` 顶部注释
- 测试：`test/` 目录，入口 `test/run_test.py`
- torchrun：`torch/distributed/run.py`（`entry_points` 指向 `torch.distributed.run:main`）
