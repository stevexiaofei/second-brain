---
title: pyproject.toml — Python 项目的声明式控制面
type: concept
status: seed
tags: [Python, Packaging, pyproject.toml, PEP-517, PEP-518, PEP-621]
created: 2026-08-31
updated: 2026-08-31
source: Python Packaging 用户指南、PEP 517/518/621 与 PyTorch 源码实践
---

# pyproject.toml — Python 项目的声明式控制面

## 一句话理解

> `pyproject.toml` 是 Python 项目根目录的标准配置入口：它声明**项目是什么**（元数据和运行依赖）、**如何构建**（构建后端及其隔离环境依赖），并集中承载格式化、静态检查等开发工具的配置。

它不是某个包管理器专属的文件；`pip`、`build`、`uv`、Poetry、Hatch 等工具都可围绕它协作。

## 为什么重要

早期 Python 项目通常把信息分散在 `setup.py`、`setup.cfg`、`requirements.txt`、`tox.ini`、`.flake8` 等文件中。分散本身不是错误，但会让下列问题更难判断：

- 安装一个项目时，构建工具到底需要先安装哪些依赖？
- 哪些依赖会随最终 wheel 一起被用户安装，哪些只服务于构建？
- 包的名称、Python 版本范围、命令行入口和可选功能在哪里声明？
- Ruff、Black、pytest 等工具各自应从哪里读取配置？

`pyproject.toml` 提供了一个标准化的顶层入口。它让构建工具能够先建立可复现的构建环境，也让项目配置更接近声明「期望结果」而不是执行任意安装脚本。

## 三层标准：构建依赖、构建接口与项目元数据

| 标准 | 解决的问题 | 在文件中的体现 |
| --- | --- | --- |
| [PEP 518](https://peps.python.org/pep-0518/) | 构建前需要安装什么、由哪个后端构建 | `[build-system]` |
| [PEP 517](https://peps.python.org/pep-0517/) | 前端（如 pip）如何调用构建后端 | `build-backend` 所指向的接口 |
| [PEP 621](https://peps.python.org/pep-0621/) | 项目元数据如何用声明式字段表达 | `[project]`、`[project.urls]` 等 |

三者可以这样理解：

```text
pip / uv / python -m build       ← 构建前端
             │
             │ 依照 PEP 518 建立隔离环境，安装 build-system.requires
             ▼
构建后端（setuptools / hatchling / ...） ← 按 PEP 517 接口生成 sdist / wheel
             │
             └── 读取 PEP 621 的 [project] 元数据（或由后端动态补全）
```

### `pip install .` 时发生什么

对于一个采用现代构建接口的项目，简化流程是：

1. `pip` 读取 `[build-system]`；
2. 在临时、隔离的构建环境中安装 `requires` 列出的包；
3. 导入 `build-backend` 指定的后端；
4. 通过 PEP 517 hook 让后端生成 wheel；
5. 将 wheel 安装到目标环境，并安装该 wheel 声明的运行时依赖。

因此，构建依赖与运行时依赖属于不同阶段，不能因为「本机刚好已安装」就混为一谈。使用 `pip install --no-build-isolation .` 会跳过第 2 步；它适合已手工准备好构建环境的源码开发场景，但会把依赖完整性的责任交给开发者。

## 最小现代库示例

以下是一个适合大多数纯 Python 库的起点：

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "acme-widget"
version = "0.1.0"
description = "A small example library"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Acme Team" }]
dependencies = [
  "httpx>=0.27",
]

[project.optional-dependencies]
test = ["pytest>=8"]
docs = ["mkdocs-material>=9"]

[project.scripts]
acme-widget = "acme_widget.cli:main"
```

这个例子中：

- `[build-system]` 仅描述**为了构建这个分发包**而必须可用的 `hatchling`；
- `[project.dependencies]` 描述用户安装该包后需要的运行时依赖；
- `[project.optional-dependencies]` 为额外功能定义可安装的 extras，例如 `pip install 'acme-widget[test]'`；
- `[project.scripts]` 将命令名映射到 Python 可调用对象，安装时生成平台对应的命令行入口。

`project` 节中常用字段还包括 `classifiers`、`keywords`、`urls`、`entry-points` 和 `gui-scripts`。它们最终进入 wheel 和包索引可见的元数据。

## `dynamic`：由构建后端补齐的字段

某些元数据不适合在 TOML 中写死，例如从 Git 标签生成版本号、从一个文件读取 README，或为了兼容现有构建逻辑而在 `setup.py` 中计算依赖。这时应显式声明：

```toml
[project]
name = "acme-widget"
dynamic = ["version"]

[tool.setuptools.dynamic]
version = { attr = "acme_widget.__version__" }
```

`dynamic` 的意思不是「该字段可有可无」，而是「该字段是项目元数据的一部分，但由后端在构建时提供」。没有静态给值、也没有列入 `dynamic` 的必填字段，会使项目元数据不完整。

实践上，优先让名称、Python 版本约束、依赖等稳定信息保持静态；只把确实需要在构建期计算的字段列为 `dynamic`。动态字段越多，越难在不执行构建逻辑时理解一个项目。

## `[tool.<工具名>]`：工具自己的配置空间

`[tool.*]` 并非 PEP 621 项目元数据的一部分，而是 TOML 为生态工具预留的命名空间。每个工具定义并解析自己的子表：

```toml
[tool.ruff]
line-length = 88
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.black]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
```

这能集中配置文件，却不代表任何工具都会读取所有表：

- Ruff 只关心 `[tool.ruff]`；
- Black 只关心 `[tool.black]`；
- 构建后端通常只关心 `[build-system]`、`[project]` 及自己约定的 `[tool.<backend>]`；
- `pip` 不会因为看到 `[tool.ruff]` 就安装 Ruff。

因此，`pyproject.toml` 是共享配置容器，而非单一语义文件。

## 与锁文件和依赖清单的分工

| 文件 | 主要表达 | 是否应跨环境稳定 |
| --- | --- | --- |
| `pyproject.toml` | 项目的意图：版本范围、构建方式、工具配置 | 是 |
| `uv.lock` / `poetry.lock` | 某时刻解析得到的精确依赖图 | 通常是；取决于工具和项目类型 |
| `requirements.txt` | 一组要安装的包，可能是直接依赖，也可能是冻结结果 | 取决于其生成方式 |

例如，[uv](./uv-python-package-manager.md) 会把项目意图写入 `pyproject.toml`，把可复现的解析结果写入 `uv.lock`。不要手改锁文件来表达「我要升级某包」：应修改依赖约束或运行对应的锁定命令，再由工具更新锁文件。

## PyTorch 案例：复杂构建与 legacy 兼容

PyTorch 根目录的 `pyproject.toml` 展示了大型原生扩展项目的一种现实形态：

```toml
[project]
name = "torch"
requires-python = ">=3.9"
dynamic = ["dependencies", "version", "scripts", "..."]

[build-system]
requires = [
  "setuptools>=62.3.0,<80.0",
  "wheel",
  "ninja",
  "cmake",
  "numpy",
  "...",
]
build-backend = "setuptools.build_meta:__legacy__"
```

它的构建不只是打包 Python 文件，还包括 C++/CUDA 编译、代码生成和多平台后端选择。因此：

- `ninja`、`cmake`、`numpy` 等出现在 `[build-system].requires`，因为构建阶段需要它们；
- 许多项目字段标为 `dynamic`，交给既有的 `setup.py` 构建逻辑计算；
- `setuptools.build_meta:__legacy__` 保留了与传统 `setup.py` 交互的兼容行为；
- `[tool.black]`、`[tool.isort]`、`[tool.ruff]`、`[tool.codespell]` 将大项目的开发质量规则集中于同一文件。

这里的 `__legacy__` 是对已有复杂构建系统的兼容选择，**不是新项目的默认模板**。新项目应先选择支持 PEP 517 的正常构建后端；只有明确依赖传统 `setup.py` 行为时，才评估 legacy backend 的必要性。PyTorch 的实际构建命令、环境变量和后端差异见 [PyTorch 构建与运行](../ai/systems/pytorch/pytorch-build-run.md)。

## 实践检查清单

1. **先区分项目类型**：应用不一定要发布为 wheel；可发布的库通常应完整填写 `[project]`。
2. **明确后端**：不要只复制 `requires`。后端、构建依赖和项目代码必须相互匹配。
3. **区分构建与运行依赖**：构建工具进 `[build-system].requires`，用户运行包所需的库进 `project.dependencies`。
4. **保持关键元数据静态**：能直接写入的版本范围、名称和依赖不要无故放进 `dynamic`。
5. **让工具配置收敛但不耦合**：可将 Ruff、Black、pytest 配置放在同一文件，但仍按各自文档维护字段。
6. **锁文件由工具生成**：提交与否取决于项目策略；提交时应在 CI 中校验其与 `pyproject.toml` 一致。
7. **发布前实际构建**：使用 `python -m build` 或项目管理器的构建命令，验证 sdist/wheel 和干净环境安装。

## 常见误区

- **把 `[build-system].requires` 当运行时依赖**：它们仅保证构建环境可用，不会自动成为最终用户环境的依赖。
- **把 `pyproject.toml` 当锁文件**：约束如 `httpx>=0.27` 不等于锁定某个精确版本。
- **以为所有项目都必须有 `[project]`**：构建型或兼容型项目可以由后端提供元数据；但新的可发布包优先采用 PEP 621 声明式元数据。
- **看到 `[tool.*]` 就以为 pip 会安装对应工具**：配置与依赖声明是两件事。
- **为「现代化」盲目删除 `setup.py`**：先确认所选后端是否承载了自定义构建、扩展编译和版本生成需求。
- **盲目使用 `--no-build-isolation`**：它可能掩盖缺失的构建依赖，只应在刻意管理构建环境时使用。

## 我的理解

- **事实**：PEP 518、517、621 分别标准化了构建依赖发现、构建前后端接口和核心项目元数据；它们共同让不同前端和后端能协作。
- **工程解释**：我把 `pyproject.toml` 看作项目的「声明式控制面」——它不承载全部实现，却决定构建工具如何准备环境、项目以何种形态发布、日常工具按什么规则工作。
- **边界**：它不是万能替代品。PyTorch 一类项目仍需要 CMake、Python 脚本和原生代码来表达复杂构建逻辑；关键在于让静态可声明部分静态化，把必要的动态行为限制在清晰的边界内。

## Related

- [uv — 极速 Python 包管理工具](./uv-python-package-manager.md) — 用 `pyproject.toml` 管理项目意图、以 `uv.lock` 锁定解析结果
- [PyTorch 构建与运行](../ai/systems/pytorch/pytorch-build-run.md) — C++/CUDA 项目如何把构建需求接入 Python 打包流程
- [Python Packaging 用户指南：pyproject.toml specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
- [PEP 517 — A build-system independent format for source trees](https://peps.python.org/pep-0517/)
- [PEP 518 — Specifying Minimum Build System Requirements](https://peps.python.org/pep-0518/)
- [PEP 621 — Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)

## References

- [Python Packaging 用户指南](https://packaging.python.org/)
- [PyPA Specifications](https://packaging.python.org/en/latest/specifications/)
- PyTorch 根目录 `pyproject.toml`（本地源码快照）
