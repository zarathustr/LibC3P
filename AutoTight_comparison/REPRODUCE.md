# 在其他主机复现安装/环境（含 Docker 最小环境）

本仓库在原始 `utiasASRL/constraint_learning` 基础上做了少量修复，使 **没有 MOSEK license** 的机器也能跑通 `demo_quick.py` / `make results_test`（可能会更慢、tightness/cert 结果可能为 `False`，但不会因为缺字段直接崩溃）。

> 适用平台建议：Linux（Ubuntu 20.04/22.04）。Jetson（aarch64）和常见 x86_64 都已验证可用思路。

---

## 0) 准备代码（务必带 submodules）

### 方式 A：直接拷贝本机目录（最稳）
把源机上的整个目录拷贝到目标机，例如：

```bash
rsync -av --exclude="_results_*" --exclude=".pytest_cache" --exclude="__pycache__" \
  /autocity/jinwu/constraint_learning/  user@HOST:/path/to/constraint_learning/
```

### 方式 B：从 Git 拉取（必须递归 submodules）
如果你有一个包含这些修复的 fork/分支，建议用下面方式拉取：

```bash
git clone --recurse-submodules <YOUR_REPO_URL> constraint_learning
cd constraint_learning
git submodule update --init --recursive
```

> 说明：本项目依赖 `certifiable-tools/` 等子模块，缺失会导致 import 报错。

---

## 1) 方案一：Conda 前缀环境（推荐，最不影响其它项目）

### 1.1 安装 Miniforge/Mambaforge（只需一次）
如果你目标机没有 conda，建议装 Miniforge（更适合 ARM/Jetson）：

```bash
# 任选其一：miniforge / mambaforge（这里不展开下载地址）
```

### 1.2 创建隔离环境（prefix，不改你其它 env）

在仓库根目录执行（很关键：`environment.yml` 里 pip 安装了 `-e .`，需要在仓库根目录跑）：

```bash
cd /path/to/constraint_learning

source ~/miniforge3/etc/profile.d/conda.sh
conda env create -p ./_conda/constraint_learning -f environment.yml
conda activate ./_conda/constraint_learning
```

验证依赖：

```bash
python -c "import cvxpy, cvxopt, numpy, scipy, pandas; import cert_tools, poly_matrix; print('ok')"
```

### 1.3 运行最小 demo（推荐先跑这个）

```bash
python demo_quick.py
```

输出会写到 `_results_test_quick/`。

### 1.4 跑完整小规模流程（`results_test`）

```bash
make results_test
```

> 提示：没有 MOSEK license 时，会自动走开源求解器（可能慢、会有 solver warning，但流程应可跑完）。

### 1.5（可选）指定 CVXPY 求解器/回退

```bash
export CERT_TOOLS_CVXPY_SOLVER=CVXOPT        # 或 SCS / CLARABEL / MOSEK
export CERT_TOOLS_CVXPY_FALLBACK_SOLVER=SCS  # 没 license 或 MOSEK 失败时回退
export CERT_TOOLS_SCS_MAX_ITERS=5000         # 仅 SCS 生效
```

### 1.6（可选）导出“可复现锁定文件”（平台相关）

如果你希望在**同平台/同架构**的机器上更可控地复现版本，可在环境创建完成后导出：

```bash
conda list -p ./_conda/constraint_learning --explicit > conda.lock.txt
python -m pip freeze > pip.lock.txt
```

> `conda.lock.txt` 一般与平台/架构强相关（例如 linux-aarch64 vs linux-64），跨平台时建议仍用 `environment.yml` 重新解依赖。

---

## 2) 方案二：Docker（最小依赖容器）

### 2.0 常见问题：`registry-1.docker.io` 超时

你看到的报错：

```
Get "https://registry-1.docker.io/v2/": ... Client.Timeout exceeded ...
```

通常是网络/代理/DNS 导致 Docker 拉取基础镜像失败，并不是 Dockerfile 语法问题。可按下面顺序处理：

1) 先验证网络是否能访问：

```bash
curl -I https://registry-1.docker.io/v2/
```

2) 配置 Docker 镜像加速/内网缓存（需要 root 权限）：

编辑 `/etc/docker/daemon.json`（没有就新建）：

```json
{
  "registry-mirrors": [
    "https://mirror.gcr.io"
  ]
}
```

然后重启 Docker：

```bash
sudo systemctl restart docker
```

3) 或者直接在构建时换用其它 registry 的同一基础镜像（见 2.1 的 `BASE_IMAGE`）。

4) 如果目标机完全无法访问外网 registry：在一台能联网的机器上先拉取并导出基础镜像，再拷贝到目标机导入：

```bash
# on a machine with access
docker pull mambaorg/micromamba:1.5.8
docker save mambaorg/micromamba:1.5.8 -o micromamba_1.5.8.tar

# on the target machine
docker load -i micromamba_1.5.8.tar
```

### 2.1 构建镜像

确保你的工作目录里已经包含 submodules（见第 0 节），然后：

```bash
cd /path/to/constraint_learning
docker build -f docker/Dockerfile -t constraint-learning:latest .
```

如果你所在网络访问 Docker Hub（`registry-1.docker.io`）不稳定/超时，可换一个镜像源（例如 GHCR）：

```bash
docker build -f docker/Dockerfile -t constraint-learning:latest . \
  --build-arg BASE_IMAGE=ghcr.io/mamba-org/micromamba:1.5.8
```

> 也可以把 `BASE_IMAGE` 指向你们内网的镜像仓库/缓存。

### 2.2 跑 demo

把输出目录挂载出来方便拿结果：

```bash
mkdir -p _results_test_quick
docker run --rm -it \
  -v "$PWD/_results_test_quick:/workspace/constraint_learning/_results_test_quick" \
  constraint-learning:latest \
  python demo_quick.py
```

### 2.3 跑 `results_test`

```bash
mkdir -p _results_test
docker run --rm -it \
  -v "$PWD/_results_test:/workspace/constraint_learning/_results_test" \
  constraint-learning:latest \
  make results_test
```

### 2.4（可选）在 Docker 里切换求解器

```bash
docker run --rm -it \
  -e CERT_TOOLS_CVXPY_FALLBACK_SOLVER=SCS \
  -e CERT_TOOLS_SCS_MAX_ITERS=5000 \
  constraint-learning:latest \
  make results_test
```

---

## 3) MOSEK license（可选）

如果你有 MOSEK license，把 license 放到：

- `~/mosek/mosek.lic`，或
- 设置 `MOSEKLM_LICENSE_FILE=/abs/path/to/mosek.lic`

然后重跑即可（速度/稳定性一般会明显更好）。
