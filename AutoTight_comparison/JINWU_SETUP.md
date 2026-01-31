# constraint_learning (Jetson /autocity/jinwu) 使用说明

如需在其它主机复现安装/环境（含 Docker 最小环境），见 `REPRODUCE.md`。

## 目录

- 代码：`/autocity/jinwu/constraint_learning`
- Conda 环境（隔离，不影响你其它项目）：`/autocity/jinwu/.conda/constraint_learning`

> 本仓库原始 README 里要求 MOSEK license。这里已做了“无 license 也能跑”的降级：
> - 若未检测到 `~/mosek/mosek.lic` / `MOSEKLM_LICENSE_FILE`，默认用 `CVXOPT`（开源）解 SDP；
> - 若你配置了 license，会优先用 `MOSEK`（更快/更稳）。

## 1) 激活环境

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /autocity/jinwu/.conda/constraint_learning
```

不想 activate 也可以：

```bash
~/miniforge3/bin/conda run -p /autocity/jinwu/.conda/constraint_learning --no-capture-output python -V
```

## 2) 跑通一个“快速 demo”（不需要 MOSEK license）

```bash
cd /autocity/jinwu/constraint_learning
python demo_quick.py
```

输出会写到：`/autocity/jinwu/constraint_learning/_results_test_quick/`。

## 3) 跑官方 results_test（可选，耗时更久）

```bash
cd /autocity/jinwu/constraint_learning
make results_test
```

说明：
- 如果你有 MOSEK license：把 license 放到 `~/mosek/mosek.lic`，或设置
  ```bash
  export MOSEKLM_LICENSE_FILE=/abs/path/to/mosek.lic
  ```
  然后再跑 `make results_test`（速度/稳定性会明显更好）。
- 如果没有 license：会自动降级到 `CVXOPT`/`SCS`，Jetson 上可能比较慢。

## 4) 手动选择 CVXPY 求解器（可选）

```bash
export CERT_TOOLS_CVXPY_SOLVER=CVXOPT   # 或 SCS / CLARABEL / MOSEK
export CERT_TOOLS_SCS_MAX_ITERS=5000    # 仅 SCS 生效
```
