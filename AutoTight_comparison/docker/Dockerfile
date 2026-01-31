ARG BASE_IMAGE=mambaorg/micromamba:1.5.8
FROM ${BASE_IMAGE}

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV MPLBACKEND=Agg
ENV PIP_NO_CACHE_DIR=1

# 1) Create a minimal runtime env first (no editable installs here).
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.runtime.yml /tmp/environment.runtime.yml
RUN micromamba create -y -n cl -f /tmp/environment.runtime.yml && micromamba clean --all -y

# 2) Copy source (expects submodules already present in the build context).
WORKDIR /workspace/constraint_learning
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# 3) Editable installs for local packages.
RUN python -m pip install -e poly_matrix -e certifiable-tools -e .

CMD ["python", "demo_quick.py"]
