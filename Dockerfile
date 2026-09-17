# GPU serving/benchmark image: NVIDIA PyTorch container with a pinned TensorRT.
# Build:  docker build -t dr_uq:trt .
# Run:    docker run --gpus all -v $PWD/runs:/app/runs dr_uq:trt dr-uq grade --engine runs/aptos-resnet50-s0/model_fp16.plan --tau 0.5 sample.png
ARG BASE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${BASE}
ARG TRT_VERSION=10.11.0.33
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 DR_UQ_DATA_ROOT=/data
WORKDIR /app
COPY pyproject.toml README.md ./
COPY dr_uq ./dr_uq
COPY configs ./configs
COPY scripts ./scripts
COPY docs ./docs
RUN python -m pip install --upgrade pip \
 && python -m pip install "tensorrt==${TRT_VERSION}" pycuda \
 && python -m pip install -e .
ENTRYPOINT []
CMD ["dr-uq", "--help"]
