FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR \
    && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -n ppi-graphomer python=3.9.18

WORKDIR /app
COPY . .

RUN conda run -n ppi-graphomer pip install --no-cache-dir -r requirements.txt

RUN wget https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp39-cp39-linux_x86_64.whl \
    && conda run -n ppi-graphomer pip install torch_scatter-2.1.2+pt21cu121-cp39-cp39-linux_x86_64.whl \
    && rm torch_scatter-2.1.2+pt21cu121-cp39-cp39-linux_x86_64.whl

SHELL ["conda", "run", "-n", "ppi-graphomer", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "-n", "ppi-graphomer", "python"]
CMD ["--help"]  # 默认显示帮助信息
