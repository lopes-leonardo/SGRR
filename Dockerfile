from pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-get update

RUN pip install numpy scikit-learn scipy cython

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

RUN pip install torch_geometric

COPY ./data/* /workspace/data/
COPY ./sgrr/*.py /workspace/sgrr/
COPY ./sgrr/sgcc/*.py /workspace/sgrr/sgcc/
COPY ./sgrr/sgcc/lhrr/* /workspace/sgrr/sgcc/lhrr/
COPY ./makefile /workspace/
COPY ./example.py /workspace/

WORKDIR /workspace

RUN python3 sgrr/sgcc/lhrr/setup.py build_ext --inplace