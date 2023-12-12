FROM nvcr.io/nvidia/pytorch:22.08-py3
RUN apt-get update && apt-get -y install git ca-certificates && apt-get clean


# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
#RUN pip install tensorboard cmake onnx   # cmake from apt-get is too old


# install detectron2
# Install g++ and other build dependencies
RUN apt-get update && apt-get install -y g++ build-essential
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
# Install detectron2
RUN pip install 'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
RUN pip install torch==2.0.1+cu117 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
