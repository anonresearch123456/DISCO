FROM continuumio/miniconda3

RUN \
  conda update -n base -c defaults conda -y --quiet && \
  conda clean --all
RUN apt-get update && apt-get install -y python3-opencv

COPY requirements.yaml /requirements.yaml

SHELL ["/bin/bash", "-c"]

RUN \
  conda env create --file /requirements.yaml

RUN conda clean --all && rm /requirements.yaml

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.8
