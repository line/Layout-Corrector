FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG WORKDIR="./"

RUN apt update -qq && apt install --no-install-recommends -y \
    git curl make emacs wget libgtk2.0-dev libgl1-mesa-dev openssh-server \
    python3 python3-dev python3-pip python-is-python3 && \
    rm -rf /var/cache/apt/*

WORKDIR ${WORKDIR}
COPY ./requirements.txt ${WORKDIR}
COPY ./requirements_extra.txt ${WORKDIR}
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install -r requirements_extra.txt && \
    rm -rf /root/.cache/pip
