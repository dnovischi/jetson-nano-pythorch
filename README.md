# Jetson Nano Pytorch Build Docker

## Introduction

Building a PyTorch wheel on the Jetson Nano takes forever, while setting up a cross-compilation build system is prone to many failures. Moreover the official wheels (right now) are built only for specific python versions (e.g. python3.6). This creates a bottleneck of workarounds, beginning with simple python environment management and ending with overly complicated containers deployed on the Jetson Nano.

As such, the target is to create a simple and manageable build environment to compile and natively deploy PyTorch for the python version of choice.

The approach taken here is to use a docker container for the desired target Jetson Nano OS running via QEMU that builds the actual package wheel on the host system. Thereby, avoiding cross-compilation package management.

## Prerequisites

1. Install docker by following [this](https://docs.docker.com/engine/install/ubuntu/) guide.

2. Install QEMU packages:

  ```
  sudo apt-get install qemu binfmt-support qemu-user-static
  ```
3. Execute the registering scripts:
  ```
  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
  ```
4. Test the QEMU emulation environment

  ```
  sudo docker run --platform linux/arm64/v8 --rm -t arm64v8/ubuntu uname -m
  ```
the output should be `# aarch64`.

Checkout the References section for more information.

## Usage

To run the build, input the following command:
  ```
  docker buildx build --platform linux/arm/v8 --rm -t jetson-pythorch-build -o wheels .
  ```
## License

BSD 3-Clause License

## References
1. [qemu-user-static](https://github.com/multiarch/qemu-user-static)
2. [Xavier Geerinck Post](https://xaviergeerinck.com/post/2021/11/25/infrastructure-nvidia-ai-nvidia-building-pytorch)
3. [Building pytorch for arm64](https://github.com/soerensen3/buildx-pytorch-jetson)
