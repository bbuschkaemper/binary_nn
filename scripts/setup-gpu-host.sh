#!/bin/bash

# This script sets up a GPU host environment by installing necessary drivers,
# CUDA toolkit, and configuring the system for optimal GPU performance.

set -e

# Update package lists
echo "Updating package lists..."
sudo apt-get update -y

# Install ubuntu-drivers-common to manage GPU drivers
echo "Installing ubuntu-drivers-common..."
sudo apt-get install -y ubuntu-drivers-common

# Install the recommended NVIDIA driver
echo "Installing recommended NVIDIA driver..."
sudo ubuntu-drivers autoinstall

# Install CUDA toolkit
echo "Installing CUDA toolkit..."
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   curl \
   gnupg2

# Add NVIDIA package repositories
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

# Install the NVIDIA container toolkit
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.1-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# Configure the NVIDIA container runtime
echo "Configuring NVIDIA container runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to apply changes
echo "Restarting Docker..."
sudo systemctl restart docker


echo "Done! Please reboot the system."