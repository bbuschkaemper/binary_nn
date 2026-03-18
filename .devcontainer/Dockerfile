FROM nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    screen \
    unzip \
    wget \
    nano \
    bash-completion \
    less \
    locales \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

# Console coloring like default devcontainers
RUN echo 'export TERM=xterm-256color' >> /etc/bash.bashrc && \
    echo 'force_color_prompt=yes' >> /etc/bash.bashrc && \
    echo 'alias ls="ls --color=auto"' >> /etc/bash.bashrc && \
    echo 'alias grep="grep --color=auto"' >> /etc/bash.bashrc && \
    echo 'alias diff="diff --color=auto"' >> /etc/bash.bashrc && \
    echo 'if [ -f /etc/bash_completion ]; then . /etc/bash_completion; fi' >> /etc/bash.bashrc

COPY --from=minio/mc:latest /usr/bin/mc /usr/local/bin/mc
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# NodeJS
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash && \
    . "$HOME/.nvm/nvm.sh" && \
    nvm install 24 && \
    nvm alias default 24 && \
    NODE_BIN="$(nvm which default)" && \
    ln -sf "$NODE_BIN" /usr/local/bin/node && \
    ln -sf "$(dirname $NODE_BIN)/npm" /usr/local/bin/npm && \
    ln -sf "$(dirname $NODE_BIN)/npx" /usr/local/bin/npx