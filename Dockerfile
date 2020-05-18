# This location of python in venv-build needs to match the location in the runtime image,
# so we're manually installing the required python environment
FROM ubuntu:18.04 as venv-build

RUN apt-get update && \
    apt-get install -y python3.7 python3.7-distutils python3.7-venv python3.7-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

# build-essential is for jsonnet
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3

COPY pyproject.toml /app/pyproject.toml
COPY poetry.lock /app/poetry.lock
WORKDIR /app

RUN python3 $HOME/.poetry/bin/poetry config virtualenvs.in-project true && \
    python3 $HOME/.poetry/bin/poetry install --no-dev

FROM nvidia/cuda:10.1-runtime-ubuntu18.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y python3.7 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/
COPY --from=venv-build /app/.venv /app/.venv

WORKDIR /app

ENTRYPOINT [".venv/bin/python", "seqvec/seqvec.py"]
