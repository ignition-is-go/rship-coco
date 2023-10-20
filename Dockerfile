FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
  apt-get install -y \
  git \
  python3-pip \
  python3-dev \
  python3-opencv \
  libglib2.0-0
# Install any python packages you need
COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY src/ .

CMD [ "python3", "main.py" ]