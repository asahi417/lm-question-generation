FROM python:3.7

#RUN apt-get update && \
#  apt-get install -y --no-install-recommends \
#  libpq-dev \
#  git \
#  gcc \
#  python-dev \
#  build-essential \
#  openssh-server \
#  libsndfile1 \
#  google-perftools && \
#  apt-get clean && \
#  rm -rf /var/lib/apt/lists/*
RUN pip3 install pip -U
RUN pip3 install --no-cache-dir .

ARG MODL
ARG MAX_LENGTH
ARG MAX_LENGTH_OUT

EXPOSE 80

COPY ./app /app

CMD ["MODEL=${MODEL}", "MAX_LENGTH=${MAX_LENGTH}", "MAX_LENGTH_OUT=${MAX_LENGTH_OUT}",
     "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]