FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /opt/app
COPY . /opt/app

RUN pip install pip -U
RUN pip install --no-cache-dir .

ARG MODEL='asahi417/question-generation-squad-t5-small'
ARG MAX_LENGTH=512
ARG MAX_LENGTH_OUTPUT=32
ENV MODEL=${MODEL} \
    MAX_LENGTH=${MAX_LENGTH} \
    MAX_LENGTH_OUTPUT=${MAX_LENGTH_OUTPUT}

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]