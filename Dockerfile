FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /opt/app
COPY . /opt/app

RUN pip install pip -U
RUN pip install --no-cache-dir .
RUN pip install uvicorn
RUN pip install fastapi
RUN pip install pydantic
RUN pip install spacy_ke
RUN pip install pytextrank
RUN pip install pydantic
RUN pip install protobuf
RUN pip install psutil
RUN pip install "lmppl>=0.1.2"

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download ja_core_news_sm
RUN python -m spacy download de_core_news_sm
RUN python -m spacy download es_core_news_sm
RUN python -m spacy download it_core_news_sm
RUN python -m spacy download ko_core_news_sm
RUN python -m spacy download ru_core_news_sm
RUN python -m spacy download fr_core_news_sm
RUN python -m spacy download zh_core_web_sm

ENV PORT 8080
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
