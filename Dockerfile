ARG BASE_IMAGE=python:3.8-buster
FROM $BASE_IMAGE

COPY poetry.lock pyproject.toml ./
RUN pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -d /home/kedro -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro

# copy the whole project except what is in .dockerignore
WORKDIR /home/kedro
COPY . .
RUN pip3 install -e src
RUN chown -R kedro:${KEDRO_GID} /home/kedro
USER kedro
RUN chmod -R a+w /home/kedro

EXPOSE ${PORT:-5000}

CMD [ "/home/kedro/mlflow_serve.sh" ]
