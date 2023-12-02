FROM opendatacube/cube-in-a-box:latest

USER root

WORKDIR /

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip3 install -r requirements.txt

RUN adduser --gecos "" asi

COPY setup ./setup
RUN chmod -R +x /setup
RUN chown -R asi /setup

WORKDIR /home/asi

USER asi

ENTRYPOINT ["/tini", "--", "bash", "/setup/entrypoint.sh"]
