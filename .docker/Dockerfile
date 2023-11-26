FROM opendatacube/cube-in-a-box:latest

USER root

WORKDIR /

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip3 install -r requirements.txt

RUN adduser --gecos "" asi

COPY init.sh .
RUN chmod +x init.sh
RUN chown -R asi init.sh

WORKDIR /home/asi

USER asi

ENTRYPOINT ["/tini", "--", "/init.sh"]
