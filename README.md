# LCZ-ODC
Project containing the Docker files for the ODC instance for the LCZ project belonging to ASI

## Docker commands
to build: <br>
docker build -t lcz_odc .

to run docker: <br>
docker run -p 8888:8888 -it lcz_odc bash

to open jupyter, run in the container terminal: <br>
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
