# LCZ-ODC
Project containing the Docker files for the ODC instance for the LCZ project belonging to ASI and Politecnico di Milano

## SETUP:
### To run the docker container you must follow one of the following steps:

#### Build the image from scratch:
1) Clone this repository
2) Move inside the cloned repository folder
3) Run the following command
$ docker build -t lcz_odc .

#### Build the image from scratch:
1) Pull the docker image directly from the Docker repository:
https://hub.docker.com/repository/docker/rodrigocedeno/lcz-odc/general

### Create the external volume to use as a data source:
$ docker volume create volume_asi

### Once the image is in the local system, run a docker container using a volume: <br>
$ docker run -p 8888:8888 --mount type=volume,src=volume_asi,target=/home/asi -it lcz_odc bash


## CONTAINER SETUP

#### Start postgresql service:
$ system postresql start;

#### Activate the conda environment
$ source activate odc_env;

#### Create datacube config folder
$ mkdir /home/asi/datacube_config;

#### Change to datacube config directory
$ cd /home/asi/datacube_config;

#### Start datacube system
$ datacube system init;

#### Return to main folder
$ cd ..

#### Once inside the Docker container, to open jupyter, run in the container terminal: <br>
$ jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root

