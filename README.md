# LCZ-ODC Project - Open Data Cube Docker

This branch contains the [Docker](https://www.docker.com/) files for the ODC (Open Data Cube) instance for the [LCZ-ODC project](https://www.asi.it/2023/05/i4dp_science-primi-traguardi-del-progetto-lcz-odc/), which is a collaboration between ASI and Politecnico di Milano.

<br>
<p align="center">
  <img src="img/odc.png" width="250" style="margin-right: 200px;">
  <img src="img/docker.png" width="300">
</p>
<br>

---

The [Open Data Cube (ODC)](https://www.opendatacube.org/) is an open-source geospatial data analysis framework that provides a scalable and efficient platform for managing and analyzing large volumes of satellite imagery and other geospatial data. It enables users to access and process satellite data in a flexible and interoperable manner.

Having a preconfigured Docker image of Open Data Cube, as provided in this repository, is essential for simplifying the setup and deployment process. The Docker image encapsulates all the necessary dependencies and configurations, allowing users to quickly and easily set up an instance of Open Data Cube for the LCZ-ODC project. This eliminates the need for manual installation and configuration of various software components, saving time and effort in the setup process.

It will be possible to launch Jupyter Notebook, and you can access it through your web browser . It will allows to use Jupyter's interactive environment for your data analysis, visualization, and exploration tasks.

> ⚠️ Note: The Dockerfile provided in this repository is tailored specifically for the ASI LCZ-ODC project. If you have specific requirements or need to customize the configuration, you can modify the Dockerfile accordingly to meet your needs.

## Setup

To run the Docker container, please follow one of the following steps:

#### Option 1: Build the image from scratch:

1. Clone this repository branch
   ```sh
   $ git clone --branch Docker-ODC https://github.com/gisgeolab/LCZ-ODC.git
   ```
2. Move inside the cloned repository folder
   ```sh
   $ cd <folder_path>
   ```
3. Run the following command:

   ```sh
   $ docker build -t lcz_odc .
   ```

#### Option 2: Download Compiled Docker image

Alternatively, you can pull the Docker image directly from the Docker repository using the following link: [LCZ-ODC Docker Image](https://hub.docker.com/repository/docker/rodrigocedeno/lcz-odc/general)

### Create the external volume to use as a data source

Before running the Docker container, create an external volume to use as a data source by executing the following command. Execute the following command to create a volume named volume_asi:

```sh
$ docker volume create volume_asi
```

Once the Docker image is available in the local system, run a Docker container using the created volume:

```sh
$ docker run -p 8888:8888 --mount type=volume,src=volume_asi,target=/home/asi -it lcz_odc bash
```

### Container Setup

Perform the following setup steps inside the Docker container.

1. Start the PostgreSQL service:

```sh
$ system postgresql start;
```

2. Activate the conda environment:

```sh
$ source activate odc_env;
```

3. Create datacube config folder:

```sh
$ mkdir /home/asi/datacube_config;
```

4. Change to datacube config directory:

```sh
$ cd /home/asi/datacube_config;
$ nano datacube.conf
```

5. Paste the following content into the `datacube.conf` file:

```sh
[datacube]
db_database: agdcintegration

#A blank host will use a local socket. Specify a hostname (such as localhost) to use TCP.
db_hostname: localhost

#Credentials are optional: you might have other Postgres authentication configured.
#The default username otherwise is the current user id.
db_username: your_username
db_password: your_password
```

6. Start datacube system

```sh
$ datacube system init;
```

7. Return to main folder

```sh
$ cd ..
```

8. Once inside the Docker container, you can open Jupyter Notebook by running the following command in the container terminal:

```sh
$ jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```

This will start Jupyter Notebook and make it accessible on port 8888. You can access it by opening your web browser and navigating to http://localhost:8888.
