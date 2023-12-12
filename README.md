# LCZ-ODC Project - Open Data Cube Docker

This branch contains the [Docker](https://www.docker.com/) files for the ODC (Open Data Cube) instance of the [LCZ-ODC project](https://www.asi.it/2023/05/i4dp_science-primi-traguardi-del-progetto-lcz-odc/), which is a project in collaboration between ASI and Politecnico di Milano.

<br>
<p align="center">
  <img src="img/odc.png" width="250" style="margin-right: 200px;">
  <img src="img/docker.png" width="300">
</p>
<br>

---

The [Open Data Cube (ODC)](https://www.opendatacube.org/) is an open-source geospatial data analysis framework that provides a scalable and efficient platform for managing and analyzing large volumes of satellite imagery and other geospatial data. It enables users to access and process satellite data in a flexible and interoperable manner.

Having a preconfigured Docker image of Open Data Cube, as provided in this repository, is essential for simplifying the setup and deployment process. The Docker image encapsulates all the necessary dependencies and configurations, allowing users to quickly and easily set up an instance of Open Data Cube for the LCZ-ODC project. This eliminates the need for manual installation and configuration of various software components, saving time and effort in the setup process.

It will be possible to launch [Jupyter Notebook](https://jupyter.org/), and you can access it through your web browser. It will allows to use Jupyter's interactive environment for your data analysis, visualization, and exploration tasks.

>
> ⚠️ **Warning**
> 
> The Dockerfile provided in this repository is tailored specifically for the ASI LCZ-ODC project. If you have specific requirements or need to customize the configuration, you can modify the Dockerfile accordingly to meet your needs.
> 

## Setup

To run the project you need to setup:
- [Docker](https://docs.docker.com/get-docker/),
- [Docker Compose](https://docs.docker.com/compose/install/), and
- [Make](https://www.gnu.org/software/make/) (optional, only used to speed up development and deployment).

The entrypoint is the [Docker Compose file](docker-compose.yml) in the root of the repository. It defines the containers needed to spin up the architecture, namely:
- a [PostgreSQL](https://www.postgresql.org/) database, and
- an ad-hoc image with an Open Data Cube and a Jupiter Notebook installation.

The architecture relies upon a set of **environment variables** sourced from a `.env` file. You can create said file starting from `.env.example` running

```sh
cp .env.example .env
```

and follow the comments in the file to properly set the variables.

Now you need to build the _lcz-odc_ container using the Dockerfile in [.docker](.docker/Dockerfile) directory. The container is based on the official ODC [cube-in-a-box](https://github.com/opendatacube/cube-in-a-box) image with a set of Python libraries is installed on top. Moreover, it mounts a [startup script](./.docker/setup/entrypoint.sh) that:
1. initializes Datacube,
2. imports the data needed by the Notebooks, and
3. starts the Jupiter Notebook server.

>
> 💡 **Tip**
>
> If you need to include other libraries in the container, add them in the [requirements.txt](./.docker/requirements.txt) file and rebuild the image.
> 

To build the image run

```sh
DOCKER_BUILDKIT=1 docker compose build
```
or
```sh
make build
```

With the image in place, you can finally spin up the architecture in the compose file. Note that both the notebooks and the data are mounted as volumes. While the notebooks are versioned, the data are not, given their size. Contact one of the authors to retrieve them and place them in the `.docker/data` directory.

To spin up the architecture run

```sh
docker compose up -d
```
or
```sh
make up
```

The notebooks will be reachable from [localhost](http://localhost).

To turn down the system run

```sh
docker compose down
```
or
```sh
make down
```

---

### Authors

- <b>_Jesus Rodrigo Cedeno Jimenez_</b> (jesusrodrigo.cedeno@polimi.it)
- <b>_Emanuele Capizzi_</b> (emanuele.capizzi@polimi.it)
- <b>_Edoardo Pessina_</b> (edoardopessina.priv@gmail.com)

### Contacts - LCZ-ODC Project

Politecnico di Milano DICA Team:

- <b>_Maria Antonia Brovelli_</b> (maria.brovelli@polimi.it)
- <b>_Barbara Betti_</b> (barbara.betti@polimi.it)
- <b>_Giovanna Venuti_</b> (giovanna.venuti@polimi.it)
- <b>_Daniele Oxoli_</b> (daniele.oxoli@polimi.it)
- <b>_Alberto Vavassori_</b> (alberto.vavassori@polimi.it)

Italian Space Agency (ASI) Team:

- <b>_Deodato Tapete_</b> (deodato.tapete@asi.it)
- <b>_Mario Siciliani de Cumis_</b> (mario.sicilianidecumis@asi.it)
- <b>_Patrizia Sacco_</b> (patrizia.sacco@asi.it)
