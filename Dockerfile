# Dockerfile created by Rodrigo Cedeno. Contact information: rodrigo.cedeno.j@gmail.com

# Use the Ubuntu 20.04 base image
FROM ubuntu:20.04 as base

# Update the package lists and install timezone data
RUN apt-get update &&\
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Install necessary packages and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bc \
    ca-certificates \
    nano \
    curl \
    git \
    build-essential \
    libgdal-dev \
    libhdf5-serial-dev \
    libnetcdf-dev \
    libgdal-doc \
    hdf5-tools \
    netcdf-bin \
    gdal-bin \
    postgresql \
    postgresql-contrib \
    sudo \
    systemctl \
    unzip \
    vim \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the shell to bash and add conda to the PATH environment variable
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# Add 'asi' user and give sudo privileges
RUN adduser --gecos "" asi
RUN echo "asi:asi" | chpasswd 
RUN usermod -aG sudo asi

# Create a directory for PostgreSQL data
RUN mkdir -p /var/lib/postgresql/data

# Change ownership of the data directory to the postgres user
RUN chown -R postgres:postgres /var/lib/postgresql/data

# Switch to the postgres user
USER postgres

# Initialize the database and create a user and a database
RUN /etc/init.d/postgresql start && \
    psql --command "CREATE USER asi WITH SUPERUSER PASSWORD 'asi';" && \
    createdb -O asi agdcintegration 

# Expose the PostgreSQL port
EXPOSE 5432

# Start the PostgreSQL service
CMD ["/usr/lib/postgresql/12/bin/postgres", "-D", "/var/lib/postgresql/data", "-c", "config_file=/etc/postgresql/12/main/postgresql.conf"]

# Switch back to the root user
USER root

# Create a directory for utilities
RUN mkdir -p /home/root/utils

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Prepend conda-forge before the default channel
RUN conda update conda && \
    conda config --prepend channels conda-forge

# Install Mamba package manager
RUN conda install mamba -n base -c conda-forge

# Add Conda environment and install required packages
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/asi/.bashrc
RUN echo "conda activate base" >> /home/asi/.bashrc

RUN mamba create --name odc_env -y python=3.8 geopandas datacube scipy rasterio scikit-learn scikit-image fiona tensorflow jupyter jupyterlab dask matplotlib plotly pytest-cov hypothesis rioxarray

# Install additional packages using pip
RUN pip install sodapy imagecodecs

# Add a Jupyter kernel for the conda environment
RUN source activate odc_env && python -m ipykernel.kernelspec \
    --name odc_env --display-name odc_env

# Expose ports for Jupyter Notebook, TensorBoard, and PostgreSQL
EXPOSE 8888 6006 5432

# Switch to the 'asi' user and start PostgreSQL and Datacube services
USER asi
CMD system activate odc_env; service postgresql start; datacube system init && tail -F

