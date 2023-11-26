#! /bin/sh

sleep 5

echo "Initializing datacube"

datacube -v system init

echo "#############################################"

exec jupyter notebook --allow-root --ip='0.0.0.0' --NotebookApp.token='' --NotebookApp.password=''