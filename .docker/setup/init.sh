#! /bin/sh

sleep 5

echo "Initializing datacube..."

datacube -v system init

echo "#############################################"

echo "Importing data..."

python3 /setup/feed_odc.py

echo "#############################################"

exec jupyter notebook --allow-root --ip='0.0.0.0' --NotebookApp.token='' --NotebookApp.password=''
