#! /bin/sh

# Run startup initialization.
CONTAINER_STARTED="/setup/container-started"

if [ ! -e $CONTAINER_STARTED ]; then

  echo "Initializing datacube..."
  bash /setup/odc-config.sh

  # echo "Importing data..."
  # python3 /setup/feed_odc.py

  touch $CONTAINER_STARTED
fi

nohup jupyter notebook --allow-root --ip='0.0.0.0' --NotebookApp.token=$NBK_SERVER_PASSWORD

exec "$@"
