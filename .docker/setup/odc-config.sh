#!/bin/sh

mkdir -p /home/asi/datacube_conf

echo """\
[datacube]
db_hostname: ${ODC_DB_HOSTNAME}
db_database: ${ODC_DB_DATABASE}
db_username: ${ODC_DB_USER}
db_password: ${ODC_DB_PASSWORD}""" > /home/asi/datacube_conf/datacube.conf

datacube -v system init
