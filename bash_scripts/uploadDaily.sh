#!/bin/bash
HOST='ftp_address_of_remote_site'
USER='  login_id  '
PASSWD=' your pwd'

ftp -p -n -v $HOST << EOT
ascii
user $USER $PASSWD
prompt
cd ....remote website directory...../projects/aurora/plots
lcd /home/GeoPhysics/Plots
ls -la
mput *.* 
bye
EOT
