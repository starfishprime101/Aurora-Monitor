#!/bin/bash
HOST='yourwebsite.co.uk'
USER='your id'
PASSWD='yourpw'

ftp -p -n -v $HOST << EOT
ascii
user $USER $PASSWD
prompt
cd StarFishPrime/projects/EFM/plots
ls -la
put /home/pi/EFM/Plots/Today.svg Today.svg
bye
EOT
