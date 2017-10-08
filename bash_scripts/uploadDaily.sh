#!/bin/bash
HOST='yourwebsite.co.uk'
USER='your id'
PASSWD='yourpw'

ftp -p -n -v $HOST << EOT
ascii
user $USER $PASSWD
prompt
cd StarFishPrime/projects/EFM/plots
lcd /home/pi/EFM/Plots
ls -la
mput *.* 
bye
EOT
