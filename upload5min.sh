#!/bin/bash
HOST='starfishprime.co.uk'
USER='u39331134'
PASSWD='Coffee#2718'

ftp -p -n -v $HOST << EOT
ascii
user $USER $PASSWD
prompt
cd StarFishPrime/projects/EFM/plots
ls -la
put /home/pi/EFM/Plots/Today.svg Today.svg
bye
EOT
