#!/bin/bash
HOST='starfishprime.co.uk'
USER='u39331134'
PASSWD='Coffee#2718'

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
