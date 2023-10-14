
HOST='ftp_address_of_remote_site'
USER='  login_id  '
PASSWD=' your pwd'

ftp -p -n -v $HOST << EOT
ascii
user $USER $PASSWD
prompt
cd ....remote website directory...../projects/aurora/plots
ls -la
put /home/GeoPhysics/Plots/today_B_Field.svg today_B_Field.svg 
put /home/GeoPhysics/Plots/weekly_B_Field.svg weekly_B_Field.svg
put /home/GeoPhysics/Plots/prev168hrs_B_Field.svg prev168hrs_B_Field.svg 

bye
EOT


