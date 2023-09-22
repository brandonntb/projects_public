#!/bin/bash

echo -e "Installing support packages ...\n"
(sudo apt update && sudo apt install openjdk-8-jdk -y && sudo apt install nginx -y && sudo apt install apt-transport-https -y && sudo apt install curl -y) >/dev/null 2>&1

wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg


echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list

# Install elastic and save the creds
echo -e "Installing Elastic ...\n"
sudo apt update >/dev/null 2>&1 && sudo apt install elasticsearch -y > elasticinfo.txt
echo "username:elastic" > elastic_user.txt
sudo grep "superuser is :" elasticinfo.txt | sudo cut -d ":" -f 2 |tr -d ' ' >> elastic_pass.txt

# Install and start SSH
echo -e "Installing and starting SSH ...\n"
(sudo apt install ssh -y && sudo systemctl start ssh) >/dev/null 2>&1

# --- yml changes
echo "Uncommenting 'network.host', changing to machines IP. Uncommenting 'http.port', change to '9200'."
for IP in $(hostname -I); do sudo sed -i "s/#network.host: 192.168.0.1/network.host: $IP/" /etc/elasticsearch/elasticsearch.yml; done
sudo sed -i 's/#http.port: 9200/http.port: 9200/' /etc/elasticsearch/elasticsearch.yml

echo "Adding transport.port"
sudo echo "transport.port: 9310" >> /etc/elasticsearch/elasticsearch.yml

# ---------- yml changes end

echo "Adding '-Xms512m' and '-Xmx512m' as seperate lines in page"
sudo echo -e "-Xms512m\n-Xmx512m" >> /etc/elasticsearch/jvm.options

## Install Kibana
echo -e "Installing Kibana ...\n"
sudo apt install kibana >/dev/null 2>&1

# Start services
echo -e "Starting Elastic and Kibana Services ...\n"
sudo systemctl start elasticsearch && sudo systemctl enable elasticsearch && sudo systemctl start kibana && sudo systemctl enable kibana

for tkn in $(sudo /usr/share/elasticsearch/bin/elasticsearch-create-enrollment-token -s kibana); do sudo /usr/share/kibana/bin/kibana-setup --enrollment-token $tkn; done

echo -e "Restarting Kibana ...\n"
sudo systemctl restart kibana

# END

echo -e "y\nelastic\nelastic\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-reset-password -i -u elastic

sudo systemctl restart elasticsearch && sudo systemctl restart kibana

echo -e "\nConnect to http://localhost:5601 for the interface, use username and password from command output"
