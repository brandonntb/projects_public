#!/bin/bash

echo -e "Installing support packages ...\n"
(sudo apt update && sudo apt install openjdk-8-jdk -y && sudo apt install nginx -y && sudo apt install apt-transport-https -y && sudo apt install curl -y) >/dev/null 2>&1

# Adding elastic to the repo list
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list

# Install elastic and save the creds
echo -e "Installing Elastic ...\n"
(sudo apt update && sudo apt install elasticsearch -y) >/dev/null 2>&1

# --- yml changes
echo "Uncommenting 'network.host', changing to machines IP. Uncommenting 'http.port', change to '9200'."
for IP in $(hostname -I); do sudo sed -i "s/#network.host: 192.168.0.1/network.host: $IP/" /etc/elasticsearch/elasticsearch.yml; done
sudo sed -i 's/#http.port: 9200/http.port: 9200/' /etc/elasticsearch/elasticsearch.yml

echo "Adding transport.port"
sudo echo "transport.port: 9310" >> /etc/elasticsearch/elasticsearch.yml

# yml changes needed for certificate communication stuff
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

# Create the certs and move them into the appropriate dirs
echo -e "\n\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-certutil ca
echo -e "\n\n\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-certutil cert --ca elastic-stack-ca.p12
sudo cp /usr/share/elasticsearch/elastic-certificates.p12 /etc/elasticsearch/certs/
sudo chown elasticsearch: /etc/elasticsearch/certs/elastic-certificates.p12

# Add all the SSL security stuff the the ymls -- for certs || adds the path the the config options and sets up th eneeded parameters for certs
sudo sed -i '/^xpack.security.transport.ssl:/,/^\s*$/ {/^\s*keystore\.path:/ s/.*/  keystore.path: certs\/elastic-certificates.p12/; /^\s*truststore\.path:/ s/.*/  truststore.path: certs\/elastic-certificates.p12/; s/^\(\s*verification_mode:\).*$/\1 certificate\n  client_authentication: required/;}' /etc/elasticsearch/elasticsearch.yml

# Set the password for the certs to null/nothing
echo ""
echo -e "y\n\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-keystore add xpack.security.transport.ssl.keystore.secure_password
echo ""

echo ""
echo -e "y\n\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-keystore add xpack.security.transport.ssl.truststore.secure_password
echo ""

# Reset the elastic password
echo -e "y\nelastic\nelastic\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-reset-password -i -u elastic

# Restart all services
sudo systemctl restart elasticsearch && sudo systemctl restart kibana

echo -e "\nConnect to http://localhost:5601 for the interface, creds are elastic:elastic"