#!/bin/bash

echo -e "\n------- MAKE SURE SSH IS INSTALLED ON REMOTE MACHINES IF RUNNING LOCAL -------\n"
echo -e "\nHit ENTER when done ...\n"
read

# Variables
## Fill in to send certificates to these IPs
remote_ips=("<insert IP>")
script_name=$0
local_cluster=$(hostname -I)
# ---------

echo -e "Select an option:\n\t1. Local Cluster\n\t2. Remote Cluster"
read selection

if [ $selection != "1" ] && [ $selection != "2" ]
then
	echo "Error, please supply a valid answer"
	exit 0
fi

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

if [ $selection = "1" ]
then
	# LOCAL CLUSTER
	echo -e "\n\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-certutil ca
	
	echo -e "\n\n\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-certutil cert --ca elastic-stack-ca.p12
	
	sudo cp /usr/share/elasticsearch/elastic-certificates.p12 /etc/elasticsearch/certs/
    
    sudo chown elasticsearch: /etc/elasticsearch/certs/elastic-certificates.p12

	# SCP file to each remote cluster
    for ip in "${remote_ips[@]}"; do scp /usr/share/elasticsearch/elastic-certificates.p12 $script_name elastic@"$ip":/home/elastic/Documents/; done

elif [ $selection = "2" ]
then
	# Remote Cluster
	for file in $(sudo find / -name elastic-certificates.p12 2>/dev/null); do sudo mv $file /etc/elasticsearch/certs/elastic-certificates.p12; done

    sudo chown elasticsearch: /etc/elasticsearch/certs/elastic-certificates.p12

fi	

# END

# Name your cluster and node - will ask for prompt
echo ""
read -p "Name your cluster: " cluster_name
read -p "Name your node: " node_name
echo ""

sudo sed -i "s/#cluster.name: my-application/cluster.name: $cluster_name/" /etc/elasticsearch/elasticsearch.yml
sudo sed -i "s/#node.name: node-1/node.name: $node_name/" /etc/elasticsearch/elasticsearch.yml

# Add all the SSL security stuff the the ymls -- for certs
sudo sed -i '/^xpack.security.transport.ssl:/,/^\s*$/ {/^\s*keystore\.path:/ s/.*/  keystore.path: certs\/elastic-certificates.p12/; /^\s*truststore\.path:/ s/.*/  truststore.path: certs\/elastic-certificates.p12/; s/^\(\s*verification_mode:\).*$/\1 certificate\n  client_authentication: required/;}' /etc/elasticsearch/elasticsearch.yml

echo ""
echo -e "y\n\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-keystore add xpack.security.transport.ssl.keystore.secure_password
echo ""

echo ""
echo -e "y\n\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-keystore add xpack.security.transport.ssl.truststore.secure_password
echo ""

echo -e "y\nelastic\nelastic\n" | sudo /usr/share/elasticsearch/bin/elasticsearch-reset-password -i -u elastic

sudo systemctl restart elasticsearch && sudo systemctl restart kibana

echo -e "\nConnect to http://localhost:5601 for the interface, use username and password from command output"
