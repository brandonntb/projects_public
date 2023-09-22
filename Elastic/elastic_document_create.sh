#!/bin/bash

index_name="cluster-1-logs"

for i in $(seq 1 100)
do
    # Elastic kicks off when you use single quotes
    echo -e "{ \"index\" : { \"_index\" : \"$index_name\", \"_id\" : \"$i\" } }\n{ \"field$i\" : \"value$i\" }"
done > data.json

echo -e "\n" >> data.json

curl -k -X POST 'https://localhost:9200/_bulk' -H 'Content-Type: application/json' -u "elastic:elastic" --data-binary "@data.json" 

