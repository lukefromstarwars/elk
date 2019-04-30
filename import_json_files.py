from utils import *
import requests
from elasticsearch import Elasticsearch
import json


cdcs_imports = {
	"sectors": "https://social.brussels/rest/sectors",
	"zones"  : "https://social.brussels/rest/actionzones"
}

for key in cdcs_imports.keys():
	print(key, cdcs_imports[ key ])
	response = requests.get(cdcs_imports[ key ])
	save_as_json(str(response.content, "utf-8", "ignore"), key)

	print(response.json())

es = Elasticsearch([ { 'host': 'localhost', 'port': '9200' } ])

i = 1
for fn in os.listdir(MyFolders.SOURCES):
	if fn.endswith(".json"):
		file_name = os.path.splitext(fn)[ 0 ]
		print(file_name)

		f = open(MyFolders.SOURCES + fn)
		body_content = f.read()
		# Send the data into es
		es.index(index=f"cdcs-{file_name}", ignore=400, id=i, body=json.loads(body_content))
		i = i + 1
		f.close()

print(cdcs_imports[ "sectors" ])
sectors = response.json()
