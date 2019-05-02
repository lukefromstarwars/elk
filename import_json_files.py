from utils import *
import requests
from elasticsearch import Elasticsearch
import json


cdcs_imports_lkups = {
	"accreditations" : "https://social.brussels/rest/agreements",
	"accreditors"    : "https://social.brussels/rest/agreementpowers",
	"categories"     : "https://social.brussels/rest/search/category?activity=*",
	"languages"      : "https://social.brussels/rest/languageoffer",
	"legal-status"   : "https://social.brussels/rest/legalstatus",
	"sectors"        : "https://social.brussels/rest/sectors",
	"service-types"  : "https://social.brussels/rest/offertypes",
	"zones"          : "https://social.brussels/rest/actionzones",
	"organisations"  : "https://social.brussels/rest/search/organisation/*"
	}

for key in cdcs_imports_lkups.keys() :
	print(key, cdcs_imports_lkups [ key ])
	response = requests.get(cdcs_imports_lkups [ key ])
	save_as_json(str(response.content, "cp1252", "ignore"), key)
	
cdcs_imports = {
	"categories-by-sector" : "https://social.brussels/rest/sector/{id}",
	"paths-by-sector"      : "https://social.brussels/rest/sector/{id}/paths"
	}

for key in cdcs_imports.keys() :
	print(key, cdcs_imports [ key ])
	response = requests.get(cdcs_imports [ key ])
	save_as_json(str(response.content, "cp1252", "ignore"), key)


	print(response.json())

es = Elasticsearch([ {'host' : 'localhost', 'port' : '9200'} ])

i = 1
for fn in os.listdir(MyFolders.SOURCES) :
	if fn.endswith(".json") :
		file_name = os.path.splitext(fn) [ 0 ]
		print(file_name)
		
		f = open(MyFolders.SOURCES + fn)
		body_content = f.read()
		# Send the data into es
		es.index(index=f"cdcs-{file_name}", ignore=400, id=i, body=json.loads(body_content))
		i = i + 1
		f.close()

print(cdcs_imports_lkups [ "sectors" ])
sectors = response.json()
