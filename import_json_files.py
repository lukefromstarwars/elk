from utils import *
import requests
from elasticsearch import Elasticsearch
import json


es_host = { 'host': 'localhost', 'port': '9200' }


def run_all():
	# import_lookups_as_json()
	create_bulk_index_json()


def import_lookups_as_json():
	cdcs_imports_lkups = {
		# "accreditations": "https://social.brussels/rest/agreements",
		# "accreditors"   : "https://social.brussels/rest/agreementpowers",
		# "categories"    : "https://social.brussels/rest/search/category?activity=*",
		# "languages"     : "https://social.brussels/rest/languageoffer",
		# "legal-status"  : "https://social.brussels/rest/legalstatus",
		# "sectors"       : "https://social.brussels/rest/sectors",
		# "service-types" : "https://social.brussels/rest/offertypes",
		# "zones"         : "https://social.brussels/rest/actionzones",
		"organisations": "https://social.brussels/rest/search/organisation/*"
		# "organisations": "https://sociaal.brussels/rest/search/organisation?agreements=648"
	}

	for key in cdcs_imports_lkups.keys():
		print(key, cdcs_imports_lkups[ key ])
		response = requests.get(cdcs_imports_lkups[ key ])

		# save_as_json(str(response.content.decode("cp1252").encode("utf8")), key)
		# save_as_json(str(response.content, encoding="cp1252", errors="ignore"), key)
		save_as_json(response.content, key)


def import_dependants_as_json():
	cdcs_imports = {
		"categories-by-sector": "https://social.brussels/rest/sector/{id}",
		"paths-by-sector"     : "https://social.brussels/rest/sector/{id}/paths"
	}

	for key in cdcs_imports.keys():
		print(key, cdcs_imports[ key ])
		response = requests.get(cdcs_imports[ key ])
		save_as_json(str(response.content, "cp1252", "ignore"), key)

	print(response.json())


def create_bulk_index_json():
	es = Elasticsearch([ es_host ])

	for fn in os.listdir(MyFolders.SOURCES):
		if fn == "organisations.json":
			# if fn.endswith(".json"):
			file_name = os.path.splitext(fn)[ 0 ]
			print(file_name)

			# Load default settings
			f = open(MyFolders.SOURCES + "settings/default-settings.json", encoding="utf-8")
			# f = open(MyFolders.SOURCES + "settings/default-settings.json", encoding="latin-1")
			default_settings = json.load(f)
			f.close

			# Create index mappings
			if not es.indices.exists(f"cdcs-{file_name}"):
				file_path = MyFolders.SOURCES + f"mappings/{file_name}-mappings.json"
				f = open(file_path)
				# j = f.read()
				index_settings = json.load(f)
				index_settings[ "settings" ] = default_settings[ "settings" ]

				es.indices.create(index=f"cdcs-{file_name}", ignore=400, body=index_settings)
				f.close

			# Index json file
			f = open(MyFolders.SOURCES + fn, encoding="utf-8")
			# f = open(MyFolders.SOURCES + fn, encoding="latin-1")
			docs = json.load(f)
			# docs.append

			# Send the data into es
			i = 0
			total = len(docs)
			print_progress_bar(i, total, prefix='Progress:', suffix='Complete', bar_length=50)

			for doc in docs:
				i = i + 1
				# print(as_percent(i / total))
				print_progress_bar(i, total, prefix='Progress:', suffix='Complete', bar_length=50)

				my_id = doc[ "id" ]
				del doc[ "id" ]
				es.index(index=f"cdcs-{file_name}", id=my_id, ignore=400, body=doc)
			f.close()

# print(cdcs_imports_lkups [ "sectors" ])
# sectors = response.json()
