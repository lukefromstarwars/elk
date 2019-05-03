from utils import *
from json_transformations import *
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
		# if fn == "organisations.json":
		if fn.endswith(".json"):
			file_name = os.path.splitext(fn)[ 0 ]
			print(file_name)

			# Load default settings
			# default_settings = dict()
			with open(MyFolders.SOURCES + "settings/default-settings.json", encoding="utf-8") as f:
				# f = open(MyFolders.SOURCES + "settings/default-settings.json", encoding="latin-1")
				default_settings = json.load(f)

			# Create index mappings
			index_name = f"cdcs-{file_name}"
			# DEV++
			es.indices.delete(index=index_name, ignore=[ 400, 404 ])
			print_debug(es.indices.exists(index_name))

			if not es.indices.exists(index_name):
				file_path = MyFolders.SOURCES + f"mappings/{file_name}-mappings.json"

				with open(file_path, encoding="utf-8") as f:
					index_settings = json.load(f)

				index_settings[ "settings" ] = default_settings[ "settings" ]
				es.indices.create(index=index_name, body=index_settings)
				print_check(es.indices.exists(index_name))

			# Index json file
			with open(MyFolders.SOURCES + fn, encoding="utf-8") as f:
				# f = open(MyFolders.SOURCES + fn, encoding="latin-1")
				docs = json.load(f)

				# DEV++
				# docs = docs[ :100 ]
				# print_debug(type(docs))

				# Transform json to custom schema
				transform_json(file_name, docs)

				# Send the data into es
				i = 0
				total = len(docs)
				print_progress_bar(i, total, prefix='Progress:', suffix='Complete', bar_length=100)

				# DEV++
				# added_ids = [ ]

				for doc in docs:
					i = i + 1
					print_progress_bar(i, total, prefix='Progress:', suffix='Complete', bar_length=100)

					my_id = doc[ "id" ]
					del doc[ "id" ]
					es.index(index=f"cdcs-{file_name}", id=my_id, ignore=400, body=doc)

				# DEV++
				# added_ids.append(my_id)

			# print_to_console(added_ids)
			print_check("Done")

