import re
from datetime import datetime


def transform_organisation( doc: dict ):
	# Transform
	if doc[ "address" ] is not None:
		doc[ "address" ][ "location" ] = [ doc[ "address" ][ "lat" ], doc[ "address" ][ "lon" ] ]

	doc[ "categoryIds" ] = [ item[ "id" ] for item in doc[ "categoriesObject" ] ]
	doc[ "companyNumber" ] = [ re.sub("[^0-9]", "", item) for item in doc[ "pmRp" ] ]
	doc[ "telFr" ] = [ re.sub("[^0-9]", "", item) for item in doc[ "telFr" ] ]
	doc[ "telNl" ] = [ re.sub("[^0-9]", "", item) for item in doc[ "telNl" ] ]
	doc[ "faxFr" ] = [ re.sub("[^0-9]", "", item) for item in doc[ "faxFr" ] ]
	doc[ "faxNl" ] = [ re.sub("[^0-9]", "", item) for item in doc[ "faxNl" ] ]
	doc[ "lastUpdate" ] = datetime.strptime(doc[ "lastUpdate" ], "%d/%m/%y")
	# "08/11/18"

	# Renaming
	doc[ "nameFr" ] = doc[ "nameOfficialFr" ]
	doc[ "nameNl" ] = doc[ "nameOfficialNl" ]
	doc[ "otherNameFr" ] = doc[ "nameAlternativeFr" ]
	doc[ "otherNameNl" ] = doc[ "nameAlternativeNl" ]
	doc[ "oldNameFr" ] = doc[ "nameFormerFr" ]
	doc[ "oldNameNl" ] = doc[ "nameFormerNl" ]

	doc[ "missionFr" ] = doc[ "descriptionFr" ]
	doc[ "missionNl" ] = doc[ "descriptionNl" ]

	doc[ "openingHoursFr" ] = doc[ "permanencyFr" ]
	doc[ "openingHoursNl" ] = doc[ "permanencyNl" ]

	doc[ "commentFr" ] = doc[ "remarkFr" ]
	doc[ "commentNl" ] = doc[ "remarkNl" ]

	doc[ "language" ] = doc[ "langStatus" ]

	doc[ "websiteFr" ] = doc[ "websiteOfficialFr" ]
	doc[ "websiteNl" ] = doc[ "websiteOfficialNl" ]
	doc[ "otherWebsiteFr" ] = doc[ "websiteUnofficialFr" ] + doc[ "websiteInfoFr" ]
	doc[ "otherWebsiteNl" ] = doc[ "websiteUnofficialNl" ] + doc[ "websiteInfoNl" ]
	doc[ "boJournalUrlFr" ] = doc[ "websiteBelgianOfficialJournalFr" ]
	doc[ "boJournalUrlNl" ] = doc[ "websiteBelgianOfficialJournalNl" ]

	# 	Remove
	if doc[ "address" ] is not None:
		del doc[ "address" ][ "id" ]
		del doc[ "address" ][ "lat" ]
		del doc[ "address" ][ "lon" ]
		del doc[ "address" ][ "postalCodeFr" ]
		del doc[ "address" ][ "postalCodeNl" ]
		del doc[ "address" ][ "x" ]
		del doc[ "address" ][ "y" ]

	del doc[ "categories" ]
	del doc[ "categoriesObject" ]
	del doc[ "descriptionFr" ]
	del doc[ "descriptionNl" ]
	del doc[ "langStatus" ]
	del doc[ "legalStatus" ]
	del doc[ "nameAlternativeFr" ]
	del doc[ "nameAlternativeNl" ]
	del doc[ "nameFormerFr" ]
	del doc[ "nameFormerNl" ]
	del doc[ "nameOfficialFr" ]
	del doc[ "nameOfficialNl" ]
	del doc[ "permanencyFr" ]
	del doc[ "permanencyNl" ]
	del doc[ "pmRp" ]
	del doc[ "remarkFr" ]
	del doc[ "remarkNl" ]
	del doc[ "sectorsObjects" ]
	del doc[ "startDate" ]
	del doc[ "type" ]
	del doc[ "websiteBelgianOfficialJournalFr" ]
	del doc[ "websiteBelgianOfficialJournalNl" ]
	del doc[ "websiteOfficialFr" ]
	del doc[ "websiteOfficialNl" ]
	del doc[ "websiteUnofficialFr" ]
	del doc[ "websiteUnofficialNl" ]
	del doc[ "websiteInfoFr" ]
	del doc[ "websiteInfoNl" ]


def transform_accreditation( doc: dict ):
	# Transform
	# Renaming
	# Remove
	del doc[ "agreementPower" ]


def transform_accreditator( doc: dict ):
	# Transform
	# Renaming
	# Remove
	pass


def transform_category( doc: dict ):
	# Transform
	# Renaming
	# Remove
	pass
	del doc[ "organisationsCount" ]


def transform_language( doc: dict ):
	# Transform
	# Renaming
	# Remove
	doc[ "nameFr" ] = str(doc[ "nameFr" ]).strip()
	doc[ "nameNl" ] = str(doc[ "nameNl" ]).strip()


def transform_legal_status( doc: dict ):
	# Transform
	# Renaming
	# Remove
	pass


def transform_service_type( doc: dict ):
	# Transform
	# Renaming
	# Remove
	pass


def transform_zone( doc: dict ):
	# Transform
	# Renaming
	# Remove
	pass


def transform_json( file_name: str, docs: list ):
	if file_name == "organisations":
		docs = [ transform_organisation(doc) for doc in docs ]

	elif file_name == "accreditations":
		docs = [ transform_accreditation(doc) for doc in docs ]

	else:
		pass
