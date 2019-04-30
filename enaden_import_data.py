from utils import *


def run_all():
	save_data_to_pickles()
	get_dataset()


def get_parent_roots( df, original_col: str, level=0, child_id='Id', parent_id='ParentId', new_parent_id='ParentId', sfx='_y' ) -> DataFrame:
	#
	df, level = get_parent(df, original_col, level, child_id, parent_id, new_parent_id, sfx)

	#
	df = set_root_columns(df, original_col, level - 1)

	#
	main_col = "{}_{}".format(original_col, 1)
	df.loc[ df[ parent_id ].isnull(), main_col ] = df[ original_col ]

	return df


def set_root_columns( df, original_col, level ):
	level_col = [ "{}_level_{}".format(original_col, i + 1) for i in range(level) ]
	level_col_id = [ "{}_{}".format(col, 'Id') for col in level_col ]
	root_cols = [ "{}_{}".format(original_col, i) for i in range(level, 0, -1) ]

	#
	df_tmp = df[ level_col ]
	for i in range(level + 1):
		col = level_col[ -1 ]
		df_tmp.loc[ df_tmp[ col ].isnull() ] = df_tmp.shift(1, axis=1)

		#
		print_debug(df_tmp, nb_rows=10)

	df = drop_columns(df, level_col + level_col_id)
	df = pd.concat([ df, df_tmp ], axis=1)

	#
	print_debug(df, nb_rows=10)

	df.columns = df.columns[ 0:-level ].tolist() + root_cols

	print_check(df, nb_rows=30)
	return df


def get_parent( df: DataFrame, original_col: str, level=0, child_id='Id', parent_id='ParentId', new_parent_id='ParentId', sfx='_y' ) -> (DataFrame, int):
	# DEV
	# df = read_pickle("source/df_sections")
	# df = df_sections
	# original_col = 'Section'
	# level = 0
	# child_id = 'Id'
	# parent_id = 'ParentId'
	# sfx = '_y'

	#
	cont = df[ parent_id ].notnull().any()

	#
	level = level + 1
	print_debug(level)

	df_tmp = df.merge(
		df,
		how='left',
		left_on=parent_id,
		right_on=child_id,
		suffixes=[ '', sfx ],
		copy=False)

	#
	root_col = "{}_level_{}".format(original_col, level)
	root_col_id = "{}_{}".format(root_col, child_id)

	rename_column_to(df_tmp, original_col + sfx, root_col)
	rename_column_to(df_tmp, new_parent_id + sfx, root_col_id)
	df_tmp = remove_col_with_suffix(df_tmp, sfx)

	#
	print_debug(df_tmp[ df_tmp[ 'Id' ] == 71 ])

	if cont:
		df, level = get_parent(df_tmp, original_col, level=level, parent_id=root_col_id)

	return df, level


def get_tables_from_db():
	sql = "SELECT TABLE_NAME FROM information_schema.tables"
	df = sql2df(sql)
	print_full(df)


def save_data_to_pickles():
	sql_plus = "WHERE RemovedOn IS NULL "

	df_sections = get_section(sql_plus)

	df_organisation_types = get_organisation_type(sql_plus)

	df_organisations = get_organisation(df_organisation_types, sql_plus)

	df_patients = get_patient(df_organisations, sql_plus)

	df_interviews = get_interview(df_patients, df_sections, sql_plus)

	df_question_types = get_question_type()

	get_question(df_question_types, df_sections, sql_plus)

	get_choice(sql_plus)

	get_response(sql_plus)

	print_check("Base datasets created")


def get_dataset():
	df_itvs = read_pickle('source/df_itvs')
	df_responses = read_pickle('source/df_responses')
	df_questions = read_pickle('source/df_questions')

	#
	get_cols_alphabetically(df_questions)
	get_cols_alphabetically(df_itvs)

	#
	start_year = 2014
	end_year = 2019

	df_itvs = df_itvs[ (df_itvs.InterviewYear >= start_year) & (df_itvs.InterviewYear <= end_year) ]

	df = df_itvs.merge(
		df_responses,
		# how='left',
		left_on='Id',
		right_on='InterviewId',
		suffixes=[ '', '_y' ])
	df = remove_col_with_suffix(df, '_y')

	get_cols_alphabetically(df)

	# HARDCODED++ Filtering by active questionnaire and organisation
	df = df[ df.QuestionnaireId.isin([ 33, 47, 97 ]) ]
	df = df[ df.OrganisationId.isin([ 11, 3, 7, 8, 9, 16 ]) ]

	df = df.merge(
		df_questions,
		left_on='QuestionId',
		right_on='Id',
		suffixes=[ '', '_y' ])
	df = remove_col_with_suffix(df, '_y')

	# DEV++ remove verification hack
	# df_responses[df_responses.InterviewId == 18995]
	# df_itvs[df_itvs.Id == 18995]

	# Line up text responses with interviews
	# ------------------------------------------------------------
	df = df[ [
		'ExtCode',
		'Id',
		'InterviewId',
		'OrganisationId',
		'PatientId',
		'Question',
		'QuestionCode',
		'QuestionId',
		'Questionnaire',
		'QuestionType',
		'QuestionTypeId',
		'ResponseIds',
	] ]

	df_simple = df[ ~(df.QuestionTypeId.isin([ 3, 4, 6 ])) ]
	# print_full(get_unique_rows(df_simple[ df_simple.ExtCode.isnull() ], [ 'Questionnaire', 'Question', 'QuestionId' ]).sort_values('QuestionId'))

	cols = [ 'InterviewId', 'ExtCode', 'ResponseIds', 'QuestionTypeId', 'Questionnaire' ]
	df_simple = df_simple[ cols ]
	pivot_cols = [ 'InterviewId', 'ExtCode', 'Questionnaire' ]
	df_simple = df_simple.pivot(index= 'InterviewId', columns='ExtCode', values='ResponseIds')
	df_simple = df_simple.dropna(axis=1, how='all').reset_index()
	get_cols_alphabetically(df_simple)
	save_as_xlsx(df_simple, 'output/Data_Other')

	#  Line up 'choice' questions with interview
	# ------------------------------------------------------------
	df = df[ [ 'Id', 'InterviewId', 'OrganisationId', 'PatientId', 'QuestionId', 'QuestionCode', 'QuestionType', 'QuestionTypeId', 'Questionnaire', 'ResponseIds' ] ]

	# df = df[df.QuestionTypeId.isin([3, 4, 6])]

	# DEV++ sol 2
	cols = [ 'InterviewId', 'QuestionCode', 'ResponseIds', 'QuestionTypeId' ]

	# df2 = df[cols]
	df_single = get_single_choice_responses(df[ cols ])
	df_multi = get_multichoice_responses(df[ cols ])

	# Split multi choice response into single lines
	df_multi = split_multi_choice_to_single(df_multi)
	# df_multi = read_pickle('source/df_responses_multiple_split')
	df2 = pd.concat([ df_single, df_multi ])

	df2.ResponseIds = pd.to_numeric(df2.ResponseIds)

	# Merge ResponseIds with Choices
	df_choices = read_pickle('source/df_choices')
	choice_cols = get_cols_with_prefix(df_choices, 'Choice')
	df_choices = df_choices[ [ 'Id', 'QuestionId' ] + choice_cols ]

	df2 = df2.merge(
		df_choices,
		how='left',
		left_on='ResponseIds',
		right_on='Id',
		suffixes=[ '', '_y' ])
	df2 = remove_col_with_suffix(df2, '_y')

	# Pivot question columns
	df3, df4 = get_itv_responses(df2)
	get_cols_alphabetically(df3)

	while len(df4) > 0:
		print_to_console(df4.shape)
		df5, df4 = get_itv_responses(df4)
		df3 = pd.concat([ df3, df5 ])
		print_to_console(df3.shape)

	df3 = df3.T.drop_duplicates().T

	# Add abstinence period
	# get_unique_rows(df3, ['Période_Abstinence', 'InterviewId'])
	# get_cols_with(df3, 'Abstinence')
	df_abstinence = df3[ df3.Période_Abstinence.notnull() ][ [ 'InterviewId', 'Période_Abstinence' ] ]
	df_abstinence[ 'En_Jours' ] = 1
	df_abstinence.loc[ df_abstinence.Période_Abstinence == 'En mois', 'En_Jours' ] = 30
	df_abstinence.loc[ df_abstinence.Période_Abstinence == 'En années', 'En_Jours' ] = 365
	df_duree = df_responses[ df_responses.QuestionId == 289 ]

	df_abstinence.set_index('InterviewId', inplace=True)
	df_duree.set_index('InterviewId', inplace=True)
	df_abstinence = pd.concat([ df_abstinence, df_duree ], axis=1)

	df_abstinence.dropna(axis=0, how='any', inplace=True)
	df_abstinence[ 'Durée_Abstinence' ] = pd.to_numeric(df_abstinence.ResponseIds) * df_abstinence.En_Jours
	df_abstinence.reset_index(inplace=True)

	df3 = df3.merge(
		df_abstinence[ [ 'InterviewId', 'Durée_Abstinence' ] ],
		how='left',
		left_on='InterviewId',
		right_on='InterviewId',
		suffixes=[ '', '_y' ],
		copy=False)
	df3 = remove_col_with_suffix(df3, '_y')

	# Add interview details
	df3 = df3.merge(
		df_itvs,
		how='left',
		left_on='InterviewId',
		right_on='Id',
		suffixes=[ '', '_y' ],
		copy=False)
	df3 = remove_col_with_suffix(df3, '_y')

	save_as_xlsx(df3, 'output/Enaden_Data')
	print_check('All data imported')

	return df


def get_itv_responses( df ):
	pivot_cols = [ 'InterviewId', 'QuestionCode' ]
	df_first = df[ ~ df.duplicated(subset=pivot_cols) ]
	df_dup = df[ df.duplicated(subset=pivot_cols) ]
	df_pivot_cat = df_first.pivot(index='InterviewId', columns='QuestionCode', values='Choice_1')
	df_pivot = df_first.pivot(index='InterviewId', columns='QuestionCode', values='Choice')
	df_pivot_ord = df_first.pivot(index='InterviewId', columns='QuestionCode', values='ChoiceOrder')
	col_add_suffix(df_pivot_cat, '', '_(Catégorie)')
	col_add_suffix(df_pivot_ord, '', '_(Ord)')
	df_itv_responses = pd.concat([ df_pivot, df_pivot_cat, df_pivot_ord ], axis=1)

	return df_itv_responses.reset_index(), df_dup


def get_multichoice_responses( df ):
	return df[ (df.QuestionTypeId == 6) ]


def get_single_choice_responses( df ):
	return df[ df.QuestionTypeId.isin([ 3, 4 ]) ]


def get_text_responses( df ):
	# return df[(df.QuestionTypeId != 3) & (df.QuestionTypeId != 4) & (df.QuestionTypeId != 6)]
	return df[ ~df.QuestionTypeId.isin([ 3, 4, 6 ]) ]


def merge_responses_questions( df_responses: DataFrame, df_questions: DataFrame ) -> DataFrame:
	# add questions
	df = df_responses.merge(
		df_questions,
		how='left',
		left_on='QuestionId',
		right_on='Id',
		suffixes=[ '', '_y' ],
		copy=False)
	df = remove_col_with_suffix(df, '_y')
	#
	# df[df.QuestionId==215]
	# print_full(	df_questions[df_questions.Id==215])
	return df


def merge_response_choice( df_categorical, df_choices ):
	# df_choices = df_choices[['Id', 'Choice', 'Choice_1']]
	#
	df_categorical = df_categorical.merge(
		df_choices,
		how='left',
		left_on='ResponseIds',
		right_on='Id',
		suffixes=[ '', '_y' ],
		copy=False)

	df_categorical = remove_col_with_suffix(df_categorical, '_y')
	return df_categorical


def split_multi_choice_to_single( df ):
	# split multiple choice responses
	df_t1 = DataFrame(columns=df.columns)
	i = 0
	total = len(df)
	for index, row in df.iterrows():
		i = i + 1
		print(as_percent(i / total))
		ixs = row[ 'ResponseIds' ].split(',')
		df_t2 = DataFrame(data=[ row ] * len(ixs), columns=df.columns)
		df_t2.ResponseIds = ixs
		df_t1 = pd.concat([ df_t1, df_t2 ])

	save_as_pickle(df_t1, 'source/df_responses_multiple_split')
	return df_t1


# region GET BASE DATA

def get_response( sql_plus ):
	sql = "SELECT Id, QuestionId, InterviewId, ResponseIds From AB_Responses "
	df_responses = sql2df(sql + sql_plus)

	df_responses = df_responses[ (df_responses.ResponseIds.notnull()) & (df_responses.ResponseIds != '') ]
	save_as_pickle(df_responses, "source/df_responses")
	return df_responses


def get_choice( sql_plus ):
	# region Choice
	sql = "SELECT " \
		  "Id, " \
		  "Name_InFr AS Choice," \
		  "QuestionId, " \
		  "ParentId, " \
		  "SortOrder AS ChoiceOrder, " \
		  "ExtCode " \
		  "From AB_Choices "
	df_choices = sql2df(sql + sql_plus)
	#
	df_choices = get_parent_roots(df_choices, 'Choice')

	df_choices = df_choices.sort_values([ 'QuestionId', 'ChoiceOrder' ])
	df_choices[ 'Sorting' ] = df_choices.groupby([ 'QuestionId' ]).cumcount() + 1
	df_choices = drop_columns(df_choices, [ 'ChoiceOrder' ])
	rename_column_to(df_choices, 'Sorting', 'ChoiceOrder')

	# print_full(df_choices[df_choices.Choice == 'Opiacés'].sort_values(['Choice_1', 'Choice']))
	#
	save_as_pickle(df_choices, "source/df_choices")
	print_full(df_choices.head(120))

	return df_choices


def get_question( df_question_types, df_sections, sql_plus ):
	# # DEV
	# df_question_types = read_pickle('source/df_question_types')
	# df_sections = read_pickle('source/df_sections')

	kept_cols = df_sections.Section_1.unique()

	#
	sql = "SELECT " \
		  "Id, " \
		  "Name_InFr AS Question, " \
		  "SectionId, " \
		  "ExtCode, " \
		  "IsTitleVisible, " \
		  "SortOrder AS QuestionOrder, " \
		  "QuestionTypeId  " \
		  "From AB_Questions "
	df_questions = sql2df(sql + sql_plus)

	# get questionnaire type
	df_questions = df_questions.merge(
		df_question_types,
		# how='left',
		left_on='QuestionTypeId',
		right_on='Id',
		suffixes=[ '', '_y' ],
		copy=False)
	df_questions = remove_col_with_suffix(df_questions, '_y')

	# get section
	df_questions = df_questions.merge(
		df_sections,
		# how='left',
		left_on='SectionId',
		right_on='Id',
		suffixes=[ '', '_y' ],
		copy=False)
	df_questions = remove_col_with_suffix(df_questions, '_y')

	# rem cols with all null values
	null_cols = get_cols_where_all_null(df_questions)
	df_questions = drop_columns(df_questions, null_cols)

	#
	cols = get_cols_with_prefix(df_questions, 'Section_')
	cols.sort()
	df_questions[ cols ] = df_questions[ cols ].fillna('--')

	df_questions_lookup = get_categorical_responses(df_questions)
	# df_questions_lookup = get_unique_rows(df_questions_lookup, cols + ['SectionId', 'Id', 'Question', 'QuestionType', 'SortOrder', 'ExtCode'])
	# df_questions_lookup = df_questions_lookup.iloc[:-1, :-1]

	save_as_xlsx(df_questions_lookup.sort_values(cols + [ 'Section', 'SectionOrder' ]), 'codes/Questions')

	df_code = pd.read_excel(MyFolders.EXCEL + 'codes/Questions_Code.xlsx')
	df_code = df_code[ [ 'Id', 'ExtCode' ] ]

	print_full(df_code)

	# Check and reset question code
	# df_questions_res = df_questions_lookup.merge(
	# 	df_code,
	# 	how='left',
	# 	left_on='Id',
	# 	right_on='Id',
	# 	suffixes=['', '_y'],
	# 	copy=False)
	# save_as_xlsx(df_questions_res.sort_values(cols + ['Section', 'SortOrder']), 'codes/Questions2')

	df_questions = df_questions.merge(
		df_code,
		how='left',
		left_on='Id',
		right_on='Id',
		suffixes=[ '', '_y' ],
		copy=False)

	get_cols_alphabetically(df_questions)
	rename_column_to(df_questions, 'ExtCode_y', 'QuestionCode')
	#
	save_as_pickle(df_questions, 'source/df_questions')

	return df_questions


def get_categorical_responses( df: DataFrame ) -> DataFrame:
	df = df[ (df.QuestionTypeId == 3) | (df.QuestionTypeId == 4) | (df.QuestionTypeId == 6) ]

	if 'ResponseIds' in df.columns:
		df = df[ (df.ResponseIds.notnull()) & (df.ResponseIds != '') ]

	return df


def get_question_type():
	# region QuestionType
	sql = "SELECT " \
		  "Id, " \
		  "Name_InFr AS QuestionType, " \
		  "SortOrder AS QuestionTypeOrder " \
		  "From AB_QuestionTypes "
	df_question_types = sql2df(sql)
	#
	save_as_pickle(df_question_types, "source/df_question_types")
	print_full(df_question_types.head(10))
	# endregion
	return df_question_types


def get_interview( df_patients, df_sections, sql_plus ):
	sql = "SELECT " \
		  "Id, " \
		  "PatientId, " \
		  "InterviewDate, " \
		  "QuestionnaireId " \
		  "FROM dbo.AB_Interviews "
	df_itvs = sql2df(sql + sql_plus + " AND InterviewDate <='{}'".format(date.today()))

	# Join to patient
	df_itvs = df_itvs.merge(
		df_patients,
		how='left',
		left_on='PatientId',
		right_on='Id',
		suffixes=[ '', '_y' ],
		copy=False)
	df_itvs = remove_col_with_suffix(df_itvs, '_y')

	#
	df_itvs = df_itvs.merge(
		df_sections[ [ 'Id', 'Section' ] ],
		# how='left',
		left_on='QuestionnaireId',
		right_on='Id',
		suffixes=[ '', '_y' ],
		copy=False)
	df_itvs = remove_col_with_suffix(df_itvs, '_y')

	# Get age at interview date
	df_itvs[ 'Age' ] = df_itvs.apply(lambda x: get_age(x.DateOfBirth, x.InterviewDate), axis=1)
	df_itvs[ 'InterviewYear' ] = df_itvs.apply(lambda x: x.InterviewDate.year, axis=1)
	df_itvs = df_itvs[ df_itvs[ 'Age' ] > 0 ]
	add_bin_labels(df_itvs, 'Age')

	# Name section questionnaire
	rename_column_to(df_itvs, 'Section', 'Questionnaire')

	# Rename isMale
	df_itvs[ 'H_F' ] = 'Femme'
	df_itvs.loc[ df_itvs.IsMale, 'H_F' ] = 'Homme'
	df_itvs = drop_columns(df_itvs, [ 'IsMale' ])

	print_full(df_itvs.head(2))

	# Get interview count
	# print_debug(get_cols_with_prefix(df_itvs, 'Organisation'))

	cols = [ 'PatientId', 'Organisation_1', 'Organisation' ]
	df_itvs = get_cumulative_count(df_itvs, cols, 'VISIT_PROGRAM')

	# Get interview count
	cols = [ 'PatientId', 'Organisation_1' ]
	df_itvs = get_cumulative_count(df_itvs, cols, 'VISIT_CENTRE')

	#
	save_as_pickle(df_itvs, "source/df_itvs")
	print_full(df_itvs.sort_values(cols).head(20))

	return df_itvs


def get_patient( df_organisations, sql_plus ):
	sql = "SELECT " \
		  "Id, " \
		  "IsMale, " \
		  "DateOfBirth, " \
		  "OrganisationId, " \
		  "CountryISO " \
		  "FROM dbo.AB_Patients "
	df_patients = sql2df(sql + sql_plus + " AND DateOfBirth <='01-01-{}'".format(date.today().year))

	#
	df_patients = df_patients.merge(df_organisations,
									how='left',
									left_on='OrganisationId',
									right_on='Id',
									suffixes=[ '', '_y' ],
									copy=False)
	df_patients = remove_col_with_suffix(df_patients, '_y')

	#
	save_as_pickle(df_patients, "source/df_patients")
	print_full(df_patients.head())

	return df_patients


def get_organisation( df_organisation_types, sql_plus ):
	sql = "SELECT " \
		  "Id, " \
		  "Name_InFr AS Organisation, " \
		  "OrganisationTypeId, " \
		  "ParentId " \
		  "From AB_Organisations "
	df_organisations = sql2df(sql + sql_plus)

	#
	df_organisations = get_parent_roots(df_organisations, 'Organisation')

	#
	df_organisations = df_organisations.merge(df_organisation_types,
											  # how='left',
											  left_on='OrganisationTypeId',
											  right_on='Id',
											  suffixes=[ '', '_y' ],
											  copy=False)
	df_organisations = remove_col_with_suffix(df_organisations, '_y')

	#
	save_as_pickle(df_organisations, "source/df_organisations")
	print_full(df_organisations.head())
	return df_organisations


def get_organisation_type( sql_plus ):
	sql = "SELECT " \
		  "Id, " \
		  "Name_InFr AS OrganisationType, " \
		  "SortOrder, " \
		  "ParentId , " \
		  "RemovedOn " \
		  "From AB_OrganisationTypes "
	df_organisation_types = sql2df(sql)

	#
	df_organisation_types = get_parent_roots(df_organisation_types, 'OrganisationType')

	#
	save_as_pickle(df_organisation_types, "source/df_organisation_types")
	print_full(df_organisation_types.head())

	return df_organisation_types


def get_section( sql_plus ):
	# sql_plus = ""
	sql = \
		"SELECT " \
		"Id, " \
		"Name_InFr AS Section," \
		"SortOrder AS SectionOrder, " \
		"ParentId " \
		"From AB_Sections "
	df_sections = sql2df(sql + sql_plus)

	# HACK
	questionnaire_list = df_sections[ df_sections.ParentId.isnull() ].Section.tolist()

	#
	df_sections = get_parent_roots(df_sections, 'Section')
	df_sections = df_sections[ df_sections.Section_1.isin(questionnaire_list) ]
	get_unique_values(df_sections, 'Section_1')

	#
	save_as_pickle(df_sections, "source/df_sections")
	print_check(df_sections)

	return df_sections

# endregion
