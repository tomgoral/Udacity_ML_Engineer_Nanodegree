from sklearn.feature_extraction.text import CountVectorizer

file_path  = 'orig_task'+complete_df['Task'][0]+'.txt'
with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
      file_text = helpers.process_file(file)

vectorizer = CountVectorizer(decode_error='ignore',strip_accents='unicode')
corpus     = open(file_path)
vector     = vectorizer.fit_transform(corpus).toarray()

print('\nANSWER TEXT:')
print(file_text)

#print('\nVOCABULARY:')
#print(vectorizer.vocabulary_)

print('\nSORTED VOCABULARY:')
print({k: v for k, v in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])})

#print('\nFULL VECTOR:')
#print(vector)

print(corpus.readline())


word_list=[]
for k, v in sorted(vectorizer.vocabulary_.items()):
    word_list.append(k)

pd.DataFrame(vector,columns=word_list)


''' RESULTS
ANSWER TEXT:
in object oriented programming inheritance is a way to form new classes instances of which are called objects using classes that have already been defined the inheritance concept was invented in 1967 for simula  the new classes known as derived classes take over or inherit attributes and behavior of the pre existing classes which are referred to as base classes or ancestor classes  it is intended to help reuse existing code with little or no modification  inheritance provides the support for representation by categorization in computer languages categorization is a powerful mechanism number of information processing crucial to human learning by means of generalization what is known about specific entities is applied to a wider group given a belongs relation can be established and cognitive economy less information needs to be stored about each specific entity only its particularities  inheritance is also sometimes called generalization because the is a relationships represent a hierarchy between classes of objects for instance a fruit is a generalization of apple  orange  mango and many others one can consider fruit to be an abstraction of apple orange etc conversely since apples are fruit i e  an apple is a fruit  apples may naturally inherit all the properties common to all fruit such as being a fleshy container for the seed of a plant  an advantage of inheritance is that modules with sufficiently similar interfaces can share a lot of code reducing the complexity of the program inheritance therefore has another view a dual called polymorphism which describes many pieces of code being controlled by shared control code inheritance is typically accomplished either by overriding replacing one or more methods exposed by ancestor or by adding new methods to those exposed by an ancestor  complex inheritance or inheritance used within a design that is not sufficiently mature may lead to the yo yo problem

SORTED VOCABULARY:
{'1967': 0, 'about': 1, 'abstraction': 2, 'accomplished': 3, 'adding': 4, 'advantage': 5, 'all': 6, 'already': 7, 'also': 8, 'an': 9, 'ancestor': 10, 'and': 11, 'another': 12, 'apple': 13, 'apples': 14, 'applied': 15, 'are': 16, 'as': 17, 'attributes': 18, 'base': 19, 'be': 20, 'because': 21, 'been': 22, 'behavior': 23, 'being': 24, 'belongs': 25, 'between': 26, 'by': 27, 'called': 28, 'can': 29, 'categorization': 30, 'classes': 31, 'code': 32, 'cognitive': 33, 'common': 34, 'complex': 35, 'complexity': 36, 'computer': 37, 'concept': 38, 'consider': 39, 'container': 40, 'control': 41, 'controlled': 42, 'conversely': 43, 'crucial': 44, 'defined': 45, 'derived': 46, 'describes': 47, 'design': 48, 'dual': 49, 'each': 50, 'economy': 51, 'either': 52, 'entities': 53, 'entity': 54, 'established': 55, 'etc': 56, 'existing': 57, 'exposed': 58, 'fleshy': 59, 'for': 60, 'form': 61, 'fruit': 62, 'generalization': 63, 'given': 64, 'group': 65, 'has': 66, 'have': 67, 'help': 68, 'hierarchy': 69, 'human': 70, 'in': 71, 'information': 72, 'inherit': 73, 'inheritance': 74, 'instance': 75, 'instances': 76, 'intended': 77, 'interfaces': 78, 'invented': 79, 'is': 80, 'it': 81, 'its': 82, 'known': 83, 'languages': 84, 'lead': 85, 'learning': 86, 'less': 87, 'little': 88, 'lot': 89, 'mango': 90, 'many': 91, 'mature': 92, 'may': 93, 'means': 94, 'mechanism': 95, 'methods': 96, 'modification': 97, 'modules': 98, 'more': 99, 'naturally': 100, 'needs': 101, 'new': 102, 'no': 103, 'not': 104, 'number': 105, 'object': 106, 'objects': 107, 'of': 108, 'one': 109, 'only': 110, 'or': 111, 'orange': 112, 'oriented': 113, 'others': 114, 'over': 115, 'overriding': 116, 'particularities': 117, 'pieces': 118, 'plant': 119, 'polymorphism': 120, 'powerful': 121, 'pre': 122, 'problem': 123, 'processing': 124, 'program': 125, 'programming': 126, 'properties': 127, 'provides': 128, 'reducing': 129, 'referred': 130, 'relation': 131, 'relationships': 132, 'replacing': 133, 'represent': 134, 'representation': 135, 'reuse': 136, 'seed': 137, 'share': 138, 'shared': 139, 'similar': 140, 'simula': 141, 'since': 142, 'sometimes': 143, 'specific': 144, 'stored': 145, 'such': 146, 'sufficiently': 147, 'support': 148, 'take': 149, 'that': 150, 'the': 151, 'therefore': 152, 'those': 153, 'to': 154, 'typically': 155, 'used': 156, 'using': 157, 'view': 158, 'was': 159, 'way': 160, 'what': 161, 'which': 162, 'wider': 163, 'with': 164, 'within': 165, 'yo': 166}

1967	about	abstraction	accomplished	adding	advantage	all	already	also	an	...	using	view	was	way	what	which	wider	with	within	yo
0	1	0	0	0	0	0	0	1	0	0	...	1	0	1	1	0	1	0	0	0	0
1	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	1	0	1	0	0
3	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	0	2	0	0	0	0	0	0	0	0	...	0	0	0	0	1	0	1	0	0	0
5	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
6	0	0	1	0	0	0	2	0	1	2	...	0	0	0	0	0	0	0	0	0	0
7	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
8	0	0	0	0	0	1	0	0	0	1	...	0	1	0	0	0	1	0	1	0	0
9	0	0	0	1	1	0	0	0	0	1	...	0	0	0	0	0	0	0	0	0	0
10	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
11	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	1	2
12 rows × 167 columns'''
