answer_filename='g0pA_taskb.txt'
n=3

answer_row  = complete_df.loc[complete_df['File'] == answer_filename]
task        = answer_row['Task'].values[0]
answer      = answer_row['Text'].values[0]
source      = complete_df.loc[(complete_df['Datatype'] == 'orig') &
                              (complete_df['Task'] == task)]['Text'].values[0]

counts         = CountVectorizer(analyzer='word', ngram_range=(n,n))         # n-gram counter
vocab2int      = counts.fit([answer, source]).vocabulary_
ngrams = counts.fit_transform([answer, source])                              # n-gram count array
ngram_array = ngrams.toarray()
intersection_list = np.amin(ngram_array, axis=0)                             # min value in each column
intersection = np.sum(intersection_list)
answer_idx = 0
answer_cnt = np.sum(ngram_array[answer_idx])
containment_value = intersection / answer_cnt


print ('SOURCE: \n',source,'\n')
print ('ANSWER: \n',answer,'\n')
print(vocab2int,'\n')
print(ngrams,'\n')
print(intersection_list,'\n')
print(intersection, answer_cnt)
print(containment_value)


if answer_row['Class'].values[0] ==1:
    print('Plagerized !')
else:
    print('not plagerized')


    '''  RESULTS
SOURCE:
 pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents such as the world wide web with the purpose of measuring its relative importance within the set the algorithm may be applied to any collection of entities with reciprocal quotations and references the numerical weight that it assigns to any given element e is also called the pagerank of e and denoted by pr e  the name pagerank is a trademark of google and the pagerank process has been patented u s patent 6 285 999  however the patent is assigned to stanford university and not to google google has exclusive license rights on the patent from stanford university the university received 1 8 million shares in google in exchange for use of the patent the shares were sold in 2005 for 336 million google describes pagerank pagerank relies on the uniquely democratic nature of the web by using its vast link structure as an indicator of an individual page s value in essence google interprets a link from page a to page b as a vote by page a for page b but google looks at more than the sheer volume of votes or links a page receives it also analyzes the page that casts the vote votes cast by pages that are themselves important weigh more heavily and help to make other pages important in other words a pagerank results from a ballot among all the other pages on the world wide web about how important a page is a hyperlink to a page counts as a vote of support the pagerank of a page is defined recursively and depends on the number and pagerank metric of all pages that link to it  incoming links  a page that is linked to by many pages with high pagerank receives a high rank itself if there are no links to a web page there is no support for that page google assigns a numeric weighting from 0 10 for each webpage on the internet this pagerank denotes a site s importance in the eyes of google the pagerank is derived from a theoretical probability value on a logarithmic scale like the richter scale the pagerank of a particular page is roughly based upon the quantity of inbound links as well as the pagerank of the pages providing the links it is known that other factors e g relevance of search words on the page and actual visits to the page reported by the google toolbar also influence the pagerank in order to prevent manipulation spoofing and spamdexing google provides no specific details about how other factors influence pagerank numerous academic papers concerning pagerank have been published since page and brin s original paper in practice the pagerank concept has proven to be vulnerable to manipulation and extensive research has been devoted to identifying falsely inflated pagerank and ways to ignore links from documents with falsely inflated pagerank other link based ranking algorithms for web pages include the hits algorithm invented by jon kleinberg used by teoma and now ask com  the ibm clever project and the trustrank algorithm

ANSWER:
 pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents such as the world wide web with the purpose of measuring its relative importance within the set google assigns a numeric weighting from 0 10 for each webpage on the internet this pagerank denotes a site s importance in the eyes of google  the pagerank is derived from a theoretical probability value on a logarithmic scale like the richter scale the pagerank of a particular page is roughly based upon the quantity of inbound links as well as the pagerank of the pages providing the links the algorithm may be applied to any collection of entities with reciprocal quotations and references the numerical weight that it assigns to any given element e is also called the pagerank of e and denoted by pr e  it is known that other factors e g relevance of search words on the page and actual visits to the page reported by the google toolbar also influence the pagerank  other link based ranking algorithms for web pages include the hits algorithm invented by jon kleinberg used by teoma and now ask com  the ibm clever project and the trustrank algorithm

{'pagerank is link': 295, 'is link analysis': 183, 'link analysis algorithm': 200, 'analysis algorithm used': 21, 'algorithm used by': 11, 'used by the': 457, 'by the google': 71, 'the google internet': 389, 'google internet search': 126, 'internet search engine': 173, 'search engine that': 357, 'engine that assigns': 97, 'that assigns numerical': 380, 'assigns numerical weighting': 51, 'numerical weighting to': 236, 'weighting to each': 480, 'to each element': 435, 'each element of': 93, 'element of hyperlinked': 96, 'of hyperlinked set': 245, 'hyperlinked set of': 146, 'set of documents': 360, 'of documents such': 241, 'documents such as': 91, 'such as the': 374, 'as the world': 44, 'the world wide': 425, 'world wide web': 492, 'wide web with': 484, 'web with the': 475, 'with the purpose': 488, 'the purpose of': 413, 'purpose of measuring': 335, 'of measuring its': 247, 'measuring its relative': 220, 'its relative importance': 192, 'relative importance within': 346, 'importance within the': 152, 'within the set': 489, 'the set google': 416, 'set google assigns': 359, 'google assigns numeric': 121, 'assigns numeric weighting': 50, 'numeric weighting from': 234, 'weighting from 10': 479, 'from 10 for': 113, '10 for each': 0, 'for each webpage': 108, 'each webpage on': 94, 'webpage on the': 476, 'on the internet': 257, 'the internet this': 393, 'internet this pagerank': 174, 'this pagerank denotes': 430, 'pagerank denotes site': 291, 'denotes site importance': 85, 'site importance in': 366, 'importance in the': 151, 'in the eyes': 163, 'the eyes of': 388, 'eyes of google': 103, 'of google the': 244, 'google the pagerank': 130, 'the pagerank is': 405, 'pagerank is derived': 294, 'is derived from': 180, 'derived from theoretical': 87, 'from theoretical probability': 118, 'theoretical probability value': 427, 'probability value on': 328, 'value on logarithmic': 460, 'on logarithmic scale': 256, 'logarithmic scale like': 213, 'scale like the': 355, 'like the richter': 199, 'the richter scale': 415, 'richter scale the': 352, 'scale the pagerank': 356, 'the pagerank of': 406, 'pagerank of particular': 301, 'of particular page': 249, 'particular page is': 318, 'page is roughly': 281, 'is roughly based': 186, 'roughly based upon': 354, 'based upon the': 56, 'upon the quantity': 454, 'the quantity of': 414, 'quantity of inbound': 336, 'of inbound links': 246, 'inbound links as': 164, 'links as well': 206, 'as well as': 47, 'well as the': 481, 'as the pagerank': 43, 'pagerank of the': 302, 'of the pages': 252, 'the pages providing': 409, 'pages providing the': 312, 'providing the links': 333, 'the links the': 395, 'links the algorithm': 211, 'the algorithm may': 387, 'algorithm may be': 10, 'may be applied': 219, 'be applied to': 57, 'applied to any': 39, 'to any collection': 431, 'any collection of': 37, 'collection of entities': 77, 'of entities with': 242, 'entities with reciprocal': 98, 'with reciprocal quotations': 487, 'reciprocal quotations and': 343, 'quotations and references': 337, 'and references the': 32, 'references the numerical': 345, 'the numerical weight': 398, 'numerical weight that': 235, 'weight that it': 478, 'that it assigns': 383, 'it assigns to': 189, 'assigns to any': 52, 'to any given': 432, 'any given element': 38, 'given element is': 119, 'element is also': 95, 'is also called': 177, 'also called the': 16, 'called the pagerank': 73, 'pagerank of and': 299, 'of and denoted': 240, 'and denoted by': 25, 'denoted by pr': 84, 'by pr it': 68, 'pr it is': 324, 'it is known': 191, 'is known that': 182, 'known that other': 197, 'that other factors': 385, 'other factors relevance': 267, 'factors relevance of': 105, 'relevance of search': 347, 'of search words': 250, 'search words on': 358, 'words on the': 490, 'on the page': 259, 'the page and': 400, 'page and actual': 272, 'and actual visits': 23, 'actual visits to': 8, 'visits to the': 462, 'to the page': 446, 'the page reported': 401, 'page reported by': 283, 'reported by the': 349, 'the google toolbar': 390, 'google toolbar also': 131, 'toolbar also influence': 448, 'also influence the': 17, 'influence the pagerank': 172, 'the pagerank other': 407, 'pagerank other link': 303, 'other link based': 268, 'link based ranking': 201, 'based ranking algorithms': 55, 'ranking algorithms for': 339, 'algorithms for web': 12, 'for web pages': 112, 'web pages include': 474, 'pages include the': 310, 'include the hits': 165, 'the hits algorithm': 391, 'hits algorithm invented': 141, 'algorithm invented by': 9, 'invented by jon': 176, 'by jon kleinberg': 64, 'jon kleinberg used': 195, 'kleinberg used by': 196, 'used by teoma': 456, 'by teoma and': 70, 'teoma and now': 377, 'and now ask': 30, 'now ask com': 232, 'ask com the': 48, 'com the ibm': 78, 'the ibm clever': 392, 'ibm clever project': 147, 'clever project and': 76, 'project and the': 330, 'and the trustrank': 35, 'the trustrank algorithm': 420, 'the set the': 417, 'set the algorithm': 361, 'by pr the': 69, 'pr the name': 325, 'the name pagerank': 396, 'name pagerank is': 226, 'pagerank is trademark': 296, 'is trademark of': 187, 'trademark of google': 449, 'of google and': 243, 'google and the': 120, 'and the pagerank': 34, 'the pagerank process': 408, 'pagerank process has': 305, 'process has been': 329, 'has been patented': 133, 'been patented patent': 60, 'patented patent 285': 323, 'patent 285 999': 319, '285 999 however': 2, '999 however the': 4, 'however the patent': 144, 'the patent is': 411, 'patent is assigned': 321, 'is assigned to': 178, 'assigned to stanford': 49, 'to stanford university': 445, 'stanford university and': 371, 'university and not': 451, 'and not to': 29, 'not to google': 231, 'to google google': 436, 'google google has': 123, 'google has exclusive': 124, 'has exclusive license': 134, 'exclusive license rights': 101, 'license rights on': 198, 'rights on the': 353, 'on the patent': 260, 'the patent from': 410, 'patent from stanford': 320, 'from stanford university': 117, 'stanford university the': 372, 'university the university': 453, 'the university received': 422, 'university received million': 452, 'received million shares': 340, 'million shares in': 223, 'shares in google': 362, 'in google in': 159, 'google in exchange': 125, 'in exchange for': 158, 'exchange for use': 100, 'for use of': 111, 'use of the': 455, 'of the patent': 253, 'the patent the': 412, 'patent the shares': 322, 'the shares were': 418, 'shares were sold': 363, 'were sold in': 482, 'sold in 2005': 367, 'in 2005 for': 156, '2005 for 336': 1, 'for 336 million': 107, '336 million google': 3, 'million google describes': 222, 'google describes pagerank': 122, 'describes pagerank pagerank': 88, 'pagerank pagerank relies': 304, 'pagerank relies on': 307, 'relies on the': 348, 'on the uniquely': 261, 'the uniquely democratic': 421, 'uniquely democratic nature': 450, 'democratic nature of': 83, 'nature of the': 227, 'of the web': 254, 'the web by': 424, 'web by using': 472, 'by using its': 72, 'using its vast': 458, 'its vast link': 193, 'vast link structure': 461, 'link structure as': 203, 'structure as an': 373, 'as an indicator': 42, 'an indicator of': 19, 'indicator of an': 167, 'of an individual': 239, 'an individual page': 20, 'individual page value': 168, 'page value in': 288, 'value in essence': 459, 'in essence google': 157, 'essence google interprets': 99, 'google interprets link': 127, 'interprets link from': 175, 'link from page': 202, 'from page to': 116, 'page to page': 287, 'to page as': 442, 'page as vote': 274, 'as vote by': 45, 'vote by page': 464, 'by page for': 66, 'page for page': 277, 'for page but': 109, 'page but google': 275, 'but google looks': 63, 'google looks at': 128, 'looks at more': 214, 'at more than': 53, 'more than the': 225, 'than the sheer': 378, 'the sheer volume': 419, 'sheer volume of': 364, 'volume of votes': 463, 'of votes or': 255, 'votes or links': 468, 'or links page': 263, 'links page receives': 209, 'page receives it': 282, 'receives it also': 342, 'it also analyzes': 188, 'also analyzes the': 15, 'analyzes the page': 22, 'the page that': 402, 'page that casts': 284, 'that casts the': 381, 'casts the vote': 75, 'the vote votes': 423, 'vote votes cast': 466, 'votes cast by': 467, 'cast by pages': 74, 'by pages that': 67, 'pages that are': 313, 'that are themselves': 379, 'are themselves important': 41, 'themselves important weigh': 426, 'important weigh more': 155, 'weigh more heavily': 477, 'more heavily and': 224, 'heavily and help': 137, 'and help to': 28, 'help to make': 138, 'to make other': 440, 'make other pages': 215, 'other pages important': 269, 'pages important in': 309, 'important in other': 153, 'in other words': 161, 'other words pagerank': 271, 'words pagerank results': 491, 'pagerank results from': 308, 'results from ballot': 351, 'from ballot among': 114, 'ballot among all': 54, 'among all the': 18, 'all the other': 14, 'the other pages': 399, 'other pages on': 270, 'pages on the': 311, 'on the world': 262, 'wide web about': 483, 'web about how': 471, 'about how important': 5, 'how important page': 142, 'important page is': 154, 'page is hyperlink': 280, 'is hyperlink to': 181, 'hyperlink to page': 145, 'to page counts': 443, 'page counts as': 276, 'counts as vote': 81, 'as vote of': 46, 'vote of support': 465, 'of support the': 251, 'support the pagerank': 376, 'pagerank of page': 300, 'of page is': 248, 'page is defined': 279, 'is defined recursively': 179, 'defined recursively and': 82, 'recursively and depends': 344, 'and depends on': 26, 'depends on the': 86, 'on the number': 258, 'the number and': 397, 'number and pagerank': 233, 'and pagerank metric': 31, 'pagerank metric of': 297, 'metric of all': 221, 'of all pages': 238, 'all pages that': 13, 'pages that link': 314, 'that link to': 384, 'link to it': 204, 'to it incoming': 439, 'it incoming links': 190, 'incoming links page': 166, 'links page that': 210, 'page that is': 285, 'that is linked': 382, 'is linked to': 184, 'linked to by': 205, 'to by many': 434, 'by many pages': 65, 'many pages with': 218, 'pages with high': 315, 'with high pagerank': 486, 'high pagerank receives': 139, 'pagerank receives high': 306, 'receives high rank': 341, 'high rank itself': 140, 'rank itself if': 338, 'itself if there': 194, 'if there are': 149, 'there are no': 428, 'are no links': 40, 'no links to': 228, 'links to web': 212, 'to web page': 447, 'web page there': 473, 'page there is': 286, 'there is no': 429, 'is no support': 185, 'no support for': 230, 'support for that': 375, 'for that page': 110, 'that page google': 386, 'page google assigns': 278, 'the links it': 394, 'links it is': 208, 'the pagerank in': 404, 'pagerank in order': 293, 'in order to': 160, 'order to prevent': 264, 'to prevent manipulation': 444, 'prevent manipulation spoofing': 327, 'manipulation spoofing and': 217, 'spoofing and spamdexing': 370, 'and spamdexing google': 33, 'spamdexing google provides': 368, 'google provides no': 129, 'provides no specific': 332, 'no specific details': 229, 'specific details about': 369, 'details about how': 89, 'about how other': 6, 'how other factors': 143, 'other factors influence': 266, 'factors influence pagerank': 104, 'influence pagerank numerous': 171, 'pagerank numerous academic': 298, 'numerous academic papers': 237, 'academic papers concerning': 7, 'papers concerning pagerank': 317, 'concerning pagerank have': 80, 'pagerank have been': 292, 'have been published': 136, 'been published since': 61, 'published since page': 334, 'since page and': 365, 'page and brin': 273, 'and brin original': 24, 'brin original paper': 62, 'original paper in': 265, 'paper in practice': 316, 'in practice the': 162, 'practice the pagerank': 326, 'the pagerank concept': 403, 'pagerank concept has': 290, 'concept has proven': 79, 'has proven to': 135, 'proven to be': 331, 'to be vulnerable': 433, 'be vulnerable to': 58, 'vulnerable to manipulation': 469, 'to manipulation and': 441, 'manipulation and extensive': 216, 'and extensive research': 27, 'extensive research has': 102, 'research has been': 350, 'has been devoted': 132, 'been devoted to': 59, 'devoted to identifying': 90, 'to identifying falsely': 437, 'identifying falsely inflated': 148, 'falsely inflated pagerank': 106, 'inflated pagerank and': 169, 'pagerank and ways': 289, 'and ways to': 36, 'ways to ignore': 470, 'to ignore links': 438, 'ignore links from': 150, 'links from documents': 207, 'from documents with': 115, 'documents with falsely': 92, 'with falsely inflated': 485, 'inflated pagerank other': 170}

  (0, 420)	1
  (0, 35)	1
  (0, 330)	1
  (0, 76)	1
  (0, 147)	1
  (0, 392)	1
  (0, 78)	1
  (0, 48)	1
  (0, 232)	1
  (0, 30)	1
  (0, 377)	1
  (0, 70)	1
  (0, 456)	1
  (0, 196)	1
  (0, 195)	1
  (0, 64)	1
  (0, 176)	1
  (0, 9)	1
  (0, 141)	1
  (0, 391)	1
  (0, 165)	1
  (0, 310)	1
  (0, 474)	1
  (0, 112)	1
  (0, 12)	1
  :	:
  (1, 374)	1
  (1, 91)	1
  (1, 241)	1
  (1, 360)	1
  (1, 146)	1
  (1, 245)	1
  (1, 96)	1
  (1, 93)	1
  (1, 435)	1
  (1, 480)	1
  (1, 236)	1
  (1, 51)	1
  (1, 380)	1
  (1, 97)	1
  (1, 357)	1
  (1, 173)	1
  (1, 126)	1
  (1, 389)	1
  (1, 71)	2
  (1, 457)	1
  (1, 11)	1
  (1, 21)	1
  (1, 200)	1
  (1, 183)	1
  (1, 295)	1

[1 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 1 1 0 0 0 1 0 1 0 1 0 0 0 0 1 0 1 0 0 1 0
 1 1 1 0 0 0 1 1 0 0 1 1 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 1 2 0 1
 0 0 1 1 1 0 0 0 0 0 1 1 0 1 0 0 0 1 0 1 1 1 1 1 1 0 0 0 0 1 0 1 0 0 1 0 0
 0 1 1 0 0 0 0 1 1 0 1 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1
 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 1 1 0 1 1 0 0 1 0 1 1 0
 0 1 0 0 1 0 1 1 0 0 1 1 1 0 1 1 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0
 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 1 1 1 0 1 1 1 1 0 1 1 0 1 0 0 0 1 1 0
 1 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 1 1
 0 0 0 1 0 1 1 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0
 1 0 1 1 1 0 1 0 0 0 1 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 0 0 0 0 0 1 0 0 0
 0 0 0 0 1 0 0 1 0 0 1 0 0 1 0 1 0 1 1 1 1 1 1 1 0 0 0 0 1 0 1 1 0 0 0 1 3
 0 0 1 0 0 0 1 1 1 0 0 0 0 1 0 0 0 0 1 0 1 0 0 1 1 1 0 0 1 0 0 0 0 0 0 0 0
 0 0 1 0 1 0 0 0 0 0 1 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 1 1
 1 0 0 1 0 0 1 1 1 1 0 1]

188 195
0.9641025641025641
Plagerized !'''
