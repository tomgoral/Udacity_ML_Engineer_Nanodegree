# Run through a dictionary key,value pair

i=0
for k, v in sorted(vectorizer.vocabulary_.items()):
        print(v,k, vector[0][i])
        i+=1


# Containment routines



from sklearn.feature_extraction.text import CountVectorizer

a_text = "This is an answer text"
s_text = "This is a source text"

# set n
n = 1

# instantiate an ngram counter
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# create a dictionary of n-grams by calling `.fit`
vocab2int = counts.fit([a_text, s_text]).vocabulary_

# print dictionary of words:index
print(vocab2int)


# create a vocabulary for 2-grams
text = [a_text, s_text]
print(text)
counts_2grams= [b for l in text for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
print('\n',counts_2grams)


# create array of n-gram counts for the answer and source text
ngrams = counts.fit_transform([a_text, s_text])

# row = the 2 texts and column = indexed vocab terms (as mapped above)
# ex. column 0 = 'an', col 1 = 'answer'.. col 4 = 'text'
ngram_array = ngrams.toarray()
print('\n',ngram_array)

def containment(ngram_array):
    ''' Containment is a measure of text similarity. It is the normalized,
       intersection of ngram word counts in two texts.
       :param ngram_array: an array of ngram counts for an answer and source text.
       :return: a normalized containment value.'''


    # your code here

    # get the min value found in each column of the 2d array
    intersection_list = np.amin(ngram_array, axis=0)

    # sum up number of the intersection counts
    intersection = np.sum(intersection_list)

    # count up the number of n-grams in the answer text
    answer_idx = 0
    answer_cnt = np.sum(ngram_array[answer_idx])

    # normalize and get final containment value
    containment_val = intersection / answer_cnt

    return containment_val


# test out your code
containment_val = containment(ngrams.toarray())

print('Containment: ', containment_val)

# note that for the given texts, and n = 1
# the containment value should be 3/5 or 0.6
assert containment_val==0.6, 'Unexpected containment value for n=1.'
print('Test passed!')


'''  RESULTS
{'this': 5, 'is': 2, 'an': 0, 'answer': 1, 'text': 4, 'source': 3}
['This is an answer text', 'This is a source text']

 [('This', 'is'), ('is', 'an'), ('an', 'answer'), ('answer', 'text'), ('This', 'is'), ('is', 'a'), ('a', 'source'), ('source', 'text')]

 [[1 1 1 0 1 1]
 [0 0 1 1 1 1]]
Containment:  0.6
Test passed'''
