
# coding: utf-8

# In[150]:

from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
from string import punctuation


# In[151]:

def download_data():

    """ Download and unzip data.

    DONE ALREADY.

    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


# In[152]:

def read_data(path):

    """

    Walks all subdirectories of this path and reads all

    the text files and labels.

    DONE ALREADY.



    Params:

      path....path to files

    Returns:

      docs.....list of strings, one per document

      labels...list of ints, 1=positive, 0=negative label.

               Inferred from file path (i.e., if it contains

               'pos', it is 1, else 0)

    """

    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


# In[194]:

def tokenize(doc, keep_internal_punct=False):

    """

    Tokenize a string.

    The string should be converted to lowercase.

    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).

    If keep_internal_punct is True, then also retain punctuation that

    is inside of a word. E.g., in the example below, the token "isn't"

    is maintained when keep_internal_punct=True; otherwise, it is

    split into "isn" and "t" tokens.



    Params:

      doc....a string.

      keep_internal_punct...see above

    Returns:

      a numpy array containing the resulting tokens.



    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)

    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 

          dtype='<U5')

    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)

    array(['hi', 'there', "isn't", 'this', 'fun'], 

          dtype='<U5')

    """

    ###TODO
    doc = doc.lower()
    replaceunderscore = punctuation.replace("_","")
    if(keep_internal_punct==True):
            token =' '.join(filter(None,(divide.strip(replaceunderscore) for divide in doc.split())))
            token = re.sub(r'\s+'," ",token).split()
            return np.array(token , dtype="unicode")
    elif(keep_internal_punct==False):
            token = re.sub(r"[^\w]"," ",doc).split()
            return np.array(token ,dtype="unicode")
    pass 
#tokenize(" Hi there! Isn't this  fun?", keep_internal_punct=False)
#array(['hi', 'there', 'isn', 't', 'this', 'fun'], dtype='<U5')
#tokenize("Hi there! isn't this fun? ", keep_internal_punct=True)
#array(['hi', 'there', "isn't", 'this', 'fun'], dtype='<U5')


# In[195]:

def token_features(tokens, feats):

    """

    Add features for each token. The feature name

    is pre-pended with the string "token=".

    Note that the feats dict is modified in place,

    so there is no return value.



    Params:

      tokens...array of token strings from a document.

      feats....dict from feature name to frequency

    Returns:

      nothing; feats is modified in place.



    >>> feats = defaultdict(lambda: 0)

    >>> token_features(['hi', 'there', 'hi'], feats)

    >>> sorted(feats.items())

    [('token=hi', 2), ('token=there', 1)]

    """

    ###TODO
    token_label = 'token='
    for key in tokens:
        token = token_label + key
        feats[token] += 1
    
    pass
#feats = defaultdict(lambda: 0)
#token_features(['hi', 'there', 'hi'], feats)
#sorted(feats.items())
#[('token=hi', 2), ('token=there', 1)]


# In[249]:

def dealwithk(list,k):
       token = len(list)
       for listk in range(0, token- k + 1):
               yield[list[eachk+listk]for eachk in range(k)]


# In[252]:

def token_pair_features(tokens, feats, k=3):

    """

    Compute features indicating that two words occur near

    each other within a window of size k.



    For example [a, b, c, d] with k=3 will consider the

    windows: [a,b,c], [b,c,d]. In the first window,

    a_b, a_c, and b_c appear; in the second window,

    b_c, c_d, and b_d appear. This example is in the

    doctest below.

    Note that the order of the tokens in the feature name

    matches the order in which they appear in the document.

    (e.g., a__b, not b__a)



    Params:

      tokens....array of token strings from a document.

      feats.....a dict from feature to value

      k.........the window size (3 by default)

    Returns:

      nothing; feats is modified in place.



    >>> feats = defaultdict(lambda: 0)

    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)

    >>> sorted(feats.items())

    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]

    """

    ###TODO
    
    
    """
    for i in range(0, len(tokens) - k + 1):
        for j in range(i, i + k - 1):
            for r in range(j + 1, i + k):
                feats[token_label+tokens[j]+'__'+tokens[r]] += 1
    pass"""
    token_label="token_pair="
    ourpair=dealwithk(tokens,k)
    for pair in ourpair:
        pairtokens = [comb[0]+"__"+comb[1] for comb in combinations(pair,2)]
        for tokens in pairtokens:
            if token_label+tokens in feats:
                feats[token_label+tokens]+=1   
            else:
                feats[token_label+tokens]=1

    pass
#feats = defaultdict(lambda: 0)
#token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
#print(sorted(feats.items()))
#[('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]



# In[186]:

neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])

pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])


# In[216]:

def lexicon_features(tokens, feats):

    """

    Add features indicating how many time a token appears that matches either

    the neg_words or pos_words (defined above). The matching should ignore

    case.



    Params:

      tokens...array of token strings from a document.

      feats....dict from feature name to frequency

    Returns:

      nothing; feats is modified in place.



    In this example, 'LOVE' and 'great' match the pos_words,

    and 'boring' matches the neg_words list.

    >>> feats = defaultdict(lambda: 0)

    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)

    >>> sorted(feats.items())

    [('neg_words', 1), ('pos_words', 2)]

    """

    ###TODO
    poswords=[]
    negwords=[]
    feats['pos_words']=0
    feats['neg_words']=0
    for word in tokens:
        
        temp = word.lower()
        
        for w in pos_words:
        #if w not in poswords:
            temp2 = w.lower()
            if temp == temp2:
                #poswords.append(w)  
                feats['pos_words'] += 1
               
     
        for w in neg_words:
        #if w not in negwords:
            temp2 = w.lower()
            if temp == temp2:
            #negwords.append(w)
                feats['neg_words'] += 1
        
                    
    pass
#feats = defaultdict(lambda: 0)
#lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
#sorted(feats.items())
#[('neg_words', 1), ('pos_words', 2)]



# In[44]:

def featurize(tokens, feature_fns):

    """

    Compute all features for a list of tokens from a single document.

    Params:

      tokens........array of token strings from a document.

      feature_fns...a list of functions, one per feature

    Returns:

      list of (feature, value) tuples, SORTED alphabetically

      by the feature name.



    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])

    >>> feats

    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]

    """
    ###TODO
    feats = defaultdict(lambda: 0)
    for feature in feature_fns:
        feature(tokens, feats)
    data = sorted(feats.items(), key=lambda x:x[0])
    return data
    pass
#feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
#feats
#[('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), 
#('token=i', 1), ('token=movie', 1), ('token=this', 1)]


# In[129]:

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):

    """
    Given the tokens for a set of documents, create a sparse 
    feature matrix, where each row represents a document, and
    each column represents a feature.
    
    Params:

      tokens_list...a list of lists; each sublist is an

                    array of token strings from a document.

      feature_fns...a list of functions, one per feature

      min_freq......Remove features that do not appear in

                    at least min_freq different documents.

    Returns:

      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.

      This is a sparse matrix (zero values are not stored).

      - vocab: a dict from feature name to column index. NOTE

      that the columns are sorted alphabetically (so, the feature

      "token=great" is column 0 and "token=horrible" is column 1

      because "great" < "horrible" alphabetically),



    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]

    >>> tokens_list = [tokenize(d) for d in docs]

    >>> feature_fns = [token_features]

    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)

    >>> type(X)

    <class 'scipy.sparse.csr.csr_matrix'>

    >>> X.toarray()

    array([[1, 0, 1, 1, 1, 1],

           [0, 2, 0, 1, 0, 0]], dtype=int32)

    >>> sorted(vocab.items(), key=lambda x: x[1])

    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]

    """
    ###TODO
    #Given the tokens for a set of documents
    #feats = defaultdict(lambda: 0)
    #for j in range(len(feature_fns)):
        #for i in range(len(tokens_list)):
            #feature_fns[j](tokens_list[i],feats)
    #data1 = sorted(feats.items(), key=lambda x:x[0])
    
    #dictionary from feature name to column index 
    #dictionary=defaultdict(lambda: 0)
    #b=np.array([d[0] for d in data1])
    #for i in range(len(b)):
        #dictionary[b[i]]=i
    if vocab!= None:
        row = []
        col = []
        data = []
        for index,token in enumerate(tokens_list):
            list2 = featurize(token,feature_fns)
            for matrix in list2 :
                if matrix[0] in vocab.keys() :
                    row.append(index)
                    col.append(vocab[matrix[0]])
                    data.append(matrix[1])
        x=csr_matrix((data,(row,col)),shape=(len(tokens_list),len(vocab)))
        return x,vocab 
    else: 
        vocab={}
        count = 0
        FirstList = defaultdict(list)
        SecondList = defaultdict(list)
        for token in range(len(tokens_list)):
            listl = featurize(tokens_list[token],feature_fns)
            dic=dict(listl)
            FirstList[token]=dic
            for d in dic:
                SecondList[d].append(token)
        
        for minifreqs in sorted(SecondList):
            if len(SecondList[minifreqs])>= min_freq:
                vocab[minifreqs] = count
                count += 1   
        row = []
        col = []
        data = []
        for index in sorted(vocab.keys()):
            for list2 in sorted(SecondList[index]):
                if index in FirstList[list2] :
                    row.append(list2)
                    col.append(vocab[index])
                    data.append(FirstList[list2][index])
        x=csr_matrix((data,(row,col)),shape=(len(tokens_list),len(vocab)))
        return x,vocab 
    
    pass
#docs = ["Isn't this movie great?", "Horrible, horrible movie"]
#tokens_list = [tokenize(d) for d in docs]
#feature_fns = [token_features]
#X,vocab = vectorize(tokens_list, feature_fns, min_freq=1)
#print(type(X))
#<class 'scipy.sparse.csr.csr_matrix'>
#print(X.toarray())

#array([[1, 0, 1, 1, 1, 1],
#[0, 2, 0, 1, 0, 0]], dtype=int64)
#sorted(vocab.items(), key=lambda x: x[1])
#[('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]


# In[46]:

def accuracy_score(truth, predicted):

    """ Compute accuracy of predictions.


    Params:

      truth.......array of true labels (0 or 1)

      predicted...array of predicted labels (0 or 1)

    """

    return len(np.where(truth==predicted)[0]) / len(truth)



# In[171]:

def eval_all_combinations(docs, labels, punct_vals, feature_fns, min_freqs):

    """

    Enumerate all possible classifier settings and compute the

    cross validation accuracy for each setting. We will use this

    to determine which setting has the best accuracy.



    For each setting, construct a LogisticRegression classifier

    and compute its cross-validation accuracy for that setting.



    In addition to looping over possible assignments to

    keep_internal_punct and min_freqs, we will enumerate all

    possible combinations of feature functions. So, if

    feature_fns = [token_features, token_pair_features, lexicon_features],

    then we will consider all 7 combinations of features (see Log.txt

    for more examples).



    Params:

      docs..........The list of original training documents.

      labels........The true labels for each training document (0 or 1)

      punct_vals....List of possible assignments to

                    keep_internal_punct (e.g., [True, False])

      feature_fns...List of possible feature functions to use

      min_freqs.....List of possible min_freq values to use

                    (e.g., [2,5,10])



    Returns:

      A list of dicts, one per combination. Each dict has

      four keys:

      'punct': True or False, the setting of keep_internal_punct

      'features': The list of functions used to compute features.

      'min_freq': The setting of the min_freq parameter.

      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.



      This list should be SORTED in descending order of accuracy.
      This function will take a bit longer to run (~20s for me).
    """

    ###TODO
    #list of dicts to store result 
    setting=[]
    result=[]
    #find combinations of features
    #settings=itertools.combinations(feature_fns,(len(feature_fns)*2+1))
    #Enumerate all possible classifier settings 
    #compute the cross validation accuracy for each setting.
    #list(combinations(feature_fns,(len(feature_fns)*2+1)))
    
    combination = []
    for i in range(1,len(feature_fns)+1):
         for j in combinations(feature_fns,i):
            combination.append(list(j))
    for features in combination:
        for punct in punct_vals:
            token=[tokenize(d,punct) for d in docs]
            for min_freq in min_freqs:
                #for i in range(len(min_freqs)):
                X, vocab =vectorize(token, features, min_freq)
                #construct a LogisticRegression classifier
                model = LogisticRegression()
                model.fit(X,labels)
                #compute its cross-validation accuracy for that setting.
                accuracy=cross_validation_accuracy(model, X, labels, k=5)
                result.append({'features':features,'punct':punct,'accuracy':accuracy,'min_freq':min_freq})
    return sorted(result,key=lambda x:(x['accuracy'],x['min_freq']),reverse=True)  
    pass

 


# In[155]:

def cross_validation_accuracy(clf, X, labels, k):

    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).
    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.
    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """

    ###TODO
    #cv = KFold(X.getnnz(), k)
    cv = KFold(X.shape[0], k)
    acc = []
    n=0
    #for traincv, testcv in cv:
        #classifier = LogisticRegression_classifier.train(training_set[traincv])
        #print ('CV_accuracy:', accuracy_score(classifier, training_set[testcv]))
    #for train_index, test_index in kf.split(X):
        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]
    for train_idx, test_idx in cv:
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        accuracy = accuracy_score(labels[test_idx], predicted)
        acc.append(accuracy)
        
    avg = np.mean(acc)
    return avg
    pass


# In[117]:

def plot_sorted_accuracies(results):

    """

    Plot all accuracies from the result of eval_all_combinations

    in ascending order of accuracy.

    Save to "accuracies.png".

    """

    ###TODO
    
    
    lengthsetting=len(results)
    accuracies = [point['accuracy'] for point in results]
    accuracy=sorted(accuracies)
    #min_freq=sorted(setting)
    #fig = pylab.figure()
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    #plt.plot(range(0,lengthsetting),accuracy)
    plt.plot(accuracy)
    plt.show()
    plt.savefig("accuracies.png", dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.show()

    pass


# In[105]:

def mean_accuracy_per_setting(results):

    """

    To determine how important each model setting is to overall accuracy,

    we'll compute the mean accuracy of all combinations with a particular

    setting. For example, compute the mean accuracy of all runs with

    min_freq=2.



    Params:

      results...The output of eval_all_combinations

    Returns:

      A list of (accuracy, setting) tuples, SORTED in

      descending order of accuracy.

    """

    ###TODO

    #features
    #min_freq
    #accuracy
    #punct
    acc=[]
    tup=[]
    #comb = []
    #Sum_punct={}
    #count_punct={}
    #Sum_features={}
    #count_features={}
    #Sum_min_freq={}
    #count_min_freq={}
    #punctt=[]
    #s3=[]
    #for i in range(1,(len(results))):
        #for j in range(0,len(results)-1):
                #if (results[i]['punct'] == results[j]['punct']):
                    #s3[e].append(results)
    features=set()
    punct=set()
    min_freq=set()
    printkey=defaultdict(lambda:(0.0,0))
    for setting in results:
        #features.add(setting['features'])
        #punct.add(setting['punct'])
        #min_freq.add(setting['min_freq'])
        key=" "
        printkey["punct="+str(setting["punct"])]=(printkey["punct="+str(setting["punct"])][0]+setting["accuracy"],printkey["punct="+str(setting["punct"])][1]+1)
        printkey["min_freq="+str(setting["min_freq"])]=(printkey["min_freq="+str(setting["min_freq"])][0]+setting["accuracy"],printkey["min_freq="+str(setting["min_freq"])][1]+1)
        for f in setting["features"]:
            key+=" "+f.__name__
        #for setting in results:
            #if(f==setting['features']):
                 #acc.append(setting['accuracy'])
        #avg = np.mean(acc)
        #printkey["features="+key.strip()]=avg
        printkey["features="+key.strip()]=(printkey["features="+key.strip()][0]+setting["accuracy"],printkey["features="+key.strip()][1]+1)
    for each in printkey.keys():
        acc=printkey[each][0]/printkey[each][1]
        tup.append((acc,each))
        
        
    #for p in punct:
        #for setting in results:
            #if(p==setting['punct']):
        #acc.append(setting['accuracy'])
        #avg = np.mean(acc)
        #printkey["punct="+str(p)]=avg
        
        
    #for f in features:
        #key+=" "+f.__name__
        #for setting in results:
            #if(f==setting['features']):
                 #acc.append(setting['accuracy'])
        #avg = np.mean(acc)
        #printkey["features="+key.strip()]=avg
        #printkey["features="+key.strip()]=(printkey["features="+key.strip()][0]+avg,printkey["features="+key.strip()][1]+1)

        
    #for m in min_freq:
        #for setting in results:
            #if(m==setting['min_freq']):
                 #acc.append(setting['accuracy'])
        #avg = np.mean(acc)
        #printkey["min_freq="+str(m)]=avg
        
                
        #features=setting['features']
        #punct=setting['punct']
        #min_freq=setting['min_freq']
        #accuracy=setting['accuracy']
        #Sum_punct[punct]+=accuracy
        #count_punct[punct]+=1
        #punctt[punct].append((features, punct, min_freq,accuracy))
        #Sum_features[features].append((features, punct, min_freq,accuracy))
        
        #Sum_min_freq[min_freq].append((features, punct, min_freq,accuracy))
    
        #for p in punctt:
            #accuracies[p] = [point['accuracy'] for point in punctt]
            #avg = np.mean(accuracies[p])
            #tup.append((p,avg))
        #total = float(sum(v['accuracy'] for k,v in punctt))
        """  Sum_features[features]+=accuracy
        count_features[features]+=1
        features={features:(Sum_features[features],count_features[features])}
        Sum_min_freq[min_freq]+=accuracy
        count_min_freq[min_freq]+=1
        min_freq={min_freq:(Sum_min_freq[min_freq],count_min_freq[min_freq])}
        #acc = np.mean(accuracy)
    for f in features:
        b=Sum_features[features]
        d=count_features[features]
        acur=b/d
        tup.append((f,acur))
            
    for p in punct:
        b=Sum_punct[punct]
        d=count_punct[punct]
        acur=b/d
        tup.append((p,acur))
      
    for a in accuracy:
        b=Sum_min_freq[min_freq]
        d=count_min_freq[min_freq]
        acur=b/d"""
    #print(punctt)

    res=sorted(tup,key=lambda x:x[0],reverse=True)
    return res
    pass


# In[51]:

def fit_best_classifier(docs, labels, best_result):

    """

    Using the best setting from eval_all_combinations,

    re-vectorize all the training data and fit a

    LogisticRegression classifier to all training data.

    (i.e., no cross-validation done here)



    Params:

      docs..........List of training document strings.

      labels........The true labels for each training document (0 or 1)

      best_result...Element of eval_all_combinations with highest accuracy

    Returns:

      clf.....A LogisticRegression classifier fit to all

            training data.

      vocab...The dict from feature name to column index.

    """
    ###TODO
    #Using the best setting from eval_all_combinations re-vectorize,training data
    tokens_list = [tokenize(d,best_result['punct']) for d in docs]
    csr, vocab = vectorize(tokens_list, best_result['features'], best_result['min_freq'])
    #fit LogisticRegression classifier to all training data.
    clf = LogisticRegression()
    clf.fit(csr, labels)
    return clf, vocab

    pass


# In[214]:

def top_coefs(clf, label, n, vocab):

    """

    Find the n features with the highest coefficients in

    this classifier for this label.

    See the .coef_ attribute of LogisticRegression.



    Params:

      clf.....LogisticRegression classifier

      label...1 or 0; if 1, return the top coefficients for the positive class; else for negative.

      n.......The number of coefficients to return.

      vocab...Dict from feature name to column index.

    Returns:

      List of (feature_name, coefficient) tuples, SORTED in descending order of the coefficient for the given class label.

    """

    ###TODO
    positive=[]
    negative=[]
    top_coef=[]
    coef = clf.coef_[0]
    if label==1:
        top_coef_ind = np.argsort(coef)[::-1][:n]
        for pos_top in top_coef_ind:
            for kay,val in vocab.items():
                if pos_top==val:
                    positive.append((kay,coef[pos_top]))
        pos =sorted(positive,key=lambda x: x[1],reverse=True)
        return pos
    elif label==0:
        neg_coef_ind = np.argsort(coef)[::1][:n]
        for neg_top in neg_coef_ind:
            for kay,val in vocab.items():
                if neg_top==val:
                    negative.append((kay,abs(coef[neg_top ])))
        neg =sorted(negative,key=lambda x: x[1],reverse=True)
        return neg
    pass


# In[202]:

def parse_test_data(best_result, vocab):

    """

    Using the vocabulary fit to the training data, read

    and vectorize the testing data. Note that vocab should

    be passed to the vectorize function to ensure the feature

    mapping is consistent from training to testing.



    Note: use read_data function defined above to read the

    test data.



    Params:

      best_result...Element of eval_all_combinations

                    with highest accuracy

      vocab.........dict from feature name to column index,

                    built from the training data.

    Returns:

      test_docs.....List of strings, one per testing document,

                    containing the raw.

      test_labels...List of ints, one per testing document,

                    1 for positive, 0 for negative.

      X_test........A csr_matrix representing the features

                    in the test data. Each row is a document,

                    each column is a feature.

    """

    ###TODO
    
    #Using the vocabulary fit to the training data, read and vectorize the testing data. 
    ##vocab should be passed to the vectorize function to ensure the feature
    #use read_data function defined above to read the test data.
    #pos_test = read_data(path,'pos','test')
    test_docs, test_labels= read_data(os.path.join('data','test'))
    #test_docs=np.concatenate([ pos_test,neg_test])
    tokens_list = [tokenize(d,best_result['punct']) for d in test_docs]
    X_test,vocab=vectorize(tokens_list, best_result['features'], best_result['min_freq'], vocab)
    #construct labels
    #label_1=np.ones((len(pos_test),), dtype=np.int)
    #label_0=np.zeros((len(neg_test),), dtype=np.int)
    #test_labels=np.concatenate([label_1, label_0])
    return test_docs, test_labels, X_test
    pass


# In[218]:

def print_top_misclassified(test_docs, test_labels, X_test, clf, n):

    """

    Print the n testing documents that are misclassified by the

    largest margin. By using the .predict_proba function of

    LogisticRegression <https://goo.gl/4WXbYA>, we can get the

    predicted probabilities of each class for each instance.

    We will first identify all incorrectly classified documents,

    then sort them in descending order of the predicted probability

    for the incorrect class.

    E.g., if document i is misclassified as positive, we will

    consider the probability of the positive class when sorting.



    Params:

      test_docs.....List of strings, one per test document

      test_labels...Array of true testing labels

      X_test........csr_matrix for test data

      clf...........LogisticRegression classifier fit on all training data.
      
      n.............The number of documents to print.

    Returns:

      Nothing; see Log.txt for example printed output.

    """

    ###TODO
    predict = clf.predict(X_test)
    predict_prob = clf.predict_proba(X_test)
    res=[]
    result={}
    for i in range(len(predict)):
        if predict[i] != test_labels[i]:
            res.append((test_labels[i],predict[i],predict_prob[i][predict[i]],test_docs[i]))
          
           
    data=sorted(res,key=lambda x:x[2] ,reverse=True)
    for d in data[:n]:
        print('truth='+ str(d[0])+"  " + 'predicted=' + str(d[1]) +" "+ 'probas=' + str(d[2]))
        print(d[3])
    pass






           


# In[254]:

def main():

    """

    Put it all together.

    ALREADY DONE.

    """

    feature_fns = [token_features, token_pair_features, lexicon_features]

    # Download and read data.

    download_data()

    docs, labels = read_data(os.path.join('data', 'train'))


    # Evaluate accuracy of many combinations

    # of tokenization/featurization.

    results = eval_all_combinations(docs, labels,

                                    [True, False],

                                    feature_fns,

                                    [2,5,10])
    print(results)
    # Print information about these results.

    best_result = results[0]

    worst_result = results[-1]

    print('best cross-validation result:\n%s' % str(best_result))

    print('worst cross-validation result:\n%s' % str(worst_result))

    plot_sorted_accuracies(results)

    print('\nMean Accuracies per Setting:')

    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))



    # Fit best classifier.

    clf, vocab = fit_best_classifier(docs, labels, results[0])



    # Print top coefficients per class.

    print('\nTOP COEFFICIENTS PER CLASS:')

    print('negative words:')

    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))

    print('\npositive words:')

    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))



    # Parse test data

    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)



    # Evaluate on test set.

    predictions = clf.predict(X_test)

    print('testing accuracy=%f' %

          accuracy_score(test_labels, predictions))



    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')

    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)





if __name__ == '__main__':

    main()

