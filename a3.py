
# coding: utf-8

# In[66]:

from collections import Counter, defaultdict

import math

import numpy as np

import os

import pandas as pd

import re

from scipy.sparse import csr_matrix

import urllib.request

import zipfile


# In[67]:

def download_data():

    """ DONE. Download and unzip data.

    """

    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'

    urllib.request.urlretrieve(url, 'ml-latest-small.zip')

    zfile = zipfile.ZipFile('ml-latest-small.zip')

    zfile.extractall()

    zfile.close()


# In[68]:

def tokenize_string(my_string):

    """ DONE. You should use this in your tokenize function.

    """

    return re.findall('[\w\-]+', my_string.lower())


# In[69]:

def tokenize(movies):

    """

    Append a new column to the movies DataFrame with header 'tokens'.

    This will contain a list of strings, one per token, extracted

    from the 'genre' field of each movie. Use the tokenize_string method above.



    Note: you may modify the movies parameter directly; no need to make

    a new copy.

    Params:

      movies...The movies DataFrame

    Returns:

      The movies DataFrame, augmented to include a new column called 'tokens'.



    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])

    >>> movies = tokenize(movies)

    >>> movies['tokens'].tolist()

    [['horror', 'romance'], ['sci-fi']]

    """

    ###TODO
    tokens = []
    random_state= np.random.randn(len(movies))
    genre_field = movies['genres'].tolist()
    movies['tokens'] = pd.Series(random_state, index=movies.index)
    #extracted from the genre field of each movie
    for each in genre_field:
    #Append a new column to the movies DataFrame with header 'tokens' by using the tokenize_string
        tokens.append(tokenize_string(each))
    tokens= pd.Series(tokens)
    movies['tokens'] = tokens.values
    return movies

    pass
#movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
#movies = tokenize(movies)
#movies['tokens'].tolist()
#[['horror', 'romance'], ['sci-fi']]


# In[196]:

def featurize(movies):

    """

    

    Each row will contain a csr_matrix of shape (1, num_features). Each

    entry in this matrix will contain the tf-idf value of the term, as

    defined in class:

    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))

    where:

    i is a term

    d is a document (movie)

    tf(i, d) is the frequency of term i in document d

    max_k tf(k, d) is the maximum frequency of any term in document d

    N is the number of documents (movies)

    df(i) is the number of unique documents containing term i



    Params:

      movies...The movies DataFrame

    Returns:

      A tuple containing:

      - The movies DataFrame, which has been modified to include a column named 'features'.

      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
     """
    
    ###TODO
   
    features=[]
    #Each row will contain a csr_matrix of shape (1, num_features).
    #Each entry in this matrix will contain the tf-idf value of the term, as
    dfs = Counter()
    data = []
    rows = []
    cols = []
    for tokens in movies['tokens']:
        dfs.update(set(tokens))
   
    vocab = {v: i for i, v in enumerate(sorted(dfs))}
    print(vocab)
    N = len(movies)
    n_cols = len(vocab)
    vectors = []
    for tokens in movies.tokens:
        tfs = Counter(tokens)
        maxtf = max(tfs.values())
        rows = [0] * len(tfs)
        cols = [vocab[t] for t in tfs]
        data = [v / maxtf * math.log10(N / dfs[t])
                for t, v in tfs.items()]
        #print('tf:', [v / maxtf for t, v in tfs.items()])
        #print('idf_weight:', [math.log10(N / dfs[t]) for t, v in tfs.items()])
        
        vectors.append(csr_matrix((data, (rows, cols)), shape=(1, n_cols)))
    #Append a new column to the movies DataFrame with header 'features'.     
    movies['features'] = vectors
    return movies, vocab
    pass

#movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Romance']], columns=['movieId', 'genres'])
#movies = tokenize(movies)
#movies, vocab = featurize(movies)
#print(movies)
#print(vocab)
#tf
#{'horror': 0, 'romance': 1}
#movies = pd.DataFrame([[123, ['horror', 'horror', 'romance', 'romance', 'romance']], [456, ['romance']]], columns=['movieId', 'tokens'])
#movies, vocab = featurize(movies)




# In[72]:

def train_test_split(ratings):

    """DONE.

    Returns a random split of the ratings matrix into a training and testing set.

    """

    test = set(range(len(ratings))[::1000])

    train = sorted(set(range(len(ratings))) - test)

    test = sorted(test)

    return ratings.iloc[train], ratings.iloc[test]


# In[150]:

def cosine_sim(a, b):

    """

    Compute the cosine similarity between two 1-d csr_matrices.

    Each matrix represents the tf-idf feature vector of a movie.


    Params:

      a...A csr_matrix with shape (1, number_features)

      b...A csr_matrix with shape (1, number_features)

    Returns:

      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||

      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.

    """

    ###TODO
    #https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    #overlap = [i for i in range(a.shape[0]) if a[i] != 0 and b[i] != 0]
    #a_mean = np.mean(a[np.where(a != 0)])
    #b_mean = np.mean(b[np.where(b != 0)])
    #numerator = ((a[overlap] - a_mean) * (b[overlap] - b_mean)).sum()
    #denominator= (math.sqrt(((a[overlap] - a_mean)**2).sum()) * math.sqrt(((b[overlap] - b_mean)**2).sum()))
    numerator=a.dot(b.T).sum()
    a_norm=(math.sqrt(((a.dot(a.T))).sum()))
    b_norm=(math.sqrt(((b.dot(b.T))).sum()))
    denominator=a_norm*b_norm
    cosine=numerator/denominator
    return cosine
    pass
#a = csr_matrix([1,2,3, 0, 0, 0, 5])
#b = csr_matrix([4,5,6, 0, 0, 0, 0])
#cosine_sim(a, b)
#0.58394549488144942
#0.584276045811 


# In[193]:

def make_predictions(movies, ratings_train, ratings_test):

    """

    Using the ratings in ratings_train, predict the ratings for each

    row in ratings_test.



    To predict the rating of user u for movie i: Compute the weighted average

    rating for every other movie that u has rated.  Restrict this weighted

    average to movies that have a positive cosine similarity with movie

    i. The weight for movie m corresponds to the cosine similarity between m

    and i.

    If there are no other movies with positive cosine similarity to use in the

    prediction, use the mean rating of the target user in ratings_train as the

    prediction.

    Params:

      movies..........The movies DataFrame.

      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.

      ratings_test....The subset of ratings that need to predicted. These are the "future" data.

    Returns:

      A numpy array containing one predicted rating for each element of ratings_test.

    """

    ###TODO
    ratings = []
    for  index,rate in ratings_test.iterrows():
        averagerating=0.0
        similerty= 0.0
        positivecosine= False
        userid= rate['userId']
        movieid= rate['movieId']
        #np.squeeze(x).shape
        Idmovies= movies.loc[movies.movieId == movieid].squeeze()['features']
        #Using the ratings in ratings_train, predict the ratings for each row in ratings_test.
        usermovies= ratings_train.loc[ratings_train.userId == userid]
        #The weight for movie m corresponds to the cosine similarity between movie m and movie i.
        for index,user in usermovies.iterrows():
            cosinesimilerty= cosine_sim(Idmovies, movies.loc[movies.movieId == user['movieId']].squeeze()['features'])
    #To predict the rating of user u for movie i:Restrict this weighted average to movies that have a positive cosine similarity with movie i.
            #if cosinesimilerty >= 0:
            if cosinesimilerty > 0:
                #rate=user['rating']
                averagerating+= cosinesimilerty * user['rating']
                similerty+= cosinesimilerty
                positivecosine= True
                #if positivecosine:
        if positivecosine== False:
            #no other movies with positive cosine similarity to use,use the mean rating of the target user in ratings_train
            meanrating=usermovies['rating'].mean()
            #print(meanrating)
            ratings.append(meanrating)
        else:
            #movies with positive cosine similarity,Compute the weighted average rating for every other movie that u has rated. 
            weightedaverage=averagerating /similerty 
            ratings.append(weightedaverage)   
    return np.array(ratings)
    pass


# In[125]:

def mean_absolute_error(predictions, ratings_test):

    """DONE.

    Return the mean absolute error of the predictions.

    """

    return np.abs(predictions - np.array(ratings_test.rating)).mean()


# In[195]:

def main():

    download_data()

    path = 'ml-latest-small'

    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')

    movies = pd.read_csv(path + os.path.sep + 'movies.csv')

    movies = tokenize(movies)

    movies, vocab = featurize(movies)

    print('vocab:')

    print(sorted(vocab.items())[:10])

    ratings_train, ratings_test = train_test_split(ratings)

    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))

    predictions = make_predictions(movies, ratings_train, ratings_test)

    print('error=%f' % mean_absolute_error(predictions, ratings_test))

    print(predictions[:10])



if __name__ == '__main__':

    main()


# In[ ]:



