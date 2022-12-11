#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
from tqdm import tqdm 


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[4]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# # Read Prediction

# ## Setting up Data

# In[64]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[65]:


allRatings[0]


# In[66]:


len(allRatings)


# In[88]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]


# In[68]:


ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set)
ratingDict = {}

for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    usersPerItem[b].add(u)
    itemsPerUser[u].add(b)
    ratingDict[(u,b)] = r


# In[69]:


trainRatings = [r[2] for r in ratingsTrain]
globalAverage = sum(trainRatings) * 1.0 / len(trainRatings)
globalAverage


# In[70]:


itemAverages = {}
userAverages = {}

for i in ratingsPerItem:
    rs = [r[1] for r in ratingsPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)

for u in ratingsPerUser: 
    rs = [r[1] for r in ratingsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)


# In[71]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > (3 * totalRead)/4: break


# In[72]:


# add neg entry for every user to validation set
userBooks = defaultdict(list)
userNoBooks = defaultdict(list)

for u,b,r in allRatings:  
    userBooks[u].append(b)

for u in userBooks: 
    for b in bookCount: 
        if b not in userBooks[u]: 
            userNoBooks[u].append(b)


# In[73]:


# add neg entry for every user to validation set
ratingsToAdd = []
for u,b,_ in ratingsValid: 
    
    rand = random.randrange(len(userNoBooks[u]))
    book = userNoBooks[u][rand]
    userNoBooks[u].remove(book)
    
    ratingsToAdd.append((u,book,-1))
    userBooks[u].append(book)


# In[74]:


ratingsValid.extend(ratingsToAdd)


# ## Trying feature vector + log regression

# In[75]:


ratingsTrain[0]


# In[76]:


def Jaccard(s1, s2):
    numerator = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    
    return numerator / denom


# In[77]:


def feature(datum): 
    feat = [1]
    
    user = datum[0]
    item = datum[1]
    r = datum[2]
    
    # popularity of item 
    count = 0
    popT = 0.0
    for ic, i in mostPopular:
        count += ic
        if i == item: 
            popT = count / totalRead
            break 
    
    # MAX jaccard sim for item 
    maxIjaccard = 0.0
    if user in ratingsPerUser: 
        jaccardList = []
        for pair in ratingsPerUser[user]:
            b2 = pair[0]
            if item == b2: 
                #alreadyRead = True
                #break
                continue

            # get users who read the book
            readB = usersPerItem[item]
            readB2 = usersPerItem[b2]

            # compute similarity between the two books 
            sim = Jaccard(readB,readB2)
            jaccardList.append(sim)

        jaccardList.sort(reverse=True)
        if len(jaccardList) >= 1: 
            maxIjaccard = jaccardList[0]
        
    
    # Max jaccard sim for users 
    maxUjaccard = 0.0
    if item in ratingsPerItem:
        jaccardList = []
        for pair in ratingsPerItem[item]:
            u2 = pair[0]
            if user == u2: 
                #alreadyRead = True
                #break
                continue

            # get books for each user 
            readU = itemsPerUser[user]
            readU2 = itemsPerUser[u2]

            # compute similarity between the two users
            sim = Jaccard(readU,readU2)
            jaccardList.append(sim)
      
        jaccardList.sort(reverse=True)
        if len(jaccardList) >= 1: 
            maxUjaccard = jaccardList[0]
    
    #if popT != 0: 
        #popT = 1 / popT 
        
    return feat + [popT, maxIjaccard, maxUjaccard]
    


# In[78]:


feature(ratingsTrain[0])


# In[79]:


# train on validation set  


# In[80]:


Xvalid = [feature(d) for d in tqdm(ratingsValid)]
Yvalid = [not(d[2] == -1) for d in ratingsValid]


# In[81]:


mod = linear_model.LogisticRegression(C=14.0, class_weight='balanced', verbose=True)
mod.fit(Xvalid,Yvalid)
predictions = mod.predict(Xvalid)


# In[82]:


predictions[:10]


# In[83]:


TP = sum([a == b and b == True for a,b in zip(predictions,Yvalid)])
TN = sum([a == b and b == False for a,b in zip(predictions,Yvalid)])
FP = sum([a != b and b == False for a,b in zip(predictions,Yvalid)])
FN = sum([a != b and b == True for a,b in zip(predictions,Yvalid)])
acc = [a == b for a,b in zip(predictions, Yvalid)]
acc = sum(acc) / len(acc)

BTP = TP / (TP + FN)
BTN = TN / (TN + FP)
BER = 1 - (BTP + BTN) / 2

print(TP,TN,FP,FN,BER, acc)


# In[84]:


# looking for best c 
vals = np.arange(1.0, 50.0, 1.0)
maxAcc = 0.0 
t = 1.0 
for c in vals:
    mod = linear_model.LogisticRegression(C=c, class_weight='balanced')
    mod.fit(Xvalid,Yvalid)
    predictions = mod.predict(Xvalid)
    
    acc = [a == b for a,b in zip(predictions, Yvalid)]
    acc = sum(acc) / len(acc)
    
    if acc > maxAcc: 
        maxAcc = acc
        t = c


# In[85]:


print(maxAcc, t)


# In[86]:


mod = linear_model.LogisticRegression(C=t, class_weight='balanced', verbose=True)
mod.fit(Xvalid,Yvalid)


# In[87]:


predictions = open("predictions_Read.csv", 'w')

for l in open("pairs_Read.csv"):
    #print(l)
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)
    
    feat = feature((u,b,-1000)) # rating doesnt matter 
    result = mod.predict([feat])
    
    pred = 0
    if result == True: 
        pred = 1

    line = u + "," + b + "," + str(pred) + "\n"
    predictions.write(line)

predictions.close()


# ###### IDEAS: 
# 
# 1. use both, improve similarity 
#     - rather than looking at the maximum similarity of a book, try avg similarity or avg similarity among 10% of total similarity entries or min 
# 2. try different similarity methods
# 3. try similarity based on Users, not items 
# 3. make feature vectors including 
#     - popularity 
#     - similarity 
#     - length of word 

# ## Conclusions: 
# 
# 1. Basic pop + sim using max(Jaccard) is very effective
# 2. not much difference between max(Jaccard) and avg(Jaccard) 
# 3. Jaccard sim over items > over users 
# 4. Feature vector has given best accuracy -> Winning Model

# # Category Prediction 

# In[42]:


import time
import nltk
from nltk.corpus import stopwords


# In[43]:


nltk.download('stopwords')


# In[44]:


data = []
reviewsPerUser = defaultdict(list)

for d in readGz("train_Category.json.gz"):
    u = d['user_id']
    r = d['review_id']
    
    reviewsPerUser[u].append(d)

    data.append(d)


# In[45]:


data[0]


# In[46]:


reviewTrain = data[:90000]
reviewValid = data[90000:]


# In[47]:


reviewTrain[50000]


# In[48]:


punctuation = set(string.punctuation)
punctuation.remove('!')
punctuation.remove('?')
punctuation


# In[49]:


# NEW CREATIVE VERSION 
stop = stopwords.words("english")
wordCount = defaultdict(int)

wordSetPerReview = defaultdict(set)
for d in reviewTrain: 
    u = d['user_id']
    r_id = d['review_id']
    
    for w in d['review_text'].split(): 
        r = ["".join([c for c in w.lower() if not c in punctuation])]
        
        # addressing ! and ? 
        if '!' in w: 
            r = r[0].split('!')
            r.append('!')
            
        if '?' in w: 
            r = r[0].split('?')
            r.append('?')
            
       
        for word in r:    
            if word in stop: 
                continue 
            wordSetPerReview[r_id].add(word)
            wordCount[word] += 1


# In[50]:


wordCount


# In[51]:


wordCount.pop('', None)
mostPopular = [(wordCount[w], w) for w in wordCount]
mostPopular.sort()
mostPopular.reverse()


# In[52]:


len(wordCount)


# In[53]:


mostPopular[:10]


# In[54]:


def Jaccard(s1, s2):
    numerator = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    
    return numerator / denom


# In[59]:


# creative feature 
def feature8C(datum): 
    f = [0]*len(wordSet)
    datumWordSet = set()
    
    for w in datum['review_text'].split(): 
        r = ["".join([c for c in w.lower() if not c in punctuation])]
        
        # addressing ! and ? 
        if '!' in w: 
            r = r[0].split('!')
            r.append('!')
        
        if '?' in w: 
            r = r[0].split('?')
            r.append('?')
            
        #if w in stop: 
            #continue 
        for word in r:  
            if word in wordSet: 
                index = wordId[word]
                f[index] += 1
                
                datumWordSet.add(word)
    
    # somehow leverage user history 
    avgRating = [0.0]*5
    u = datum['user_id']
    simPerGenre = [-1.0]*5 
    if u in reviewsPerUser: 
        # find avg rating user has given for each genre and check similarities of words in reviews, take max for each genre 
        reviews = reviewsPerUser[u]
        numReviews = [0]*5
        
        
        for rev in reviews: 
            if rev['review_id'] == datum['review_id']: 
                continue
            rating = rev['rating']
            ind = rev['genreID']
            
            # avg stuff 
            avgRating[ind] += rating 
            numReviews[ind] += 1
            
            # similarity stuff
            revWordSet = wordSetPerReview[rev['review_id']]
            
            if len(revWordSet) == 0 and len(datumWordSet) == 0: 
                sim = 0.0 
                #print(rev['review_id'], datum['review_id'])
            else: 
                sim = Jaccard(revWordSet, datumWordSet)
        
            simPerGenre[ind] = max(sim, simPerGenre[ind])

        # more avg stuff 
        for val in range(0,5): 
            if numReviews[val] != 0: 
                avgRating[val] = avgRating[val] / numReviews[val]
    
    
    return f + avgRating + simPerGenre + [1]


# In[60]:


# run regression on dict size 10000 
for val in [10000]: 
    start = time.perf_counter()
    
    words = [x[1] for x in mostPopular[:val]]
    wordId = dict(zip(words, range(len(words))))
    wordSet = set(words)

    X = [feature8C(x) for x in data]
    y = [x['genreID'] for x in data]

    Xtrain = X[:9*len(X)//10]
    ytrain = y[:9*len(y)//10]
    Xvalid = X[9*len(X)//10:]
    yvalid = y[9*len(y)//10:]
    
    mod = linear_model.LogisticRegression(C=1, verbose=True)
    mod.fit(Xtrain,ytrain)

    pred = mod.predict(Xvalid)
    correct = [(p == l) for (p,l) in zip(pred, yvalid)]
    acc7 = sum(correct) / len(correct)
    
    final = time.perf_counter() - start 
    
    acc7


# In[61]:


print(acc7, final / 60)

# dict size = 5000 
# no punctuation but keeping ! / ? = 0.7093 
# no punctuation, keeping ! / ?, no stopwords = 0.7189 
# no punctuation, = 0.7128 
# no punctuation, no stopwords = 0.7169

# n-grams 
    # no stopwords for 1 gram, up to 2 grams - 0.6852
    # up to 3 grams - 0.68 
    # no stopwords for 1 gram, up to 5 - 0.6779 
    
    # no stopwards for all grams, up to 2 grams - 0.7066
    # "                         ", up to 2, keeping ! / ? = 0.7049 

# dict size = 10000
    #-  n gram "                         ", up to 2, keeping ! / ? = 0.7163 (8 minutes to run)
    #-  n gram "                         ", up to 3, keeping ! / ? = 0.7127 (9 minutes to run)
    #-  no punctuation, keeping ! / ?, no stopwords = 0.7344 ()
    
# extra features 
    # no punctuation sol + avg rating of genre = 0.7345, 9.5 minutes 
    # no punc, avg rating, similarity = 0.7556 


# In[ ]:


## Get predictions


# In[62]:


test = []
reviewPerId = defaultdict(set)
for d in readGz("test_Category.json.gz"):
    test.append(d)
    revId = d['review_id']
    
    reviewPerId[revId] = d


# In[63]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    
    review = reviewPerId[b]
    res = mod.predict([feature8C(review)])
    
    line = u + "," + b + "," + str(res[0]) + "\n"
    predictions.write(line)
    # (etc.)
predictions.close()


# In[ ]:




