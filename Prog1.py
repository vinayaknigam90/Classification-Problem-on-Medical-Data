
# coding: utf-8

# In[78]:


import re 
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise


# In[79]:


# Reading Training DataSet 

trainDf = pd.read_csv(
    filepath_or_buffer='./train.dat', 
    header=None, 
    sep='\t')

print len(trainDf)


# In[80]:


# Reading Test DataSet 

testDf = pd.read_csv(
    filepath_or_buffer='./test.dat', 
    header=None, 
    sep='\t')

print len(testDf)


# In[81]:


# Removed All Html Tags and Special Character and Number Except Space  from Trainging and Test data set 


# TrainingSet
trainvalues = trainDf.iloc[:,:].values

#print trainvalues

trainRatings = []
trainMedicalAbstract = []

for value in trainvalues:
    trainRatings.append(value[0])
    trainMedicalAbstract.append(re.sub('[^A-z -]', '', re.sub('<[^>]*>','',value[1])).lower())
    
 
# TestSet 

testvalues = testDf.iloc[:,:].values
testMedicalAbstract = []

for value in testvalues:
    testMedicalAbstract.append(re.sub('[^A-z -]', '', re.sub('<[^>]*>','',value[0])).lower())


# In[82]:


def filterLen(docs, minlen):
    return[ [t for t in d if len(t) >= minlen ] for d in docs ]


# In[83]:


#Removing All words whose lenth is less than 4 as those words doesnot add any value to analysis it is just a noise for us .

trainDocs = [l.split() for l in trainMedicalAbstract]
testDocs = [l.split() for l in testMedicalAbstract]

trainDocs1 = filterLen(trainDocs,4)
testDocs1 =  filterLen(testDocs ,4)

 

#After Processing 

print trainDocs1
#print len(testDocs1[0])

#print testDocs1[0]



# In[84]:


combinedList = trainDocs1 + testDocs1
combinedList


# In[85]:


def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat


# In[86]:


combinedMatrix = build_matrix(combinedList)
combinedMatrix.shape
trainMatrix = combinedMatrix[0:14438]
testMatrix = combinedMatrix[14438:28880]


# In[87]:


def cosine_similarity_n_space(m1, m2, batch_size=10):
    assert m1.shape[1] == m2.shape[1]
    ret = np.ndarray((m1.shape[0], m2.shape[0]))
    for row_i in range(0, int(m1.shape[0] / batch_size) + 1):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, m1.shape[0]])
        if end <= start:
            break 
        rows = m1[start: end]
        sim = cosine_similarity(rows, m2) # rows is O(1) size
        ret[start: end] = sim
    return ret



# In[88]:


cosineSimilarityValue = cosine_similarity_n_space(testMatrix,trainMatrix)


# In[89]:


f = open('./format.dat', 'w')
count = 0
 
for row in cosineSimilarityValue:

    
    #kneighbours = heapq.nlargest(5, row)
    k=3
    partitioned_row_byindex = np.argpartition(-row, k)  
    similar_index = partitioned_row_byindex[:k]

    
    DiseasesTypeFirst = 0
    DiseasesTypeSecond = 0
    DiseasesTypeThird = 0
    DiseasesTypeFour = 0
    DiseasesTypeFive = 0

    for index in similar_index:

        if trainvalues[index][0] == 1:
               DiseasesTypeFirst+=1
        elif trainvalues[index][0] == 2:
               DiseasesTypeSecond+=1
        elif trainvalues[index][0] == 3:
               DiseasesTypeThird+=1
        elif trainvalues[index][0] == 4:
               DiseasesTypeFour+=1
        elif trainvalues[index][0] == 5:
               DiseasesTypeFive+=1
     
    maxValue = max(DiseasesTypeFirst,DiseasesTypeSecond,DiseasesTypeThird,DiseasesTypeFour,DiseasesTypeFive)
    if  maxValue == DiseasesTypeFirst:
        f.write('1\n')
        count +=1
    elif maxValue == DiseasesTypeSecond:
        f.write('2\n')
        count +=1
    elif maxValue == DiseasesTypeThird:
        f.write('3\n')
        count +=1
    elif maxValue == DiseasesTypeFour:
        f.write('4\n')
        count +=1
    elif maxValue == DiseasesTypeFive:
        f.write('5\n')
        count +=1
     
        
print("count : ",count)
 
 


# In[ ]:




