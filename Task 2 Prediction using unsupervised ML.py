#!/usr/bin/env python
# coding: utf-8

# # SAKSHI GUPTA
# # Task-2
# # From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.

# In[2]:


#importing all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as mt
from sklearn import datasets
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df.head(150)


# In[10]:


x=df.iloc[:,[0,1,2,3]].values
from sklearn.cluster import KMeans
list=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    list.append(kmeans.inertia_)

mt.plot(range(1,11),list)
mt.title("The Elbow Method")
mt.xlabel("Number of Clusters")
mt.ylabel("WCSS")
mt.show()


# In[11]:


#creating the kmeans classifier
kmeans=KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(x)


# In[17]:


#visualising th clusters
mt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1], s=100, c='green', label='Iris-setosa')
mt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1], s=100, c='orange', label='Iris-versicolor')
mt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1], s=100, c='blue', label='Iris-verginica')

#plotting centriods of clusters
mt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centriods')
mt.legend()

