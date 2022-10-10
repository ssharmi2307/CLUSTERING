# -*- coding: utf-8 -*-
"""
@author: Gopinath
"""

import warnings
warnings.filterwarnings('ignore')
#Loading dataset
import pandas as pd
df=pd.read_csv("crime_data.csv")
df
df.shape


#Normalisation
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
#feature variable
df_norm=norm_func(df.iloc[:,1:])
df_norm



#1.Heirarchical clustering
#creating dendogram
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df_norm,method='average'))
#create clusters
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=4,affinity='euclidean')
#saving cluster as chart
y_hc=hc.fit_predict(df_norm)
clusters=pd.DataFrame(y_hc,columns=['clustetrs'])
df_norm['h_clusterid']=clusters
df_norm.h_clusterid.value_counts()
df_norm.sort_values("h_clusterid")



#2.KMeans clustering
import pandas as pd
df1=pd.read_csv("crime_data.csv")
df1
df1.shape
#Normalisation
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
#feature variable
df1_norm=norm_func(df.iloc[:,1:])
df1_norm
#To find optimum method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df1_norm)
    dist=kmeans.inertia_
    wcss.append(dist)
#data visualization
import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.title("Elbow method")
plt.xlabel("NO OF CLUSTERS")
plt.ylabel("Within cluster sum of square")
plt.grid()
plt.show()
#cluster algorithm
from sklearn.cluster  import KMeans
clusters_new=KMeans(n_clusters=4,random_state=0)
clusters_new.fit(df1_norm)
y=kmeans.predict(df1_norm)
#standardised values
clusters_new.cluster_centers_
x=pd.Series(clusters_new.labels_)
df1['Clust']=x
df1
df1.iloc[:,1:5].groupby(df1.Clust).mean()




#3.DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
#Loading dataset
df2=pd.read_csv('crime_data.csv')
df2.info()
#feature variable
df2=df2.iloc[:,1:5]
df2.values
#Standardisation
sc=StandardScaler().fit(df2.values)
x=sc.transform(df2.values)
x
dbscan=DBSCAN(eps=2,min_samples=5)
dbscan.fit(x)
dbscan.labels_
cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
pd.concat([df2,cl],axis=1)















