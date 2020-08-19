# EastWest Airlines Data Hierarchical Clustering

import pandas as pd
import matplotlib.pylab as plt 
ewa = pd.read_csv('E:\Data\Assignments\i made\clusterinng\EastWestAirlines.csv')

df = pd.DataFrame(ewa)
EWA = df.drop(columns = ['ID#'], axis=1)
             
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# alternative normalization function 

#def norm_func(i):
#    x = (i-i.mean())/(i.std())
#    return (x)

# Normalized data frame (considering the numerical ignoring the nominal)
df_norm = norm_func(EWA.iloc[:,0:12])
df_norm.describe() # this needs to be max=1 and min=0 means the normalization done properly

# applying linkage (single, complete, average, weighted, centroid, so on)
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

help(linkage)

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 

# to check the to which cluster the data point belongs to
cluster_labels=pd.Series(h_complete.labels_)
cluster_labels

# creating a  new column clust and assigning it to new column 
EWA['clust']=cluster_labels # creating a  new column and assigning it to new column 
EWA = EWA.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
EWA.head()

# getting aggregate mean of each cluster
EWA.groupby(EWA.clust).mean()

# creating a csv file 
EWA.to_csv("crime_data.csv",encoding="utf-8")
EWA.to_csv("crimedata.csv",index=False)




# EastWest Airlines Data K-Means Clustering

import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

ewa = pd.read_csv('E:\Data\Assignments\i made\clusterinng\EastWestAirlines.csv')

#.....EDA..........
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ewa.iloc[:,1:])


df_norm.head(10)  # Top 10 rows

###### scree plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
ewa['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()

EWA = ewa.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

EWA.iloc[:,1:12].groupby(EWA.clust).mean()

EWA.to_csv("eastwestairlines.csv")

















