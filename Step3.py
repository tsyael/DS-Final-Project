import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from plotly.tools import FigureFactory as FF
from scipy.cluster.hierarchy import linkage
import plotly.plotly as py

def plot_dendrogram(model, **kwargs):
    children = model.children_
    distance = np.arange(children.shape[0])
    no_of_observations = np.arange(2, children.shape[0] + 2)
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def min_max(x):
    # Formula for min-max normalization : (by discount price)
    # newvalue= (max'-min')/(max-min)*(value-max)+max'
    max_x = x.max()
    min_x = x[x != -1].min()
    x = x.apply(lambda v: -1 if np.isnan(v) else
                (100-0)/(max_x-min_x)*(v-max_x)+100)

    return x

if __name__ == "__main__":
    df = pd.read_csv("Hotels_data_Changed.csv")
    df.set_index('Unnamed: 0', inplace=True)
    new_df = df.groupby('Hotel Name')['Hotel Name'].count().reset_index(name='count').\
       sort_values(['count'], ascending=False).head(150)
    new_df = pd.merge(new_df[['Hotel Name']], df, on='Hotel Name', how='left')

    new_df_1 = new_df.groupby('Checkin Date')['Checkin Date'].count().reset_index(name='count_checkin').\
        sort_values(['count_checkin'],ascending=False).head(40)

    new_df_1 = pd.merge(new_df_1[['Checkin Date']], new_df, on='Checkin Date', how='left')

    se = new_df_1.groupby(['Hotel Name', 'Checkin Date', 'Discount Code'])['Discount Price'].min()
    new_df_2 = pd.DataFrame(se)
    new_df_2.reset_index(inplace=True)

    new_df_2['Checkin and code'] = new_df_2['Checkin Date'].astype(str) + ' && ' +  new_df_2['Discount Code'].astype(str)

    new_df_3 = new_df_2.pivot(index='Hotel Name', columns='Checkin and code', values='Discount Price')

    new_df_3.columns.name = None
    new_df_3 = new_df_3.apply(min_max,axis=1).fillna(-1)
    #new_df_3.reset_index(inplace=True)

    new_df_3.to_csv("Hotels_data_for_clustering.csv")

    ## Put in a new file!!!

    model = AgglomerativeClustering(n_clusters=4, compute_full_tree=False)
    model = model.fit(new_df_3.head(20)) # Remove the head!!
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, labels=model.labels_)
    plt.show()

    ##figure = FF.create_dendrogram(model, orientation='bottom', labels=model.labels_, linkagefun=lambda x: linkage(model, 'ward', metric='euclidean'))
    ##figure = FF.create_dendrogram(new_df_3, orientation='bottom',linkagefun=lambda x: linkage(new_df_3, 'ward', metric='euclidean'))
    ##py.iplot(figure, filename='dendrogram_with_labels')





