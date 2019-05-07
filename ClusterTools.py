# -*- coding:utf-8 -*-
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from sklearn import metrics
from matplotlib.colors import LogNorm


#--------Basic Process--------
def display_columns_all(width=1000,max_columns=500):
    '''
    To adjust display width and max_columns
    :param width:int
    :param max_columns:int
    :return:none
    '''
    pd.set_option('display.width',width)
    pd.set_option('display.max_columns',max_columns)

#--------Preprocess Data -----------
def get_attribute(data,attr_str):
    '''
    Get factors in an attribution and give them an integer ID (start from 0)
    :data:DataFrame
    :attr_str:str
    :return:DataFrame,dict,dict
    '''
    attr=pd.unique(data[attr_str])
    attr_df = pd.DataFrame([], columns=['attr_code', 'attr_name'])
    attr_dict_name2code = {}
    attr_dict_code2name = {}
    for i in range(len(attr)):
        attr_df=attr_df.append(pd.DataFrame([[i,attr[i]]],columns=['attr_code','attr_name']))
        attr_dict_name2code[attr[i]]=i
        attr_dict_code2name[i]=attr[i]

    return attr_df,attr_dict_name2code,attr_dict_code2name

def print_dict(data):
    '''
    Print data in a dictionary
    :param data:dict
    :return: None
    '''
    for i in data:
        print str(i) + ":" +str(data[i])

def deal_na_row(data,how='drop'):
    '''
    Drop or fill all na in every row
    :param data: DataFrame
    :param how: str
    :return: DataFrame,bool
    '''
    if how=='drop':
       new_data=data.dropna(axis=0,how='any')
    elif how=='fill':
        new_data=data.fillna(axis=0,how='ffill')
        new_data=new_data.fillna(axis=0,how='bfill')
    existNa=new_data.isna().any().any()
    return new_data,existNa

def get_na_row(data):
    '''
    Get rows contain na
    :param data: DataFrame
    :return: na_data
    '''
    return data[data.isna().any(axis=1)]

def drop_negative(data):
    '''
    Drop rows contain any negative
    :param data: DataFrame
    :return: DataFrame,DataFrame
    '''
    dataBool=data<0
    data_negative=data[dataBool.any(axis=1)]
    data_none=data[~dataBool.any(axis=1)]
    return data_none,data_negative

def normalize_data(data):
    '''
    To nomalize data
    :param data:DataFrame
    :return: DataFrame
    '''
    from sklearn import preprocessing
    data_value_scaled=preprocessing.scale(data.values)
    new_data=pd.DataFrame(data_value_scaled,index=data.index,columns=data.columns)
    return new_data

def percent_by_row(data):
    '''
    Calculate every single columns' percent of sum of all columns
    :param data: DataFrame
    :return: DataFrame
    '''
    new_data=data.copy()
    new_data['all_sum']=new_data.sum(axis=1)
    new_data=new_data.div(new_data['all_sum'],axis=0)
    del new_data['all_sum']
    return new_data

#-----------Detect the data-----------
def show_boxplot(data,title='Boxplot of features'):
    '''
    Plot boxplot
    :param data:DataFrame
    :return:None
    '''
    from matplotlib.font_manager import FontProperties
    font_set = FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc', size = 8)
    columns_labels = data.columns
    n_features = len(columns_labels)
    x_array = range(1,n_features+1)
    data.boxplot()
    plt.xticks(x_array, columns_labels, fontproperties=font_set, rotation=30)
    plt.title(title, fontproperties=font_set)
    plt.show()

def show_hist(data,bins=30,rows=6,cols=8,title='Distribution of Features'):
    '''

    :param data:DataFrame
    :param bins: int
    :param rows:int
    :param cols:int
    :param title: str
    :return: None
    '''

    fig,axes=plt.subplots(rows,cols)
    from matplotlib.font_manager import FontProperties
    font_set = FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc', size=8)
    for i in range(len(data.columns)):
        row=i/cols
        col=i%cols
        axes[row][col].hist(data[data.columns[i]],bins=bins)
    plt.show()



def show_2d_distribution(x1, x2, y, hist_nbins=50, title="",
                         x0_label="", x1_label="", figsize=(10, 6)):
    from matplotlib.font_manager import FontProperties
    font_set = FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc', size=8)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    # define the axis for the first plot
    left, width = 0.1, 0.7
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.07, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)
    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01
    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    axes=(ax_scatter, ax_histy, ax_histx)
    ax, hist_x1, hist_x2 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label,FontProperties=font_set)
    ax.set_ylabel(x1_label,FontProperties=font_set)
    # The scatter plot
    colors = cm.plasma_r(y)
    ax.scatter(x1, x2, alpha=1, marker='o', s=10, lw=0, c=colors)
    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Histogram for axis X1 (feature 5)
    hist_x1.set_ylim(ax.get_ylim())
    hist_x1.hist(x1, bins=hist_nbins, orientation='horizontal',
                 color='grey', ec='grey')
    hist_x1.axis('off')
    # Histogram for axis X0 (feature 0)
    hist_x2.set_xlim(ax.get_xlim())
    hist_x2.hist(x2, bins=hist_nbins, orientation='vertical',
                 color='grey', ec='grey')
    hist_x2.axis('off')
    norm = mpl.colors.Normalize(y.min(), y.max())
    mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cm.plasma_r,
                              norm=norm, orientation='vertical',
                              label='Color mapping for values of y')
    plt.show()

#--------Cluster--------
def cluster_estimate(Estimator,cluster_x):
    '''
    Using estimator which has been trained to predict new points
    :param Estimater:Cluster
    :param cluster_x: [int]
    :return:
    '''
    return Estimator.fit_predict(cluster_x)

#--------Analysis clustering result-------
def cluster_count(cluster_labels):
    '''
    Calculate the number of points in each cluster
    :param cluster_labels:narray[int]
    :return: None
    '''
    nclusters=cluster_labels.max()+1
    count_list=[]
    for i in range(nclusters):
        count_list.append(0)
    for i in cluster_labels:
        count_list[i]+=1
    print '----------------'
    for i in range(nclusters):
        print 'cluster ' + str(i) + ':  ' + str(count_list[i])
    print '----------------'
    plt.bar(range(nclusters),count_list)
    for a,b in zip(range(nclusters),count_list):
        plt.text(a,b+0.05,'%.0f'%b,ha='center',va='bottom',fontsize=7)

def plot_features(cluster_x,columns_labels,title):
    '''
    Plot line chart which x axis contains all features for high dimension points
    :param cluster_x:narray[[float]]
    :param columns_labels:[str]
    :param title:str
    :return: None
    '''
    from matplotlib.font_manager import  FontProperties
    import matplotlib.pyplot as plt
    font_set=FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc', size=8)
    n_features=len(cluster_x[0])
    x_array=range(n_features)
    for x in cluster_x:
        plt.plot(x_array,x,alpha=0.35)
    plt.xticks(x_array,columns_labels,fontproperties=font_set,rotation=30)
    plt.title(title,fontproperties=font_set)

def get_cluster_x(cluster_x,cluster_labels,cluster):
    '''
    Get cluster_x which contains all points in the cluster
    :param cluster_x: [[float]]
    :param columns_labels: [int]
    :param cluster: int
    :return: [[float]]
    '''
    x=[]
    n=len(cluster_labels)
    for i in range(n):
        if cluster_labels[i]==cluster:
            x.append(cluster_x[i])
    return x

def get_cluster_score(cluster_x,cluster_labels,cluster_name):
    '''
    Get cluster score (Silhouette Score and Calinski Harabaz)
    :param cluster_x:[[float]]
    :param cluster_labels: [int]
    :param cluster_name: str
    :return:float,float
    '''
    silhouette_score = metrics.silhouette_score(cluster_x,cluster_labels,metric='euclidean')
    calinski_harabaz_score = metrics.calinski_harabaz_score(cluster_x,cluster_labels)
    print '-----' + cluster_name + '-----'
    print 'Silhouette Score is %.2f' % silhouette_score
    print 'Calinski Harabaz is %.2f' % calinski_harabaz_score
    return silhouette_score,calinski_harabaz_score

def plot_features_all_clusters(cluster_x,cluster_y,columns_labels,title):
    for i in np.unique(cluster_y):
        plt.figure()
        plot_features(cluster_x[cluster_y==i],columns_labels,title + str(i))


def plot_heatmap(cluster_x,columns_labels,title='heatmap'):
    x, y = np.random.rand(10), np.random.rand(10)
    plt.imshow(cluster_x, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
               cmap=cm.hot, norm=LogNorm())
    plt.colorbar()
    plt.show()

def plot_heatmap(cluster_x,columns_labels,title='heatmap'):
    from matplotlib.font_manager import  FontProperties
    font_set=FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc', size=6)
    plt.imshow(cluster_x,cmap=cm.hot,aspect='auto')
    plt.tight_layout()
    plt.colorbar()
    plt.xticks(range(0,len(columns_labels)),columns_labels,FontProperties=font_set,rotation=30)
    plt.title(title, FontProperties=font_set)
    plt.xlabel(u'行业',FontProperties=font_set)
    plt.ylabel(u'客户', FontProperties=font_set)
    plt.show()

def plot_all_cluster_heatmap(cluster_x,cluster_y,columns_labels,save=True,path=""):
    cluster_labels=np.unique(cluster_y)
    for i in cluster_labels:
        plt.figure(figsize=(12,8))
        plot_heatmap(cluster_x[cluster_y==i],columns_labels,'Cluster id is %0.0d and number of customers is %0.0d' %(i,np.array(cluster_y==i).sum()))
        if save:
            plt.savefig(path+'Cluster id is %0.0d and number of customers is %0.0d' %(i,np.array(cluster_y==i).sum()))

def plot_box(df,cluster_y,title='Boxplot'):
    uniq_y=np.unique(cluster_y)
    d=[]
    for i in uniq_y:
        d.append(df[cluster_y==i].values)
    plt.boxplot(d)
    plt.title(title)
    plt.show()
