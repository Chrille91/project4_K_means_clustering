import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn import set_config
set_config(transform_output="pandas")

##############################################################################################################################
# This function takes as input the original DataFrame and the scalar name and returns the scaled DataFrame
# The original DataFrame should not contain columns with string values

def getScaled_df(name_df,scaler_name):
    # Create a scaler object
    match scaler_name:
        case "MinMaxScaler":scaler = MinMaxScaler()
        case "StandardScaler":scaler = StandardScaler()
        case "RobustScaler":scaler = RobustScaler()
        case "QuantileTransformer":scaler = QuantileTransformer(n_quantiles = name_df.shape[0])
        case "PowerTransformer":scaler = StandardScaler()
    
    # Scale the DataFrame
    scaled_df = scaler.fit_transform(name_df)

    return scaled_df

##############################################################################################################################
# This function takes as input the number of clusters, the random seed and the scaled DataFrame and returns the cluster output

def get_clusters(number_clusters,random_seed,scaled_df):
    # Initialize the model
    my_kmeans = KMeans(n_clusters = number_clusters,
                   random_state = random_seed)
    # Fit the model to the data
    my_kmeans.fit(scaled_df)
    # Obtain the cluster output
    cluster_output = my_kmeans.labels_
    
    return cluster_output
    
##############################################################################################################################
# This function takes as input the cluster_output and the scaled DataFrame and returns the cluster mean DataFrame

def get_cluster_mean_df(cluster_output, scaled_df):
    # Attach the cluster output to our original DataFrame
    scaled_df["cluster"]=cluster_output
    cluster_mean_df=scaled_df.groupby(by="cluster").mean()
    return cluster_mean_df

##############################################################################################################################
# This function takes as input the cluster mean DataFrame and the scaled DataFrame and plot the radar shart

def plot_radar_chart(scaled_df,cluster_mean_df):

    # State the label for each arm of the chart
    categories = list(scaled_df.columns.values)
    categories.remove("cluster")

    # Create an empty list to store the objects
    trace_objects = []

    # Iterate over the unique cluster numbers and add an object for each cluster to the list
    for cluster in sorted(scaled_df['cluster'].unique()):
        r_list=[]
        for cat in categories: 
            r_list.append(scaled_df.loc[scaled_df["cluster"] == cluster, cat].mean())
               
        cluster_mean_df = go.Scatterpolar(
        r=r_list,
        theta=categories,
        fill='toself',
        name=f'Cluster {cluster}'
        )
    
        trace_objects.append(cluster_mean_df)

    # Add the objects to the figure
    fig = go.Figure()
    fig.add_traces(trace_objects)

    # Add extras to the plot such as title
    # You'll always need `polar=dict(radialaxis=dict(visible=True,range=[0, 1]))` when creating a radar plot
    fig.update_layout(
    title_text = 'Radar chart of mean food preferences by cluster',
    height = 600,
    width = 800,
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    # Show the initialised plot and the layers
    fig.show()

##############################################################################################################################
# This function uses all the above functions
# It takes as input the original DataFrame, the scalar name, the number of cluster and the random seed and plot the radar shart
def plot_radar_chart_clusters(name_df,scaler_name,number_clusters,random_seed):
    scaled_df=getScaled_df(name_df,scaler_name)
    cluster_output=get_clusters(number_clusters,random_seed,scaled_df)
    cluster_mean_df=get_cluster_mean_df(cluster_output, scaled_df)
    plot_radar_chart(scaled_df,cluster_mean_df)
