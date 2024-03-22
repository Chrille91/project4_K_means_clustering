import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from sklearn import set_config
set_config(transform_output="pandas")

##############################################################################################################################
# This function takes as input the original DataFrame and the scalar name and returns the scaled DataFrame
# The original DataFrame should not contain columns with string values

def get_scaled_df(name_df,scaler_name):
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
# It also plots the distance between centroids on heatmap if the "with_distances_heatmap" is set to "True"

def get_clusters(number_clusters,random_state,scaled_df,with_distances_heatmap):
    # Initialize the model
    my_kmeans = KMeans(n_clusters = number_clusters,
                   random_state = random_state)
    # Fit the model to the data
    my_kmeans.fit(scaled_df)
    # Obtain the cluster output
    cluster_output = my_kmeans.labels_
    
    if (with_distances_heatmap==True):
        # Find the coordinates of each centroid using the cluster_centers_ attribute
        centroids = my_kmeans.cluster_centers_
        
        # Calculate the euclidean distance between the centroids
        centroid_distances = pairwise_distances(centroids)

        # Plot distances on heatmap
        sns.heatmap(centroid_distances,
                annot=True,
                linewidths=1);
    
    return cluster_output

##############################################################################################################################
# This function takes as input the cluster_output and the scaled DataFrame and returns the cluster mean DataFrame

def get_cluster_mean_df(cluster_output, scaled_df):
    # Attach the cluster output to our original DataFrame
    scaled_df["cluster"]=cluster_output
    cluster_mean_df=scaled_df.groupby(by="cluster").mean()
    return cluster_mean_df

##############################################################################################################################
# This function takes as input the cluster mean DataFrame and the scaled DataFrame and plots the radar chart

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
# It takes as input the original DataFrame, the scalar name, the number of cluster and the random state and the option if we want to plot the clusters distances heatmap or not and then plots the radar chart

def plot_radar_chart_clusters(name_df,scaler_name,number_clusters,random_state,with_distances_heatmap):
    scaled_df=get_scaled_df(name_df,scaler_name)
    cluster_output=get_clusters(number_clusters,random_state,scaled_df,with_distances_heatmap)
    cluster_mean_df=get_cluster_mean_df(cluster_output, scaled_df)
    plot_radar_chart(scaled_df,cluster_mean_df)
    return scaled_df

##############################################################################################################################
# This function takes as input the original DataFrame, the scalar name, the minimum and maximum number of cluster, the random seed and the scoring method and plots the inertia scores

def plot_scores(name_df,scaler_name,nb_min_clusters,nb_max_clusters,random_state,method_name):
    scaled_df=get_scaled_df(name_df,scaler_name)

    # Create an empty list to store the inertia scores
    score_list = []

    # Iterate over the range of cluster numbers
    for k in range(nb_min_clusters, nb_max_clusters):

        # Create a KMeans object with the specified number of clusters
        myKMeans = KMeans(n_clusters=k,
                        n_init="auto",
                        random_state = random_state)

        # Fit the KMeans model to the scaled data
        myKMeans.fit(scaled_df)

        if method_name=="Inertia":
            # Append the inertia score to the list
            score_list.append(myKMeans.inertia_)
        elif method_name=="Silhouette":
            # Get the cluster labels
            labels = myKMeans.labels_
            # Calculate the silhouette score
            score = silhouette_score(scaled_df, labels)
            # Append the silhouette score to the list
            score_list.append(score)

    # Set the Seaborn theme to darkgrid
    sns.set_theme(style='darkgrid')

    (
    # Create a line plot of the inertia scores
    sns.relplot(y = score_list,
                x = range(nb_min_clusters, nb_max_clusters),
                kind = 'line',
                marker = 'o',
                height = 8,
                aspect = 2)
    # Set the title of the plot
    .set(title=f"{method_name} score from {nb_min_clusters} to {nb_max_clusters} clusters")
    # Set the axis labels
    .set_axis_labels("Number of clusters", f"{method_name} score")
    );
