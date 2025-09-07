import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
# Read the points from the Excel file into a DataFrame
df = pd.read_excel('data_ogm.xlsx', header=None, names=['timestamp','x', 'y'])


#df['timestamp'] = df['timestamp'].div(10**9).astype(int)

# Group the data by timestamp
groups = df.groupby(df['timestamp'])

# Create an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['timestamp', 'Hull number', 'Vertices', 'Hull Indices', 'Edges', 'Width', 'Height', 'Area', 'Aspect Ratio of Hull', 'Compactness of Hull', 'Circularity of Hull'])

# Iterate over the groups
for name, group in groups:
    # Extract the points as a NumPy array
    points = group[['x','y']].values
    # Run the DBSCAN clustering algorithm
    dbscan = DBSCAN(eps=2, min_samples=3).fit(points)

    # Extract the labels of the clusters
    labels = dbscan.labels_

    # Get the number of clusters
    num_clusters = len(np.unique(labels))
    #print(num_clusters)
    # Create a color map for the clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))

    # Plot the points
    for cluster, color in zip(range(num_clusters), colors):
        # Extract the points for the current cluster
        cluster_points = points[labels == cluster]

        # Skip clusters with fewer than 3 points
        if cluster_points.shape[0] < 3:
            continue

        # Plot the points in the cluster
        #plt.scatter(cluster_points[:,0], cluster_points[:,1], color=color, s=50, label='Cluster '+str(cluster))
        #for i, point in enumerate(cluster_points):
            #plt.text(point[0], point[1], str(i), color='black', fontsize=8)

        # Compute the convex hull of the points
        hull = ConvexHull(cluster_points, qhull_options='QJ')


        # Extract the vertices of the convex hull
        vertices = cluster_points[hull.vertices,:]

        # Extract the indices of the points that are part of the convex hull
        hull_indices = hull.vertices

        # Extract the edges of the convex hull
        edges = cluster_points[hull.simplices,:]

        # Get the width and height of the bounding box of the convex hull
        min_x, min_y = vertices.min(axis=0)
        max_x, max_y = vertices.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y

        # Get the area of the convex hull
        area = hull.area

        # Get the number of points in the cluster
        num_points = cluster_points.shape[0]

        # Calculate the density of the hull
        density = area / num_points
        #print("Density of Hull:",density)
        
        # Calculate the standard deviation of the x and y coordinates of the points in the cluster
        std_x = np.std(cluster_points[:, 0])
        std_y = np.std(cluster_points[:, 1])
        #print("Standard deviation of x coordinates:", std_x)
        #print("Standard deviation of y coordinates:", std_y)

        # Calculate the compactness of the hull
        if width*height != 0:
            compactness = area / (width * height)
        else:
            compactness = 0

        # Calculate the aspect ratio of the hull
        if height != 0:
            aspect_ratio = width / height
        else:
            aspect_ratio = 0

        # Calculate the circularity of the hull
        circularity = (4 * np.pi * area) / (width * width + height * height)

        # Print the results
        #print("Cluster: ", cluster)
        #print("Vertices: ")
        #print(vertices)
        #print("Hull indices: ")
        #print(hull_indices)
        #print("Edges: ")
        #print(edges)
        #print("Width: ",width)
        #print("Height: ",height)
        #print("Area: ",area)
        #print("Aspect Ratio of Hull:", aspect_ratio)
        #print("Compactness of Hull:", compactness)
        print(name)
        #print("Circularity of Hull:", circularity)

        #result_df = result_df.append({'timestamp': name, 'Cluster': cluster, 'Vertices': vertices, 'Hull Indices': hull_indices, 'Edges': edges, 'Width': width, 'Height': height, 'Area': area, 'Aspect Ratio of Hull': aspect_ratio, 'Compactness of Hull': compactness, 'Circularity of Hull': circularity}, ignore_index=True)
        temp_df = pd.DataFrame({'timestamp': [name], 'Hull number': [cluster], 'Vertices': [vertices], 'Hull Indices': [hull_indices], 'Edges': [edges], 'Width': [width], 'Height': [height], 'Area': [area], 'Aspect Ratio of Hull': [aspect_ratio], 'Compactness of Hull': [compactness], 'Circularity of Hull': [circularity]})
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
        # Plot the points in the cluster
        plt.scatter(-cluster_points[:,0], cluster_points[:,1], color=color, s=50, label='Cluster '+str(cluster))
        # Mark the indices of the points on the convex hull
        for i in hull_indices:
            if cluster_points.shape[0] == 0:
                continue
            plt.text(-cluster_points[i,0], cluster_points[i,1], str(i), color='b', fontsize=10)
        # Plot the convex hull around the cluster
        for simplex in hull.simplices:
            if cluster_points.shape[0] == 0:
                continue
            plt.plot(-cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-')
    # Set the title of the plot to the timestamp
    plt.title(name)

    # Display the plot
    
    plt.legend()
    

    plt.show()

    # Write the results to an Excel file
result_df.to_excel('result_final.xlsx', index=False)






