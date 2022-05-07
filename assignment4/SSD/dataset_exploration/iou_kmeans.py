import numpy as np

def iou(boxes, clusters):
    n = boxes.shape[0]
    k = clusters.shape[0]    

    box_area = boxes[:, 0] * boxes[:, 1]
    box_area = box_area.repeat(k).reshape((-1, k))

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = cluster_area.repeat(n).reshape((k,n)).T


    def get_min_across_axis(boxes, clusters, axis):
        n = boxes.shape[0] # Number of boxes
        k = clusters.shape[0] # Number of clusters
        
        # Broadcasts the boxes and clusters and returns an (n, k) array where
        # each row represents the minimum intersection each box has with each cluster (column) on that axis.

        boxes = boxes[:,axis]
        clusters = clusters[:,axis]
    
        # Each box axis value is repeated for each cluster
        box_axis_broadcast_k = boxes.repeat(k).reshape((n, k))
        # Each cluster axis value is repeated for each box, reshaped and transposed.
        cluster_axis_broadcast_n = clusters.repeat(n).reshape((k, n)).T
        return np.minimum(box_axis_broadcast_k, cluster_axis_broadcast_n)


    min_w_similarity = get_min_across_axis(boxes, clusters, 0)
    min_h_similarity = get_min_across_axis(boxes, clusters, 1)
    intersection_area = min_w_similarity * min_h_similarity

    # Union = Area Box + Area Cluster - Area Intersection
    return intersection_area / (box_area + cluster_area - intersection_area)


def avg_iou(boxes, clusters):
    highest_iou = np.max(iou(boxes, clusters), axis=1)
    return np.mean([highest_iou])


def iou_kmeans(boxes, k, random_state=None):
    # Implements a standard k-means algorithm with a custom distance metric.
    
    num_boxes = boxes.shape[0]

    distance = np.zeros((num_boxes, k)) # Create a [num_box, k] array to hold distance for each box to each cluster centroid.
    last_cluster = np.zeros((num_boxes, )) # Initialize each box to be tagged as cluster 0.
    init_box_choices = np.random.choice(num_boxes, k, replace=False) # Randomly pick a starting centroid.
    clusters = boxes[init_box_choices]

    mean_losses = []

    while True:

        distance = 1 - iou(boxes, clusters)
        mean_losses.append(np.mean(distance))

        current_nearest = np.argmin(distance, axis=1)
        if (last_cluster == current_nearest).all():
            break # We are done

        for cluster_idx in range(k):
            new_boxes_in_cluster = boxes[current_nearest == cluster_idx]
            clusters[cluster_idx] = np.mean(new_boxes_in_cluster, axis=0)

        last_cluster = current_nearest
        
    return clusters, last_cluster