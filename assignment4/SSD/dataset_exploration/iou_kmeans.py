def iou(boxes, clusters):
    """ Calculates Intersection over Union between the provided boxes and cluster centroids
        Input:
            boxes: Bounding boxes
            clusters: cluster centroids
        Output:
            IoU between boxes and cluster centroids
    """
    n = boxes.shape[0]
    k = np.shape(clusters)[0]

    box_area = boxes[:, 0] * boxes[:, 1] # Area = width * height
    # Repeating the area for every cluster as we need to calculate IoU with every cluster
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))


    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.multiply(min_w_matrix, min_h_matrix)

    result = inter_area / (box_area + cluster_area - inter_area)
    return result


def avg_iou(boxes, clusters):
    """ Calculates average IoU between the GT boxes and clusters 
        Input:
            boxes: array, having width and height of all the GT boxes
        Output:
            Returns numpy array of average IoU with all the clusters
    """
    return np.mean([np.max(iou(boxes, clusters), axis=1)])


def iou_kmeans(boxes, k):
    """ Executes k-means clustering on the rovided width and height of boxes with IoU as
        distance metric.
        Input:
            boxes: numpy array containing with and height of all the boxes
        Output:
            clusters after convergence
    """
    num_boxes = boxes.shape[0]
    distances = np.empty((num_boxes, k))
    last_cluster = np.zeros((num_boxes, ))

    # Initializing the clusters
    np.random.seed()
    clusters = boxes[np.random.choice(num_boxes, k, replace=False)]

    # Optimizarion loop
    while True:

        distances = 1 - iou(boxes, clusters)
        mean_distance = np.mean(distances)
        sys.stdout.write('\r>> Mean loss: %f' % (mean_distance))
        sys.stdout.flush()

        current_nearest = np.argmin(distances, axis=1)
        if(last_cluster == current_nearest).all():
            break # The model is converged
        for cluster in range(k):
            clusters[cluster] = np.mean(boxes[current_nearest == cluster], axis=0)

        last_cluster = current_nearest
    return clusters, last_cluster