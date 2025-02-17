import numpy as np

def nms(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on the bounding boxes.

    Parameters:
    - boxes: List of bounding boxes (x1, y1, x2, y2)
    - scores: List of confidence scores for each box
    - iou_threshold: Intersection-over-Union (IoU) threshold for NMS

    Returns:
    - List of indices of the boxes to keep
    """
    if len(boxes) == 0:
        return []

    # Convert to numpy arrays for vectorized operations
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the boxes by their confidence scores in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of the remaining boxes with the highest scored box
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than the threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# Example usage
boxes = [[100, 100, 210, 210], [105, 105, 215, 215], [150, 150, 250, 250]]
scores = [0.9, 0.75, 0.6]
iou_threshold = 0.3
selected_indices = nms(boxes, scores, iou_threshold)
print("Selected indices:", selected_indices)
