"""Split two distinct writing lobes inside one detector balloon box.

A single bubble_xyxy may surround two touching speech balloons. The original
text boxes remain separate; partition the measured interior at the two lobes'
distance-transform peaks so their translations never share a writing region.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def _centre(box):
    return np.array([(float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2])


def separated_text_boxes(boxes, bubble):
    if len(boxes) != 2:
        return False
    a, b = boxes
    areas = [max(1., float(t[2]-t[0]) * float(t[3]-t[1])) for t in boxes]
    overlap = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    if overlap > 0.08 * min(areas):
        return False
    span = min(float(bubble[2]-bubble[0]), float(bubble[3]-bubble[1]))
    return np.linalg.norm(_centre(a)-_centre(b)) > 0.36 * span


def split_source_paragraphs(image, text_box, bubble):
    """Split a merged OCR region only with BOTH a paragraph gap and a two-lobe silhouette.

    Text inside one ordinary oval remains one block, however many rows it has.
    Runs are measured on the original pixels, before OCR or translation.
    """
    from modules.rendering.text_style import row_runs
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = map(int, bubble)
    bx1, by1, bx2, by2 = max(0, bx1), max(0, by1), min(w, bx2), min(h, by2)
    x1, y1, x2, y2 = map(int, text_box)
    x1, y1, x2, y2 = max(bx1, x1), max(by1, y1), min(bx2, x2), min(by2, y2)
    if x2-x1 < 40 or y2-y1 < 100:
        return None
    crop = image[by1:by2, bx1:bx2, :3]
    if not crop.size:
        return None
    bright = ((crop.min(axis=2) > 205) & (crop.max(axis=2)-crop.min(axis=2) < 45)).astype(np.uint8)
    _, labels = cv2.connectedComponents(bright, connectivity=4)
    counts = np.bincount(labels[y1-by1:y2-by1, x1-bx1:x2-bx1].ravel())
    if len(counts) < 2:
        return None
    counts[0] = 0
    comp = (labels == int(counts.argmax())).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    interior = np.zeros_like(comp)
    cv2.drawContours(interior, contours, -1, 1, -1)
    interior = cv2.erode(interior, np.ones((7, 7), np.uint8))
    ink = (image[y1:y2, x1:x2, :3].mean(axis=2) < 150) & \
          interior[y1-by1:y2-by1, x1-bx1:x2-bx1].astype(bool)
    runs = row_runs(ink, min_height=8)
    if len(runs) < 4:
        return None
    candidates = [(runs[i+1][0]-runs[i][1], i) for i in range(1, len(runs)-2)]
    if not candidates:
        return None
    gap, index = max(candidates)
    if gap < max(24, 1.2*np.median([b-a for a, b in runs])):
        return None
    cut = (runs[index][1] + runs[index+1][0]) // 2
    boxes = []
    for top, bottom in [(0, cut), (cut, y2-y1)]:
        ys, xs = np.nonzero(ink[top:bottom])
        if not len(xs):
            return None
        boxes.append(np.array([max(x1, x1+xs.min()-3), max(y1, y1+top+ys.min()-3),
                               min(x2, x1+xs.max()+4), min(y2, y1+top+ys.max()+4)], dtype=np.int32))
    if split_lobe_interiors(interior, (bx1, by1), boxes, bubble) is None:
        return None
    return boxes


def _peak_near(dist, centre, radius):
    h, w = dist.shape
    x, y = map(int, np.rint(centre))
    left, right = max(0, x-radius), min(w, x+radius+1)
    top, bottom = max(0, y-radius), min(h, y+radius+1)
    if left >= right or top >= bottom:
        return None
    patch = dist[top:bottom, left:right]
    yy, xx = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return left+xx, top+yy, float(patch[yy, xx])


def _largest_rect(grid):
    heights = np.zeros(grid.shape[1], dtype=np.int32)
    best_area, best = 0, (0, 0, 0, 0)
    for y in range(grid.shape[0]):
        heights = np.where(grid[y], heights+1, 0)
        stack = []
        for x in range(grid.shape[1]+1):
            height = heights[x] if x < grid.shape[1] else 0
            while stack and heights[stack[-1]] >= height:
                i = stack.pop()
                left = stack[-1]+1 if stack else 0
                area = int(heights[i]) * (x-left)
                if area > best_area:
                    best_area, best = area, (left, y-int(heights[i])+1, x, y+1)
            stack.append(x)
    return best


def split_lobe_interiors(inner, offset, boxes, bubble):
    """Two interior masks and safe rectangles, or None for an ordinary ellipse."""
    if inner is None or not separated_text_boxes(boxes, bubble):
        return None
    h, w = inner.shape
    if min(h, w) < 65:
        return None
    mask = np.ascontiguousarray(inner, dtype=np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    radius = max(14, int(0.17 * min(h, w)))
    peaks = [_peak_near(dist, _centre(b)-np.asarray(offset), radius) for b in boxes]
    if any(p is None for p in peaks):
        return None
    a, b = peaks
    sep = math.hypot(a[0]-b[0], a[1]-b[1])
    if sep < 0.38 * min(h, w) or min(a[2], b[2]) < 0.115 * min(h, w):
        return None
    # Joined lobes have a narrow saddle; one continuous ellipse does not.
    samples = []
    for t in np.linspace(0.32, 0.68, 9):
        x = int(round(a[0]*(1-t)+b[0]*t))
        y = int(round(a[1]*(1-t)+b[1]*t))
        samples.append(float(dist[y, x]))
    if min(samples) > 0.86 * min(a[2], b[2]):
        return None
    yy, xx = np.ogrid[:h, :w]
    # Peaks identify the silhouette, but are poor ownership anchors when one
    # paragraph is much wider than the other. Distance to the source rectangles
    # keeps each complete paragraph in its own lobe instead of cutting its left
    # half off with a diagonal bisector through the original lettering.
    distances = []
    for box in boxes:
        x1, y1, x2, y2 = np.asarray(box, dtype=float) - np.tile(offset, 2)
        dx = np.maximum(np.maximum(x1-xx, xx-x2), 0)
        dy = np.maximum(np.maximum(y1-yy, yy-y2), 0)
        distances.append(dx*dx + dy*dy)
    d0, d1 = distances
    regions = []
    for part in ((mask.astype(bool) & (d0 <= d1)), (mask.astype(bool) & (d0 > d1))):
        step = max(1, min(h, w)//210)
        x1, y1, x2, y2 = _largest_rect(part[::step, ::step])
        rect = (x1*step+offset[0], y1*step+offset[1],
                min(w, x2*step)+offset[0], min(h, y2*step)+offset[1])
        if (rect[2]-rect[0])*(rect[3]-rect[1]) < 0.025*w*h:
            return None
        regions.append((rect, part, offset))
    return regions


def fallback_lobe_regions(boxes, image_shape):
    """For open/unmeasurable outlines keep each OCR text near its own origin."""
    h, w = image_shape[:2]
    centers = [_centre(b) for b in boxes]
    regions = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = x2-x1, y2-y1
        region = [max(0, int(x1-.22*bw)), max(0, int(y1-.30*bh)),
                  min(w, int(x2+.22*bw)), min(h, int(y2+.30*bh))]
        if abs(centers[0][1]-centers[1][1]) >= abs(centers[0][0]-centers[1][0]):
            middle = int(round((centers[0][1]+centers[1][1])/2))
            if centers[i][1] < middle:
                region[3] = min(region[3], middle)
            else:
                region[1] = max(region[1], middle)
        else:
            middle = int(round((centers[0][0]+centers[1][0])/2))
            if centers[i][0] < middle:
                region[2] = min(region[2], middle)
            else:
                region[0] = max(region[0], middle)
        regions.append(tuple(region))
    return regions
