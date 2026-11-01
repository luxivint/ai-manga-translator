"""Model-free RGB inpainting. Never changes pixels outside the supplied mask.

adaptive: conservative planar colour/gradient fill, Telea elsewhere.
No detector, residual scan, network, neural backend or implicit fallback to LaMa.
"""
import cv2
import numpy as np


def _plane(image, hole, forbidden):
    near = cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ring = (cv2.dilate(hole.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0) & ~near & ~forbidden
    # Also test ALL known interior pixels: robust fitting must not hide line art.
    ys, xs = np.where(hole)
    support = ring.copy()
    support[ys.min():ys.max()+1, xs.min():xs.max()+1] |= ~near[ys.min():ys.max()+1, xs.min():xs.max()+1] & ~forbidden[ys.min():ys.max()+1, xs.min():xs.max()+1]
    yy, xx = np.where(support)
    if len(yy) < 80:
        return None
    pick = np.linspace(0, len(yy)-1, min(2400, len(yy)), dtype=int)
    yy, xx = yy[pick], xx[pick]
    scale = max(image.shape[:2]); cx, cy = float(xs.mean()), float(ys.mean())
    def design(y, x):
        return np.column_stack((np.ones(len(y)), (x-cx)/scale, (y-cy)/scale)).astype(np.float32)
    a = design(yy, xx); values = image[yy, xx].astype(np.float32)
    co = np.linalg.lstsq(a, values, rcond=None)[0]
    errors = np.max(np.abs(a @ co-values), axis=1)
    # Reject structure/texture, including a narrow dark line in otherwise white.
    if np.percentile(errors, 95) > 3.5 or np.max(errors) > 14:
        return None
    prediction = design(ys, xs) @ co
    if prediction.min() < -3 or prediction.max() > 258:
        return None
    return ys, xs, np.clip(np.rint(prediction), 0, 255).astype(np.uint8)




def clean(image, mask, mode="adaptive", radius=3, report=None):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB uint8 image")
    if mask.shape != image.shape[:2]:
        raise ValueError("Mask/image shape mismatch")
    if mode not in {"adaptive", "telea", "ns"}:
        raise ValueError("Unknown CV2 mode")
    if not 1 <= radius <= 15:
        raise ValueError("CV2 radius must be between 1 and 15")
    binary = (mask > 0).astype(np.uint8)*255
    result=image.copy(); stats={"components":0,"plane":0,"telea":0,"ns":0}
    if not binary.any():
        if report is not None: report.update(stats)
        return result
    n,labels,boxes,_=cv2.connectedComponentsWithStats(binary,8)
    for label in range(1,n):
        x,y,w,h,area=map(int,boxes[label]); margin=max(24,min(96,max(w,h)//2))
        x0,y0=max(0,x-margin),max(0,y-margin)
        x1,y1=min(image.shape[1],x+w+margin),min(image.shape[0],y+h+margin)
        crop=np.ascontiguousarray(image[y0:y1,x0:x1]); all_mask=binary[y0:y1,x0:x1]
        hole=labels[y0:y1,x0:x1]==label
        stats["components"]+=1
        plane=_plane(crop,hole,all_mask>0) if mode=="adaptive" else None
        if plane is not None:
            yy,xx,values=plane; result[y0:y1,x0:x1][yy,xx]=values;stats["plane"]+=1
            continue
        kind="ns" if mode=="ns" else "telea"
        filled=cv2.inpaint(crop,all_mask,radius,cv2.INPAINT_NS if kind=="ns" else cv2.INPAINT_TELEA)
        result[y0:y1,x0:x1][hole]=filled[hole];stats[kind]+=1
    if report is not None: report.update(stats)
    return result
