"""Conservative first-mask refinement for outlined free text. No models or rescans."""
import os
import cv2
import numpy as np


def refine_free_text(image, mask, blocks):
    result = mask.copy(); changed = 0
    for block in blocks or []:
        if getattr(block, 'text_class', None) != 'text_free' or getattr(block, 'is_ad', False):
            continue
        # Never target untranslated source lettering.
        text = (getattr(block, 'translation', '') or getattr(block, 'text', '') or '').strip()
        if len(text) < 2:
            continue
        bx, by, ex, ey = map(int, block.xyxy)
        bx, by, ex, ey = max(0,bx), max(0,by), min(image.shape[1],ex), min(image.shape[0],ey)
        if ex-bx < 12 or ey-by < 8:
            continue
        crop = image[by:ey,bx:ex]; old = mask[by:ey,bx:ex]
        # Already tight segmentation is retained; this refines only broad masks.
        if (old > 0).mean() < .55:
            continue
        gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY);chroma=np.ptp(crop.astype(np.int16),axis=2)
        candidates=[]
        for light in (True,False):
            raw=(((crop.min(2)>185)&(chroma<55)) if light else ((gray<75)&(chroma<55))).astype(np.uint8)
            n,labels,stats,_=cv2.connectedComponentsWithStats(raw,8);seed=np.zeros_like(raw);count=0
            for label in range(1,n):
                x,y,w,h,area=map(int,stats[label])
                if not (2 <= area <= 2600 and w <= min(100,.9*(ex-bx)) and h <= min(65,.95*(ey-by))):
                    continue
                part=(labels==label).astype(np.uint8)
                ring=(cv2.dilate(part,np.ones((5,5),np.uint8))>0)&~(part>0)
                if not ring.any():continue
                support=float((gray[ring]<85).mean()) if light else float((gray[ring]>140).mean())
                if support < (.20 if light else .60):continue
                seed[part>0]=1;count+=1
            if count<3 or seed.sum()<10:continue
            ys,xs=np.where(seed)
            if np.ptp(xs)<.40*(ex-bx):continue
            fraction=float(seed.mean())
            if not .015 <= fraction <= .55:continue
            candidates.append((light,seed))
        if not candidates:continue
        # Strong white-letter/black-outline evidence wins over dark camouflage pixels.
        light,seed=candidates[0]
        # The initial segmenter is allowed to reach CLEAN_TEXT_PAD beyond the
        # OCR box. Once glyph evidence is strong, remove its stray art masks too.
        pad=max(8,int(os.environ.get('CLEAN_TEXT_PAD','50')))
        x0,y0=max(0,bx-pad),max(0,by-pad);x1,y1=min(image.shape[1],ex+pad),min(image.shape[0],ey+pad)
        local=np.zeros((y1-y0,x1-x0),np.uint8);local[by-y0:ey-y0,bx-x0:ex-x0]=seed
        radius=6 if light else 3
        local=cv2.dilate(local,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))*255
        previous=mask[y0:y1,x0:x1]
        # Cannot introduce new erased pixels; never remove the entire source box.
        local=cv2.bitwise_and(local,previous)
        # Neighbouring text targets, balloons and ads retain their original mask.
        for other in blocks or []:
            if other is block:continue
            ox,oy,ox1,oy1=map(int,other.xyxy)
            ax,ay=max(x0,ox-8),max(y0,oy-8);az,at=min(x1,ox1+8),min(y1,oy1+8)
            if az>ax and at>ay:
                local[ay-y0:at-y0,ax-x0:az-x0]=previous[ay-y0:at-y0,ax-x0:az-x0]
        before=int((previous>0).sum());after=int((local>0).sum())
        if not before or not .12*before <= after <= .9*before:continue
        # Reject source targets whose detected glyphs are not covered by the old mask.
        if (old[seed>0]>0).mean()<.98:continue
        result[y0:y1,x0:x1]=local;changed+=1
    return result,changed
