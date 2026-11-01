"""Typed first-pass geometry reuse. No detector calls, network, or mutable cache."""
import hashlib
from pathlib import Path
import numpy as np


def pixel_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def window_observations(result, top, height, strip_height, width):
    """None means legacy/untyped callback: never cache an incomplete set as empty."""
    if not isinstance(result,dict):
        return list(result), None
    if set(result) != {'bubbles','texts'}:
        raise ValueError('Typed detection needs bubbles and texts')
    raw=[];observations=[]
    for key,kind in [('bubbles','bubble'),('texts','text')]:
        for box in result[key]:
            if len(box)!=4 or any(not np.isfinite(v) for v in box):
                raise ValueError('Invalid first-pass detection box')
            x1,y1,x2,y2=map(int,box)
            if not 0<=x1<x2<=width or not 0<=y1<y2<=height:
                raise ValueError('First-pass box is outside its image')
            raw.append([x1,y1,x2,y2])
            # At natural chapter boundaries we only need the visible part.
            partial=(top>0 and y1<=2) or (top+height<strip_height and y2>=height-2)
            observations.append({'kind':kind,'box':[x1,y1+top,x2,y2+top],
                                 'window':top,'complete':not partial})
    return raw,observations


def merge_observations(observations):
    """Prefer complete views. Never combine two targets from the SAME window."""
    def area(box):return (box[2]-box[0])*(box[3]-box[1])
    groups=[]
    for obs in sorted(observations,key=lambda x:(not x['complete'],-area(x['box']),x['window'],x['box'])):
        a=obs['box'];match=None
        for group in groups:
            if group['kind']!=obs['kind'] or obs['window'] in group['windows']:continue
            b=group['box'];inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=area(a)+area(b)-inter
            if inter and (inter/union>=.65 or inter/min(area(a),area(b))>=.90):
                match=group;break
        if match is None:
            groups.append({'kind':obs['kind'],'box':list(a),'complete':obs['complete'],'windows':{obs['window']}})
        else:
            match['windows'].add(obs['window'])
            # Complete views sort first. A clipped view never shrinks/replaces them.
            if obs['complete'] and not match['complete']:
                match['box']=list(a);match['complete']=True
    return [{'kind':g['kind'],'box':g['box'],'complete':g['complete']}
            for g in sorted(groups,key=lambda g:(g['box'][1],g['box'][0],g['kind'],g['box'][3],g['box'][2]))]


def page_record(groups,top,bottom,width,digest):
    texts=[];bubbles=[];reasons=set()
    for group in groups:
        x1,y1,x2,y2=group['box']
        if y2<=top or y1>=bottom:continue
        if not group['complete']:reasons.add('clipped_scan_window')
        if y1<top or y2>bottom:reasons.add('region_crosses_cut')
        box=[x1,max(y1,top)-top,x2,min(y2,bottom)-top]
        (texts if group['kind']=='text' else bubbles).append(box)
    return {'version':1,'width':width,'height':bottom-top,'pixel_sha256':digest,
            'texts':texts,'bubbles':bubbles,'reusable':not reasons,'reason':','.join(sorted(reasons))}


def lookup(context,path,source,identity,crop_box=None):
    """Return ((text_boxes,bubble_boxes), reason). None asks for normal detection."""
    if not context:return None,'no_first_pass'
    try:
        if not identity or context['identity']!=identity:return None,'detector_changed'
        record=context['records'].get(str(Path(path).resolve()))
        if not record:return None,'page_not_in_plan'
        h,w=source.shape[:2]
        if source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2]!=3:return None,'invalid_image'
        if record['version']!=1 or (record['width'],record['height'])!=(w,h):return None,'image_size_changed'
        if record['pixel_sha256']!=pixel_hash(source):return None,'pixels_changed'
        if not record['reusable']:return None,record['reason'] or 'uncertain_first_pass'
        cx,cy,cx2,cy2=crop_box or (0,0,w,h)
        if not 0<=cx<cx2<=w or not 0<=cy<cy2<=h:return None,'invalid_crop'
        output=[]
        for key in ['texts','bubbles']:
            boxes=[]
            for values in record[key]:
                if len(values)!=4 or any(type(v)!=int for v in values):return None,'invalid_cached_box'
                x1,y1,x2,y2=values
                if not 0<=x1<x2<=w or not 0<=y1<y2<=h:return None,'invalid_cached_box'
                if x2<=cx or x1>=cx2 or y2<=cy or y1>=cy2:continue
                if key=='texts' and (x1<cx or y1<cy or x2>cx2 or y2>cy2):
                    return None,'crop_splits_text'
                boxes.append([max(x1,cx)-cx,max(y1,cy)-cy,min(x2,cx2)-cx,min(y2,cy2)-cy])
            output.append(np.asarray(boxes,dtype=np.int32).reshape(-1,4))
        return tuple(output),'reused'
    except (KeyError,TypeError,ValueError,IndexError):
        return None,'invalid_first_pass_record'
