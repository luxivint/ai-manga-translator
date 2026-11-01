"""Shared PC/container cleaner contract. Called before any queue job is claimed."""
import os


def configure_cleaner():
    engine=os.environ.get('CLEAN_ENGINE','cv2').strip().lower()
    if engine!='cv2':
        raise ValueError('Only CLEAN_ENGINE=cv2 is supported; remove the obsolete cleaner setting')
    mode=os.environ.get('CLEAN_CV2_MODE','adaptive').strip().lower()
    if mode not in ('adaptive','telea','ns'):
        raise ValueError('CLEAN_CV2_MODE must be adaptive, telea or ns')
    radius=int(os.environ.get('CLEAN_CV2_RADIUS','3'))
    threads=int(os.environ.get('OPENCV_NUM_THREADS','2'))
    if not 1<=radius<=15 or not 1<=threads<=16:
        raise ValueError('Invalid CV2 radius (1..15) or thread count (1..16)')
    os.environ.update(CLEAN_ENGINE='cv2',CLEAN_POSTPROCESS='false')
    for k,v in {'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2'}.items():os.environ.setdefault(k,v)
    import cv2
    cv2.setNumThreads(threads)
    import sys
    torch_module=sys.modules.get('torch')
    if torch_module is not None:
        torch_module.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    return {'engine':'cv2','mode':mode,'radius':radius,'threads':threads,'postprocess':False}
