# -*- coding: utf-8 -*-
import cv2
import numpy as np

def apply_rotation_metadata(frame, img_rgb: np.ndarray) -> np.ndarray:
    try:
        rot = None
        if hasattr(frame, 'metadata') and frame.metadata:
            r = frame.metadata.get('rotate')
            if r is not None:
                try:
                    rot = int(r)
                except:
                    pass
        if rot is None and getattr(frame, 'side_data', None):
            for sd in frame.side_data:
                t = getattr(sd, 'type', None)
                if t and getattr(t, 'name', '').upper() == 'DISPLAYMATRIX':
                    to = getattr(sd, 'to_dict', None)
                    if callable(to):
                        try:
                            d = to(); rr = d.get('rotation')
                            if rr is not None:
                                rot = int(round(rr)); break
                        except:
                            pass
        if rot is None:
            return img_rgb
        rot %= 360
        if rot == 90:
            return cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
        if rot == 180:
            return cv2.rotate(img_rgb, cv2.ROTATE_180)
        if rot == 270:
            return cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img_rgb
    except Exception:
        return img_rgb
