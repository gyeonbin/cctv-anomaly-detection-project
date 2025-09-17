# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np

from core.gpu_worker import FEAT_DIM
from app.wiring.bootstrap import MAX_MISSES, KEEP_RENDER_MISSES, EMB_EMA
from app.tracking.kalman_ds import KalmanDS

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size==0 or b.size==0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1,ay1,ax2,ay2 = a[:,0,None], a[:,1,None], a[:,2,None], a[:,3,None]
    bx1,by1,bx2,by2 = b[None,:,0], b[None,:,1], b[None,:,2], b[None,:,3]
    inter_w = np.maximum(0, np.minimum(ax2,bx2) - np.maximum(ax1,bx1))
    inter_h = np.maximum(0, np.minimum(ay2,by2) - np.maximum(ay1,by1))
    inter = inter_w*inter_h
    area_a=(ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by1)
    union=np.maximum(area_a+area_b-inter, 1e-6)
    return (inter/union).astype(np.float32)

class ReIDTracker:
    def __init__(self, feat_dim:int=FEAT_DIM, dt:float=1/30.0,
                 w_app:float=0.5, w_iou:float=0.3, w_mot:float=0.2,
                 gate_chi2_95:float=9.4877):
        self.next_id=1
        self.feat_dim=feat_dim
        self.dt = float(dt)
        self.w_app, self.w_iou, self.w_mot = float(w_app), float(w_iou), float(w_mot)
        self.gate = float(gate_chi2_95)

        self.tracks_feat: Dict[int,np.ndarray] = {}
        self.tracks_miss: Dict[int,int] = {}
        self.tracks_box:  Dict[int,np.ndarray] = {}   # filtered xyxy
        self.kf: Dict[int, KalmanDS] = {}

        self.time_since_update: Dict[int, int] = {}  # 마지막 업데이트 이후 지난 프레임 수
        self.visible: Dict[int, bool] = {}           # 이번 프레임에 검출로 갱신됐는가

    def _new_id(self)->int:
        tid=self.next_id; self.next_id+=1
        if self.next_id>1_000_000_000: self.next_id=1
        return tid

    @staticmethod
    def _cosine_sim(A:np.ndarray, B:np.ndarray) -> np.ndarray:
        if A.size==0 or B.size==0:
            return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
        A=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-8)
        B=B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-8)
        return A@B.T

    def _predict_all(self, tids:List[int]) -> np.ndarray:
        preds=[]
        for t in tids:
            preds.append(self.kf[t].predict())
        return np.stack(preds, axis=0) if preds else np.zeros((0,4), np.float32)

    def _assign(self, boxes:np.ndarray, feats:Optional[np.ndarray],
                tids:List[int], preds:np.ndarray) -> List[Optional[int]]:
        N = boxes.shape[0] if boxes is not None else 0
        M = len(tids)
        if N==0:
            for tid in list(self.tracks_miss.keys()):
                self.tracks_miss[tid] = self.tracks_miss.get(tid,0)+1
            return []

        # 1) appearance
        if feats is not None and feats.shape[0]==N and M>0:
            G = np.stack([self.tracks_feat[t] for t in tids], axis=0) if M>0 else np.zeros((0,self.feat_dim),np.float32)
            S_app = self._cosine_sim(feats, G)
        else:
            S_app = np.zeros((N,M), dtype=np.float32)

        # 2) IoU vs predicted boxes
        S_iou = _iou_xyxy(boxes, preds) if (N>0 and M>0) else np.zeros((N,M), np.float32)

        # 3) motion gating (Mahalanobis)
        S_mot = np.zeros((N,M), dtype=np.float32)
        if M>0 and N>0:
            for i in range(N):
                for j, tid in enumerate(tids):
                    dM = self.kf[tid].maha_xyxy(boxes[i])
                    if dM <= self.gate:
                        S_mot[i,j] = float(np.exp(-0.5*min(dM, 100.0)))
                    else:
                        S_mot[i,j] = 0.0  # hard gate

        # 4) combine
        S = self.w_app*S_app + self.w_iou*S_iou + self.w_mot*S_mot

        # greedy matching
        ids=[None]*N
        used_d, used_g = set(), set()
        cand = [(S[i,j],i,j) for i in range(N) for j in range(M)]
        cand.sort(reverse=True)
        for score,i,j in cand:
            if i in used_d or j in used_g: continue
            if score < 0.0: continue
            tid=tids[j]; ids[i]=tid
            used_d.add(i); used_g.add(j)

        # 신규 트랙
        for i in range(N):
            if ids[i] is None:
                tid=self._new_id()
                ids[i]=tid
                k = KalmanDS(dt=self.dt)
                k.init_from_xyxy(boxes[i])
                self.kf[tid] = k
                self.tracks_box[tid] = boxes[i].copy()
                if feats is not None and feats.shape[0]==N:
                    self.tracks_feat[tid]=feats[i].copy()
                else:
                    self.tracks_feat[tid]=np.zeros((self.feat_dim,),np.float32)
                self.tracks_miss[tid]=0

        # miss 증가
        matched=set([ids[i] for i in range(N) if ids[i] is not None])
        for tid in tids:
            if tid not in matched:
                self.tracks_miss[tid] = self.tracks_miss.get(tid,0)+1

        return ids

    def _prune(self):
        dead=[tid for tid,m in self.tracks_miss.items() if m>MAX_MISSES]
        for tid in dead:
            self.tracks_miss.pop(tid,None)
            self.tracks_feat.pop(tid,None)
            self.tracks_box.pop(tid,None)
            self.kf.pop(tid,None)

    def update(self, pts: float, boxes: np.ndarray, feats: Optional[np.ndarray]):
        N = 0 if boxes is None else boxes.shape[0]
        if N == 0:
            # 1) predict only
            for tid in list(self.kf.keys()):
                self.kf[tid].predict()
                self.tracks_miss[tid] = self.tracks_miss.get(tid, 0) + 1
                self.time_since_update[tid] = self.time_since_update.get(tid, 0) + 1
                self.visible[tid] = False
            # 2) prune
            self._prune()
            # 3) display predicted boxes for a short while
            disp_boxes = []
            disp_ids = []
            for tid in self.kf.keys():
                if self.time_since_update.get(tid, 0) <= KEEP_RENDER_MISSES:
                    disp_boxes.append(self.kf[tid].get_xyxy())
                    disp_ids.append(tid)
            if len(disp_boxes) == 0:
                return [], np.zeros((0, 4), np.float32)
            return disp_ids, np.stack(disp_boxes, axis=0).astype(np.float32)

        # normal path
        tids = list(self.kf.keys())
        preds = self._predict_all(tids)
        ids = self._assign(boxes, feats, tids, preds)

        for i, tid in enumerate(ids):
            if tid in self.kf:
                self.kf[tid].update_xyxy(boxes[i])
                self.tracks_box[tid] = self.kf[tid].get_xyxy()
                self.tracks_miss[tid] = 0
                self.time_since_update[tid] = 0
                self.visible[tid] = True
                if feats is not None and feats.shape[0] == N:
                    f_old = self.tracks_feat[tid]
                    f_new = feats[i]
                    f = EMB_EMA * f_old + (1 - EMB_EMA) * f_new
                    n = np.linalg.norm(f) + 1e-8
                    self.tracks_feat[tid] = (f / n).astype(np.float32)
            else:
                k = KalmanDS(dt=self.dt)
                k.init_from_xyxy(boxes[i])
                self.kf[tid] = k
                self.tracks_box[tid] = boxes[i].copy()
                self.tracks_miss[tid] = 0
                self.time_since_update[tid] = 0
                self.visible[tid] = True
                if feats is not None and feats.shape[0] == N:
                    self.tracks_feat[tid] = feats[i].copy()
                else:
                    self.tracks_feat[tid] = np.zeros((self.feat_dim,), np.float32)

        matched = set([ids[i] for i in range(N) if ids[i] is not None])
        for tid in tids:
            if tid not in matched:
                self.kf[tid].predict()
                self.tracks_miss[tid] = self.tracks_miss.get(tid, 0) + 1
                self.time_since_update[tid] = self.time_since_update.get(tid, 0) + 1
                self.visible[tid] = False

        self._prune()

        smoothed = np.zeros_like(boxes, dtype=np.float32)
        for i, tid in enumerate(ids):
            if tid in self.kf:
                smoothed[i, :] = self.kf[tid].get_xyxy()
            else:
                smoothed[i, :] = boxes[i]

        # add predicted boxes for non-updated tracks (short display)
        extra_boxes = []
        extra_ids = []
        for tid in self.kf.keys():
            if not self.visible.get(tid, False) and self.time_since_update.get(tid, 0) <= KEEP_RENDER_MISSES:
                extra_boxes.append(self.kf[tid].get_xyxy())
                extra_ids.append(tid)

        if len(extra_boxes) > 0:
            smoothed = np.vstack([smoothed, np.stack(extra_boxes, axis=0).astype(np.float32)])
            ids = list(ids) + extra_ids

        return ids, smoothed
