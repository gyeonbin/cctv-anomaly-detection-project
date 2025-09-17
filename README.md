# CCTV Anomaly Detection Project

본 프로젝트는 **4K @ 30fps 영상 스트림을 안정적으로 처리**하면서 이상행동을 감지하는 실시간 영상처리 파이프라인입니다.  
대규모 CCTV 환경에서 안정적·지속적인 처리를 목표로 설계되었습니다.

---

## 주요 특징 (Features)

- **4K @ 30fps 안정적 처리 성능** 검증  
- **GPU VRAM RingBuffer** 구조로 장시간 안정적 버퍼링  
- **Telemetry 로깅 시스템**: 성능 지표(렌더링 지연, dropped frames 등)를 자동 기록
- **Kalman Filter 기반 Tracking**: 관측값이 없는 구간에서도 박스 위치를 예측하도록 해보았습니다.

---

## 시스템 구조 (Architecture)

```
Source (File)
   ↓
Decoder (PyAV / NVDEC)
   ↓
VRAM RingBuffer + MetaQueue
   ↓
Detector / Kalman Filter Tracker
   ↓
Renderer + Telemetry Logger
```

- **MetaQueue** : 프레임과 연동된 메타데이터(타임스탬프, 디텍션 결과)를 전달  
- **VRAM RingBuffer** : 메모리 재할당/복사 없이 장시간 안정적으로 프레임 순환  
- **Telemetry** : 실행 중 성능 지표를 수집하고, 종료 시 CSV로 일괄 저장

---

## 실행 방법 (Usage)

```bash
# 파일 입력
python -m app.main "video.mp4"
```

---

## 성능 검증 (Performance)

- **평균 처리 지연**: 33ms 이하 (4K @ 30fps 실시간 처리 조건 충족)   
- **Telemetry.csv**를 통해 실험 환경별 성능 로그를 확보

---

## 연구 확장 가능성 (Future Work)

- **멀티 스트림 처리 확장**
- **ReID 및 Tracking 모듈 강화**
- **실시간 이상행동 탐지 알고리즘 고도화**

---
