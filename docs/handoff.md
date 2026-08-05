# 세션 인수인계 (새 대화 시작용)

이 문서 하나로 이어서 작업할 수 있도록 정리한 브리핑이다.
상세는 [project_summary.md](project_summary.md) · [robot_interface_ICD.md](robot_interface_ICD.md) ·
[architecture.md](architecture.md) 참조.

---

## 1. 프로젝트 한 줄 요약

컨베이어 정지 스테이션에서 MC 나일론 블록 3종을 **EfficientAD로 불량 검사**하고,
양품이면 **FoundationPose로 6D 포즈**를 계산해 **ROS 2로 로봇 부서에 전달**한다.
로봇 제어·grasp·pick&place는 로봇 부서 범위(우리 범위 밖).

원 요구사항은 저장소 루트의 `claude.md`.

---

## 2. 확정된 전체 프로세스 (2차 협의 반영)

```
컨베이어 ──▶ [벨트 센서가 감지해 정지]   ← 정지 제어는 컨베이어 측 (우리 범위 밖)
                    │
                    ▼   카메라 + 로봇팔 위치
              ① EfficientAD 불량 검사
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   불량 (STATUS_NG)         양품 (STATUS_OK)
        │                       │
   로봇 부서가 벨트 재이동    ② FoundationPose 포즈 계산
   (끝에서 배출)                │
                                ▼
                     로봇 pick & place → 벨트 재이동
```

**확정 사항**
- 정지 스테이션(index) 방식 — 이동 중 실시간 검사 아님
- 벨트 정지는 **컨베이어 센서**가 처리. 우리는 정지 후 트리거만 받음
- 벨트 재이동 명령은 **로봇 부서가 `part_result`의 `status`를 보고 자체 판단**
  → 새 메시지 불필요, 기존 계약으로 충분
- 불량품은 벨트 끝에서 배출
- 스테이션 부품 간 최소 간격 **500 mm**

---

## 3. 지금 바로 할 작업 (중단된 지점)

### 스테이션 ROI — ✅ 구현 완료 (2026-08-05)

간격 200 mm에서 **후속 부품의 포즈(y=+0.195 m)를 발행**하던 문제를 막는 게이트다.
후보(`Candidate`)의 **중심이 카메라 좌표계 기준 상자 안**에 있어야 선택 대상이 되고,
상자 밖 부품만 있으면 선택을 거부한다.
⚠️ 다만 이때 발행되는 status는 `STATUS_NO_POSE`가 아니라 `STATUS_NG`다 — **§10 참조**.

| 파일 | 내용 |
|---|---|
| `roboworld_core/segmentation.py` | `StationRoi`, `station_roi_from_config`, `segment_part(station_roi=...)` |
| `roboworld_core/pose/__init__.py`·`inspection/__init__.py` | **양쪽 팩토리**에 배선 (§8 함정) |
| `roboworld_bringup/config/pose.yaml` | `pose.segmentation.station_roi` |
| `roboworld_core/viz.py` | `draw_station_roi` — 상자 와이어프레임 투영 |
| `tools/live_view.py` | 분할 패널에 ROI 표시 + `OUTSIDE STATION` HUD |
| `test/test_station_roi.py` | 11개 테스트 (200 mm 회귀 포함) |

```yaml
station_roi:
  enabled: true
  center_m: [0.0, 0.0, 0.60]
  half_extents_m: [0.15, 0.15, 0.12]
```

**설계 근거 (과장 없이)**
- 물리 단위(m)로 지정 → 컨베이어 도면 값을 그대로 사용, 해상도·렌즈 변경에 불변
- ❗ "3D가 카메라 이동에 견고하다"는 **사실이 아님** — 카메라 좌표계라 같이 깨진다.
  진짜 불변은 station 프레임 + TF가 필요한데 ICD §4 미결 사항

**구현하며 내린 판단**
- 후보 중심은 **관측된 점군의 centroid**. depth는 카메라 쪽 면만 보므로 실제 중심보다
  약 30 mm 앞. ROI가 ±100 mm대라 문제없지만 **위치 추정값이 아니다**
- 정렬은 2단(`(not in_roi, 기존 기준)`). ROI 밖 후보도 `candidates`에 남겨
  "부품이 200 mm 밖에 있다"를 뷰어와 `message`에서 볼 수 있게 함
- ROI 게이트가 치수 게이트보다 **먼저**. 사유가 구분되어야 조치가 갈린다
  (도착 대기 vs 부품 아님)
- `capture_part.py`·`reconstruct_part.py`는 ROI **OFF** — 등록·복원은 스테이션이
  아닌 곳에서 촬영한다 (`identify_by_size`를 끄는 것과 같은 이유)

**검사용 ROI와 혼동 금지**: EfficientAD가 보는 픽셀 영역은 depth 분할
마스크(부품 실루엣)이며 이 상자와 무관하다. 건드리지 말 것.

**미확인 사항**: 벨트 진행 방향이 카메라 화면의 가로(x)인지 세로(y)인지.
알면 ROI를 비대칭으로(진행 방향 좁게) 잡을 수 있다 → ICD §8 체크리스트에 추가함.

---

## 4. 현재 상태

### 커밋

- `main` 최신 = 스테이션 ROI 커밋 — **아직 푸시 안 됨**
- 원격 https://github.com/Shinhyoung/vision_robotworld 는 `55f2f87`까지 반영됨

### 검증

`make check PYTHON=/usr/bin/python3.10` — 린트 · **182개 테스트** · 계약 · 정확도 전부 통과

### 측정 결과 (모의 데이터, CPU 백엔드)

| 항목 | 결과 |
|---|---|
| 검사 | 미검출 0/25, 과검출 1/25 (3부품 공통) |
| 포즈 | 위치오차 0.9~1.4 mm, 자세오차 0.5~0.7° (0.6 m 거리) |
| 택트 | 중앙값 258 ms (예산 1200 ms) |
| 작업 거리 | 0.8 m 이내 권장. 1.25 m부터 급격히 열화 |

---

## 5. 환경 (새 세션에서 반드시 알아야 할 것)

| 항목 | 값 |
|---|---|
| 작업 경로 | `/home/shinhyoung/RoboWorld_Demo` |
| **파이썬** | **`/usr/bin/python3.10`** — 기본 `python3`는 conda라 numpy조차 없음 |
| 실행 예 | `make check PYTHON=/usr/bin/python3.10` |
| 테스트 | `PYTHONPATH=ros2_ws/src/roboworld_core /usr/bin/python3.10 -m pytest ros2_ws/src/roboworld_core/test -q` |
| ROS 2 | **미설치**. 모든 검증은 ROS 없이 수행 |

### 설치된 패키지

| 패키지 | 상태 |
|---|---|
| numpy, PyYAML, pytest, ruff | ✅ |
| opencv-python (창), pyrealsense2 (D455) | ✅ |
| **torch, anomalib, open3d** | ❌ 미설치 |

### D455 카메라

- Windows에 연결되어 있고 usbipd로 WSL에 attach하면 사용 가능
- **현재 미연결 상태** — 필요 시:
  ```bash
  "/mnt/c/Program Files/usbipd-win/usbipd.exe" attach --wsl --busid 8-1
  ```
- 실측: S/N 324422301079, 34 fps, 내부 파라미터 fx≈387 (config 추정치 384와 거의 일치)
- WSLg 사용 가능 (`DISPLAY=:0`) — GUI 창이 뜬다

---

## 6. 아키텍처 핵심 (이것만은 이해하고 시작할 것)

### ROS 비의존 코어

```
roboworld_core/       ← 모든 로직. rclpy import 없음
   ↑
roboworld_ros_utils/  ← 메시지 변환 (유일한 ROS 경계)
   ↑
inspection / pose / pipeline 노드   ← 얇은 어댑터
```

덕분에 ROS·GPU·카메라 없이 182개 테스트와 E2E dry-run이 전부 돈다.
**이 구조를 깨지 말 것.**

### 이중 백엔드

| 기능 | 운영 목표 | CPU 참조 | 현재 사용 중 |
|---|---|---|---|
| 검사 | EfficientAD | `statistical` | **`statistical`** |
| 포즈 | FoundationPose | `icp` | **`icp`** |

**EfficientAD·FoundationPose는 아직 적용 안 됨** (torch/anomalib 미설치, 체크포인트 없음).
지금까지의 모든 수치는 CPU 참조 구현 기준.

### 검증 계층 — 섞지 말 것

- `tools/e2e_dryrun.py` = **계약 게이트** (인터페이스·배관). 소수 사이클로 정확도 판정 안 함
- `tools/evaluate.py` = **정확도 게이트** (검출률·포즈 오차, 허용치)

---

## 7. ⚠️ 로봇 부서 미결 사항 (최우선)

### 7.1 자세 대칭 모호성 — 설계 변경까지 갈 수 있음

세 부품 모두 55×55 정사각 단면이라 단일 시점 depth로는 자세가 **order-8 대칭군까지만**
결정된다. 위치는 영향 없음.

| 부품 | 실측 뒤집힘 빈도 | 형상 구분 |
|---|---|---|
| guide_block | **84 %** | 불가 |
| spacer_block | **68 %** | 불가 |
| end_stopper | 12 % | 가능 |

**질문: 파지(grasp)가 이 대칭에 불변인가?**
- 장축 중앙을 위에서 잡음 → 불변, 문제 없음
- 특정 피처 기준으로 잡음 → **최대 120 mm 오차**. guide_block/spacer_block은
  알고리즘으로 해결 불가 → 비대칭 마커(설계 변경) 또는 카메라 추가 필요

### 7.2 기타

| 항목 | 내용 |
|---|---|
| 좌표계 | `world` 프레임 정의, hand-eye 캘리브레이션 책임 주체 |
| 시스템 구성 | PC 1대 / 2대 선택 → 전달물이 달라짐 (ICD §10) |
| 설치 거리 | 권장 0.6 m, 상한 0.8 m |
| 벨트 진행 방향 | ROI를 진행 방향으로 좁히려면 필요 (ICD §6.2.1) |
| 정지 위치 실측 좌표 | `station_roi.center_m` 확정용. 현재는 설계값 [0, 0, 0.60] |

---

## 8. 이미 겪은 함정 (반복하지 말 것)

| 함정 | 교훈 |
|---|---|
| 포즈 초기값을 카메라 쪽으로 배치 | 위치오차가 정확히 블록 두께(55 mm)만큼 발생. depth는 근접면만 보므로 중심은 관측면 **뒤** |
| 점수 정규화를 학습 분포 **폭**으로 | 학습 데이터를 늘리면 폭이 넓어져 불량이 임계값 아래로. **수준**에 고정할 것 |
| 검사 특징에 절대 밝기 사용 | 조명·노출·부품 위치 변화가 전부 "불량"이 됨 → median 밝기 정규화 필요 |
| 식별 로직을 `segment_from_config`에만 배선 | 백엔드는 `segment_part`를 직접 호출해 우회함. 양쪽 팩토리에 배선 필요 |
| 복원 중 크기 식별 활성화 | 치수를 정의하는 중인데 치수로 거르면 순환 논리. 복원·촬영 도구는 식별 OFF |
| `--level`이 보정항 오프셋까지 제거 | 부품 높이가 53.6 → 37.6 mm로 축소. 보정은 중심화해서 적용 |
| 창 닫힘을 `getWindowProperty < 1`로만 판정 | Qt는 예외를 던짐. 예외도 "닫힘"으로 처리 |
| 로컬 등록 부품이 출하 부품 허용치로 평가 | `locally_registered` 마커로 `--part all`에서 제외 |
| 치수 식별이면 다중 부품도 안전하다고 가정 | 뒤 부품도 치수가 **똑같이** 맞는다. 위치 게이트(스테이션 ROI)가 별도로 필요 |
| 문서의 status 값을 실행으로 확인 안 함 | ICD가 `NO_POSE`라 적힌 3개 경우가 실제로는 전부 `NG`였다. **파이프라인을 돌려봐야 안다** |

---

## 9. 남은 작업 (우선순위)

1. ~~스테이션 ROI 구현~~ ✅ 완료 (§3)
2. 스테이션 ROI 커밋·푸시
3. ⚠️ **부품 미검출 시 `STATUS_NG` → `STATUS_NO_POSE` 수정** (§10)
4. 로봇 부서 회신 확보 (§7) — 특히 대칭 불변성, 벨트 진행 방향
5. **GPU 환경 구축** — EfficientAD·FoundationPose 공통 선행 조건.
   RTX 5070은 Blackwell(sm_120)이라 지원 PyTorch 빌드 확인이 첫 관문
6. EfficientAD 실학습 (INS-3) / Isaac ROS FoundationPose 브릿지 (POSE-4)
7. `TopicFrameSource`(실 카메라 ROS 구독 경로) 실측 검증 — **아직 미검증**
8. 실 카메라로 ROI 확인 — D455를 attach하고
   `tools/live_view.py --inspect`로 상자가 정지 위치에 맞는지 눈으로 볼 것

---

## 10. ⚠️ 실행으로 발견한 미해결 결함 — 부품 미검출이 `STATUS_NG`가 된다

**증상**: 분할이 부품을 못 찾으면 `STATUS_NO_POSE`가 아니라 **`STATUS_NG`**가 발행된다.
ICD §3.1 권장 동작대로면 로봇이 **존재하지 않는 부품을 불량 배출**한다.

**실측** (파이프라인 직접 실행, `statistical` + `icp`):

| 경우 | `seg.ok` | 실제 status | ICD 표기(수정 전) |
|---|---|---|---|
| 벨트에 아무것도 없음 | False | **NG** (score 1.000) | NO_POSE |
| 치수 안 맞는 물체만 있음 | False | **NG** (score 1.000) | NO_POSE |
| 부품이 스테이션 밖에만 있음 | False | **NG** (score 1.000) | NO_POSE |

**원인**: 검사가 포즈보다 먼저 돈다. `inspection/base.py`의 `decide()`가 ROI가 비면
score 1.0을 반환하고(“조용히 OK가 되는 것보다 낫다”는 의도), 파이프라인은 그것을
불량으로 보고 포즈 단계를 건너뛴다.

```python
# roboworld_core/inspection/base.py
if not roi.any():
    return 1.0, np.zeros(anomaly_map.shape, dtype=bool)
```

**스테이션 ROI 이전부터 있던 문제다.** ROI는 같은 경로에 경우를 하나 더 추가했을 뿐.

**수정 방향**: `InspectionResult`에 "부품을 찾았는가"를 실어 파이프라인이
불량과 미검출을 구분하게 한다. `PartResult` 메시지 규격(ICD §3)은 바뀌지 않는다 —
`status`와 `message`는 이미 있다.

**결정 필요**: `is_good` 값을 무엇으로 할지. "부품 없음"은 양품도 불량도 아니다.
로봇 부서 회신 항목으로 ICD 체크리스트에 넣어 두었다.

---

## 11. 새 세션 시작 시 붙여넣을 문장

> `/home/shinhyoung/RoboWorld_Demo` 프로젝트를 이어서 작업한다.
> `docs/handoff.md`를 먼저 읽고 현재 상태를 파악할 것.
> 파이썬은 `/usr/bin/python3.10`을 쓴다(기본 python3는 conda라 numpy 없음).
> 다음 작업은 handoff.md §9의 우선순위 목록을 따른다.
