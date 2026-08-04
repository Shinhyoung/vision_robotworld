# 아키텍처 문서

컨베이어 비전 검사 & 6D 포즈 계산 시스템 — 설계 및 구조

---

## 1. 설계 원칙

claude.md §2 "계약 먼저, 그다음 병렬"을 구조로 강제한다.

### 1.1 핵심 결정: ROS 없는 코어 + 얇은 ROS 어댑터

```
roboworld_core/          ← 모든 로직. rclpy import 없음. GPU·카메라 없이 테스트 가능
   ↑
roboworld_ros_utils/     ← 메시지 ↔ numpy 변환, QoS, 파라미터 (유일한 ROS 경계)
   ↑
roboworld_inspection/    roboworld_pose/    roboworld_pipeline/    ← 노드 = 얇은 래퍼
```

**이유**: ROS·CUDA·카메라가 준비되기 전에도 4명이 병렬로 개발·테스트할 수 있어야 한다.
현재 이 저장소는 **ROS 미설치 환경에서 154개 유닛 테스트와 E2E dry-run이 모두 통과**한다.

부작용으로 얻는 것:
- 분기 로직·택트 예산·NG-포즈-생략 규칙이 단위 테스트로 검증된다 (`test_pipeline.py`).
- 동일한 상태 기계가 in-process(테스트)와 3-프로세스(운영)에서 **한 벌만** 존재한다.
  `ServiceInspectionBackend`/`ServicePoseBackend`가 코어 인터페이스를 ROS 서비스로
  구현하기 때문이다.

### 1.2 Inspection ↔ Pose 디커플

claude.md §2 요구사항. Inspection 노드는 Pose 노드를 **모른다**.
OK/NG 분기는 파이프라인 노드가 소유한다.

```
pipeline_node ──InspectPart(srv)──▶ inspection_node
      │
      ├─ NG  → 즉시 PartResult 발행 (포즈 요청 자체를 하지 않음)
      │
      └─ OK  ──EstimatePose(srv)──▶ pose_node
                                        │
                    ◀───────────────────┘
              PartResult 발행
```

---

## 2. 디렉토리 구조

```
RoboWorld_Demo/
├── claude.md                       # 원 요구사항
├── README.md
├── 01_input/                       # 부품 CAD (PLY, 3종)
├── docs/
│   ├── architecture.md             # 본 문서
│   ├── robot_interface_ICD.md      # ★ 로봇 부서 인터페이스 계약
│   ├── setup_wsl.md                # WSL2/CUDA/usbipd/colcon 구축 절차
│   ├── git_strategy.md             # 브랜치·PR·머지 게이트
│   └── agent_tickets.md            # 에이전트별 작업 티켓
├── ros2_ws/src/
│   ├── roboworld_interfaces/       # ★ msg/srv/action (Team Lead 단독 관리)
│   ├── roboworld_core/             # ROS 비의존 코어 + 테스트
│   ├── roboworld_ros_utils/        # ROS 어댑터 헬퍼
│   ├── roboworld_inspection/       # 검사 노드
│   ├── roboworld_pose/             # 포즈 노드
│   ├── roboworld_pipeline/         # 오케스트레이션·프레임 소스·목업 카메라·트리거
│   └── roboworld_bringup/          # launch + config(단일 진실 원천) + 메시 사본
├── tools/                          # ROS 없이 실행되는 개발 도구
│   ├── generate_mock_dataset.py
│   ├── train_inspection.py
│   ├── evaluate.py                 # 정확도 측정 (허용치 게이트)
│   ├── e2e_dryrun.py               # ★ 계약 검증 게이트
│   ├── visualize.py                # 결과 영상 패널 1장 (PNG)
│   ├── live_view.py                # 실시간 RGB-D 창 / MJPEG 스트림
│   ├── reconstruct_part.py         # D455 depth로 메시 복원 (CAD 불필요)
│   ├── capture_part.py             # D455로 새 부품 촬영 (CAD 불필요)
│   ├── register_part.py            # 촬영본으로 검사 모델 학습 + 부품 등록
│   ├── export_mock_images.py       # npz → PNG (anomalib 데이터셋)
│   └── check_symmetry.py
├── data/                           # 생성물 (git 제외)
└── .github/workflows/ci.yml
```

### 2.1 패키지 책임 (claude.md §3 역할 대응)

| 패키지 | 담당 에이전트 | 책임 |
|---|---|---|
| `roboworld_interfaces` | **Team Lead** | 메시지 계약. 변경 시 ICD 개정 필수 |
| `roboworld_core` | 공통 | 설정·기하·목업·백엔드·상태기계 |
| `roboworld_inspection` | **Inspection** | EfficientAD / statistical 백엔드, 검사 노드 |
| `roboworld_pose` | **Pose** | FoundationPose / ICP 백엔드, 포즈 노드, TF 변환 |
| `roboworld_pipeline`, `roboworld_ros_utils` | **ROS2** | 트리거·동기화·프레임 소스·최종 발행 |
| `roboworld_bringup` | **Team Lead** | launch, config, 메시 배포 |

---

## 3. 데이터 흐름

### 3.1 정상 사이클 (양품)

```
   PLC 포토센서            pipeline_node                inspection_node        pose_node
        │                        │                            │                    │
        │  std_msgs/Bool ↑edge   │                            │                    │
        ├───────────────────────▶│                            │                    │
        │                        │ ① capture                  │                    │
        │                        │   (FrameSource)            │                    │
        │                        │                            │                    │
        │                        │ ② InspectPart ────────────▶│                    │
        │                        │◀─────────── is_good=true ──┤                    │
        │                        │                            │                    │
        │                        │ ③ EstimatePose ───────────────────────────────▶│
        │                        │◀──────────────────── PoseStamped (m, quat) ────┤
        │                        │                            │                    │
        │                        │ ④ PartResult(STATUS_OK) ──▶ /roboworld/part_result
```

### 3.2 불량 사이클

②에서 `is_good=false` → ③ **생략** → `STATUS_NG` 즉시 발행.
포즈 계산 비용(≈110 ms, GPU 사용 시 더 큼)을 불량품에 쓰지 않는다.

### 3.3 프레임 소스 3종 (`camera.source`)

| 값 | 경로 | 용도 |
|---|---|---|
| `realsense` | realsense-ros 토픽 구독 | 실 운전 |
| `rosbag` | 동일 구독 경로, `ros2 bag play` | **하드웨어 없는 개발 (권장)** |
| `mock` | CAD 렌더링 in-process | ROS 트래픽 없는 최속 dry-run |

세 경로 모두 동일한 `Frame` 객체를 산출하므로 하위 로직은 구분하지 않는다.

---

## 4. TF 좌표계

```
world ─(A)─ station_base ─(B)─ camera_link ─(C)─ camera_color_optical_frame ─(D)─ part
```

| 구간 | 제공 | 상태 |
|---|---|---|
| C | realsense-ros 자동 | 확정 |
| D | 포즈 추정 결과 | 확정 |
| A, B | 로봇 부서 합의 필요 | **미결** — ICD §4 참조 |

**기본 발행 프레임**: `camera_color_optical_frame` (OpenCV 규약: +x 오른쪽, +y 아래, +z 전방).
`world` 발행을 원하면 `pose.transform_to_output_frame: true` 설정 + TF 체인 필요.
TF가 없으면 카메라 프레임으로 fallback 하고 `message`에 기록한다 — **조용히 틀린 프레임으로
발행하지 않는다**.

**부품 모델 프레임**: CAD AABB 중심 원점, +x = 장축(200 mm). 발행 위치는 블록의 기하 중심.

---

## 5. 택트 타임 예산

`config/pipeline.yaml`의 `tact_budget_ms`. 초과 시 경고 로그, 하드 리밋 초과 시 `STATUS_ERROR`.

| 단계 | 예산 | CPU 모의 실측 | 비고 |
|---|---|---|---|
| capture | 120 ms | (모의 렌더 ~150 ms) | 실 카메라는 프레임 대기만 |
| inspect | 400 ms | 62 ms | EfficientAD(GPU)는 별도 측정 필요 |
| pose | 900 ms | ~110 ms | FoundationPose(GPU)는 별도 측정 필요 |
| **total (warn)** | 1200 ms | **258 ms (중앙값)** | |
| total (limit) | 5000 ms | | 초과 시 ERROR |

---

## 6. 검사 백엔드

| 백엔드 | 용도 | 요구사항 |
|---|---|---|
| `efficientad` | **운영 목표** (claude.md §1 고정) | anomalib + torch + CUDA + 체크포인트 |
| `statistical` | CPU 참조/폴백, CI | numpy만 |
| `stub` | dry-run, 인터페이스 테스트 | 없음 |

### 6.1 `statistical` 동작 원리

위치 비의존 PaDiM 변형. 부품이 임의의 yaw로 도착하므로 **위치별** 가우시안이 아니라
**전역** 가우시안 하나를 사용한다.

1. 격자 패치(16 px, stride 8)로 분할
2. 패치당 7차원 특징: `[mean_r, mean_g, mean_b, std_gray, mean_grad, max_grad, mean_gray-min_gray]`
3. 정상 부품 표면 패치 전체에 다변량 가우시안 1개 적합
4. 마할라노비스 거리 → 이상 점수

**조명 정규화**: 특징 추출 전 부품 영역의 median 밝기를 128로 맞춘다.
이것이 없으면 주변광 변화·노출 자동조정·부품이 화면 가장자리로 이동해 램버시안 항이
떨어지는 것까지 전부 "불량"으로 읽힌다.

**점수 정규화**: `score = clip(0.5 · raw / (k · high), 0, 1)`.
`high`는 학습 점수의 95 백분위, `k = 1.6`.
초기 구현은 학습 분포의 **폭**(high − low)으로 나눴는데, 이 경우 학습 데이터를 늘리면
폭이 넓어져 실제 불량 점수까지 임계값 아래로 내려갔다. 폭이 아니라 **수준**에 고정하도록
변경했다.

### 6.2 판정 규칙 (모든 백엔드 공통, `InspectionBackend.decide`)

- ROI(부품 영역) 밖 픽셀 무시
- 최대값의 55 % 이상인 연결 영역만 후보
- `min_defect_area_px = 12` 미만 영역은 노이즈로 간주하고 기각
  → 단일 핫 픽셀이 양품을 기각하지 못한다
- **ROI가 비면 점수 1.0 (NG)** — 부품 미검출은 fail-safe

---

## 7. 포즈 백엔드

| 백엔드 | 용도 | 요구사항 |
|---|---|---|
| `foundationpose` | **운영 목표** (claude.md §1 고정) | Isaac ROS + CUDA. **공개판 비상업 라이선스** |
| `icp` | CPU 참조/폴백, CI | numpy (Open3D 있으면 사용) |
| `stub` | dry-run | 없음 |

### 7.1 `icp` 동작 원리

1. **분할**: RANSAC으로 컨베이어 평면 추정 → 평면 위 높이 밴드 → 최대 연결 성분
2. **초기값**: 이산 8가지 가설 — 블록이 놓일 수 있는 4개 면 × 장축 방향 2가지
   - 임의 yaw 스윕이 아니다. 200 mm 블록은 세워지지 않으므로 ±x 면은 제외
   - 각 가설은 **모델 자신의 카메라 대향 표면**을 관측 표면에 맞춰 배치한다
     (extent 절반 가정이 아님 → End Stopper 같은 계단형 부품에서도 정확)
3. **정합**: scene→model 방향 point-to-point ICP
   - depth는 한쪽 면만 보므로, 모든 scene 점은 대응 model 점을 갖지만 그 역은 아니다.
     반대 방향으로 매칭하면 보이지 않는 뒷면으로 끌려간다
4. **2단계 탐색**: 8개 가설을 저해상도로 12회 반복 평가 → 승자만 전해상도 정밀화
   (전부 전해상도로 돌리면 정확도 이득 없이 8배 비용)

### 7.2 수용 게이트 (`PoseBackend.validate`)

모든 백엔드에 동일 적용. 하나라도 실패하면 `valid=false` → `STATUS_NO_POSE`.

| 조건 | 기본값 |
|---|---|
| fitness | ≥ 0.35 |
| RMSE | ≤ 6 mm |
| z 범위 | 0.30 ~ 1.00 m |
| 측방 오프셋 | ≤ 0.25 m |

### 7.3 대칭 모호성 ⚠️

세 부품 모두 55×55 정사각 단면이다. 피처가 대칭을 깨는 정도가 ICP inlier 허용치(6 mm)보다
작으면 단일 시점 depth로는 구분할 수 없다.

| 부품 | 자체 회전 chamfer | 구분 가능성 | 실측 뒤집힘 빈도 |
|---|---|---|---|
| guide_block | 3.2 ~ 4.7 mm | 불가 | 21/25 (84 %) |
| spacer_block | 3.2 ~ 3.9 mm | 불가 | 17/25 (68 %) |
| end_stopper | 6.8 ~ 12.0 mm | **가능** | 3/25 (12 %) |

**두 종류의 문제가 섞여 있으므로 구분해서 다뤄야 한다.**

- guide_block / spacer_block은 **원리적 한계**다. 알고리즘 개선으로 해결되지 않는다.
- end_stopper는 형상만으로 구분 가능한데도 12 % 뒤집힌다 → **ICP의 개선 여지**
  (피처 기반 판별, 티켓 POSE-6).

그래서 평가 지표는 항상 **원시 오차·대칭 보정 오차·뒤집힘 빈도 3가지를 함께** 보고한다.
보정값만 보면 실제 실패를 숨기고, 원시값만 보면 기하학적으로 동일한 해를 180° 실패로
계산한다. 뒤집힘 빈도가 로봇 부서에 실제로 필요한 숫자다.

로봇 부서 확인 필요 사항은 ICD §6.1 참조.

---

## 8. 측정 결과

`python3 tools/evaluate.py --part all` (모의 데이터, CPU 백엔드)

| 부품 | 미검출 | 과검출 | 위치오차(중앙/최대) | 자세오차(중앙/최대) |
|---|---|---|---|---|
| guide_block | 0/25 | 1/25 | 1.36 / 3.69 mm | 0.72 / 1.78° |
| spacer_block | 0/25 | 1/25 | 0.95 / 2.52 mm | 0.58 / 2.29° |
| end_stopper | 0/25 | 1/25 | 0.93 / 2.40 mm | 0.52 / 1.83° |

게이트: 미검출 0 % (불량 유출 불가), 과검출 ≤ 5 %, 위치 ≤ 5 mm, 자세 ≤ 5°.

**과검출 3건은 모두 배치 범위 극단(±35 mm, ±180° yaw, ±4.7° 기울기)의 프레임이다.**
실제 정지 스테이션의 기계식 스토퍼 반복도는 이보다 훨씬 좁으므로 실측 시 개선이 예상되나,
모의 조건을 완화해 수치를 좋게 만들지 않았다.

### 8.1 미검출 0건보다 margin을 볼 것 ⚠️

`tools/evaluate.py`는 **margin**(양품 최고점 − 불량 최저점)을 함께 보고한다.
집계상 미검출 0건이어도 margin이 없으면 실 데이터에서 곧바로 깨진다.

| 부품 | margin |
|---|---|
| guide_block | +0.047 |
| end_stopper | **+0.002** |
| spacer_block | **−0.162** (두 분포가 겹침) |

end_stopper의 얕은 스크래치 1건은 임계값 0.500에 대해 학습량에 따라 0.492 ↔ 0.526으로
진동한다. **결함 위치는 정확히 특정되지만 점수만 경계에 걸린다** — 특징 민감도의 한계이며
학습량으로 해결되지 않는다. 재현 절차와 대응 후보는 `docs/agent_tickets.md` INS-7 참조.

이것이 `statistical`을 CPU **참조** 백엔드로만 두고 EfficientAD를 운영 목표로 유지하는
이유다.

---

## 9. 검증 계층

| 계층 | 도구 | 검증 대상 |
|---|---|---|
| 유닛 | `pytest ros2_ws/src/roboworld_core/test` (146개) | 기하·설정·메시·백엔드·상태기계 |
| **계약** | `test_contract.py` + `e2e_dryrun.py` | `.msg` ↔ 데이터클래스 정합, 발행값 규칙 |
| 정확도 | `tools/evaluate.py` | 검출률·포즈 오차 (허용치 게이트) |
| 통합 | `colcon test` | ROS 빌드·린트 |

**계약 검증과 정확도 검증은 분리되어 있다.** dry-run은 인터페이스와 배관을 지키는 게이트이며,
소수 사이클로 검출 정확도를 판정하지 않는다. 섞으면 CI가 인터페이스와 무관한 이유로
깨진다.

---

## 10. 미결 사항

1. **ICD §4 좌표계 합의** — hand-eye 캘리브레이션 책임 주체 (최우선)
2. **ICD §6.1 대칭** — grasp이 order-8 대칭에 불변인지 로봇 부서 확인
3. EfficientAD 실학습 (정상 이미지 수집 필요) — 티켓 INS-3
4. Isaac ROS FoundationPose 브릿지 구현 — 티켓 POSE-4
5. D455 실물 연결 및 rosbag 녹화 — 티켓 ROS-5
6. 요구 택트 타임·정확도 목표치 확정 (현재는 비전팀 가정값)
