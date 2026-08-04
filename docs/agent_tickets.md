# 에이전트 작업 티켓

claude.md §6.2 — Team Lead 산출물이 `main`에 올라간 이후 3개 에이전트가 병렬 착수한다.
**모든 티켓은 목업 기반으로 시작할 수 있다.** 하드웨어·GPU 대기는 착수 조건이 아니다.

각 티켓은 완료 시 §5의 머지 게이트를 통과해야 한다.

---

## 완료됨 — Team Lead (claude.md §6.1)

| ID | 내용 | 산출물 |
|---|---|---|
| TL-1 | 아키텍처 설계 | `docs/architecture.md` |
| TL-2 | 로봇 부서 ICD | `docs/robot_interface_ICD.md` |
| TL-3 | 커스텀 msg/srv/action | `roboworld_interfaces` (3 msg, 3 srv, 1 action) |
| TL-4 | WSL2 구축 절차 | `docs/setup_wsl.md` |
| TL-5 | 목업 데이터 + 스텁 노드 | `mock_data.py`, `render.py`, `stub` 백엔드 ×2 |
| TL-6 | colcon 워크스페이스 스켈레톤 | 7개 패키지 |
| TL-7 | CI (빌드/린트/E2E dry-run) | `.github/workflows/ci.yml` |
| TL-8 | git 전략 · 티켓 | `docs/git_strategy.md`, 본 문서 |

**DoD 확인**: 스텁만으로 E2E dry-run 통과 (`python3 tools/e2e_dryrun.py`), 유닛 테스트 146개 통과.

### Team Lead 잔여

| ID | 우선순위 | 내용 |
|---|---|---|
| **TL-9** | **최상** | 로봇 부서와 ICD §4 좌표계·hand-eye 캘리브레이션 책임 확정 |
| **TL-10** | **상** | 로봇 부서에 ICD §6.1 대칭 모호성 통지 및 grasp 불변성 확인 |
| TL-11 | 중 | 요구 택트 타임·정확도 목표치 확정 (현재 예산은 비전팀 가정값) |
| TL-12 | 중 | 실 하드웨어 도입 후 통합 및 ICD §7 수치 갱신 |

---

## Inspection Agent (`feat/inspection`)

**DoD (claude.md §3.2)**: 목업 입력에 대해 OK/NG + score + mask를 계약대로 발행하고,
임계값이 config로 조정된다.

| ID | 우선순위 | 내용 | 착수 조건 |
|---|---|---|---|
| **INS-1** | 상 | ✅ *완료* — `statistical` CPU 백엔드 + 판정 로직 | — |
| **INS-2** | 상 | ✅ *완료* — 검사 노드 (`InspectPart` 서비스 + 진단 토픽) | — |
| **INS-3** | **상** | **EfficientAD 실학습** — 정상 이미지 수집 → anomalib 학습 → 체크포인트 | 없음 (목업 이미지로 착수 가능) |
| INS-4 | 상 | ✅ *완료* — `tools/export_mock_images.py` (npz → PNG, anomalib Folder 데이터셋용) | — |
| INS-5 | 중 | EfficientAD 점수 정규화 검증 — `_adaptive_threshold` 경로가 실제 anomalib 출력과 맞는지 확인 | INS-3 |
| INS-6 | 중 | 부품별 임계값 분리 (현재 전역 `inspection.threshold`) | 없음 |
| INS-7 | **상** | 과검출 원인 분석 (배치 극단 3건) + **end_stopper 얕은 스크래치 마진 부족** (아래) | 없음 |
| INS-8 | 하 | anomaly map RViz 시각화 설정 | INS-2 |
| INS-9 | 하 | 결함 종류 분류 (scratch/dent/stain/chip) — 현재는 OK/NG만 | INS-3 |

### INS-3 상세

```bash
# 1. 목업 데이터셋 생성
python3 tools/generate_mock_dataset.py --part all --train 200 --test 50

# 2. PNG 내보내기
python3 tools/export_mock_images.py --part guide_block

# 3. 학습
python3 tools/train_inspection.py --part guide_block --backend efficientad

# 4. 평가
python3 tools/evaluate.py --part guide_block --inspection efficientad

# 5. 실행
ros2 launch roboworld_bringup pipeline.launch.py inspection_backend:=efficientad
```

실 부품 이미지로 전환할 때는 **정상품만** 200장 이상, 조명·배치 변동을 포함해 촬영한다.
EfficientAD는 비지도 학습이므로 불량 이미지는 학습에 쓰지 않고 검증에만 쓴다.

### INS-7 상세 — end_stopper 얕은 스크래치 마진 부족 ⚠️

`tools/evaluate.py`가 보고하는 end_stopper의 **margin은 +0.002**이다(양품 최고점과
불량 최저점의 차). 집계상으로는 미검출 0건이지만 마진이 사실상 없다.

재현:

```bash
python3 tools/visualize.py --part end_stopper --defect scratch --seed 3 --fit-frames 25
#   → OK (양품)  score=0.495  threshold=0.500   ← 미검출
python3 tools/visualize.py --part end_stopper --defect scratch --seed 3 --fit-frames 60
#   → NG (불량)  score=0.526  threshold=0.500
```

**결함 위치는 정확히 특정된다**(defect_px≈350, anomaly 패널에서 빨간 영역이 스크래치와
일치). 점수만 임계값 근처에서 진동한다. 학습 프레임 수를 25→60으로 늘려도 점수가
0.492↔0.526 범위에 머물러, 학습량 문제가 아니라 **특징 자체의 민감도 한계**다.

대응 후보:
- EfficientAD 전환(INS-3) 후 재측정 — 이 백엔드는 CPU 참조 구현이다
- 얕은 선형 결함에 반응하는 특징 추가 (방향성 필터, 국소 라인 검출)
- 부품별 임계값 분리(INS-6)로 end_stopper만 하향

**집계 지표가 아니라 margin을 봐야 한다.** 미검출 0건이라는 숫자는 실 촬영 데이터에서
그대로 유지되지 않는다.

---

## Pose Agent (`feat/pose`)

**DoD (claude.md §3.3)**: 목업/샘플 RGB-D + CAD로 pose를 ICD대로 발행하고,
좌표계·단위가 문서와 일치한다.

| ID | 우선순위 | 내용 | 착수 조건 |
|---|---|---|---|
| **POSE-1** | 상 | ✅ *완료* — CAD 로더, 분할, ICP 백엔드 | — |
| **POSE-2** | 상 | ✅ *완료* — 포즈 노드 (`EstimatePose` 서비스 + TF 변환) | — |
| **POSE-3** | 상 | ✅ *완료* — 대칭 처리 및 평가 지표 | — |
| **POSE-4** | **상** | **Isaac ROS FoundationPose 브릿지 구현** (`pose_node._foundationpose_bridge`) | Isaac ROS 컨테이너 |
| POSE-5 | 상 | FoundationPose용 CAD 준비 — 텍스처 확인, 단위 검증, 필요 시 OBJ 변환 | 없음 |
| POSE-6 | **중** | 대칭 판별 — ICD §6.1 회신에 따라 필요 시 피처 기반 자세 결정 추가 | TL-10 회신 |
| POSE-7 | 중 | 공분산 산출 (`PoseResult.covariance` 현재 0) — 로봇 부서가 요구하면 | TL-9 |
| POSE-8 | 중 | Open3D 경로 검증 — 현재 numpy fallback만 실측됨 | Open3D 설치 |
| POSE-9 | 하 | 다중 부품 처리 (현재 최대 클러스터 1개만) | 없음 |
| POSE-10 | 하 | `roi_mask` 활용 — `EstimatePose.roi_mask`가 오면 분할 생략 | 없음 |

### POSE-4 상세

`roboworld_pose/pose_node.py`의 `_foundationpose_bridge`가 현재 `NotImplementedError`를
던진다. **의도적이다** — Isaac ROS 그래프가 없을 때 조용히 틀린 포즈를 반환하지 않는다.

구현 시:
1. RGB-D + 객체 마스크를 `isaac_ros_foundationpose` 입력 토픽으로 발행
2. `/pose_estimation/output` 구독 후 4×4 변환 + fitness 반환
3. **라이선스 경고 로그를 제거하지 말 것** (`LICENSE_NOTICE`)
4. `tools/evaluate.py --pose foundationpose`로 ICP 대비 정확도 비교

---

## ROS2 Agent (`feat/ros2`)

**DoD (claude.md §3.4)**: 실 카메라 또는 rosbag 재생으로 프레임·트리거가 정상 발행되고,
검사/포즈 결과가 ICD대로 최종 토픽에 발행된다.

| ID | 우선순위 | 내용 | 착수 조건 |
|---|---|---|---|
| **ROS-1** | 상 | ✅ *완료* — 패키지 구조, 메시지 변환, QoS | — |
| **ROS-2** | 상 | ✅ *완료* — 파이프라인 노드, 트리거, 분기 | — |
| **ROS-3** | 상 | ✅ *완료* — 프레임 소스 3종 (realsense/rosbag/mock) | — |
| **ROS-4** | 상 | ✅ *완료* — 목업 카메라 노드, 트리거 노드, launch 3종 | — |
| **ROS-5** | **상** | **D455 실물 연결 + rosbag 녹화** (`docs/setup_wsl.md` §4) | 하드웨어 |
| ROS-6 | **상** | `TopicFrameSource` 실측 검증 — 동기화 tolerance, 프레임 age 처리 | ROS-5 |
| ROS-7 | 상 | 실 트리거(포토센서/PLC) 연동 — 현재는 `trigger_node` 시뮬레이션 | 하드웨어 |
| ROS-8 | 중 | rosbag 자동 저장 (`pipeline.record_rosbag_on_ng`) 구현 | 없음 |
| ROS-9 | 중 | `InspectAndLocate` 액션 서버 구현 (현재 msg만 정의) | 없음 |
| ROS-10 | 중 | TF 정적 발행 (`station_base` ↔ `camera_link`) — 캘리브레이션 결과 반영 | TL-9 |
| ROS-11 | 중 | 노드 레벨 통합 테스트 (`launch_testing`) | 없음 |
| ROS-12 | 하 | RViz 설정 파일 (`roboworld_bringup/rviz/`) | 없음 |
| ROS-13 | 하 | 컴포저블 노드 전환 (프로세스 내 통신으로 이미지 복사 제거) | 성능 필요 시 |

### ROS-6 상세

`TopicFrameSource`는 ROS 없는 환경에서 검증할 수 없어 현재 **미실측**이다.
실 하드웨어 또는 rosbag으로 확인할 항목:

- `ApproximateTimeSynchronizer` slop 30 ms가 D455 실제 지터에 충분한가
- `frame_timeout_s` 1.0 s가 정지 스테이션 사이클에 적절한가
- `aligned_depth_to_color`가 실제로 컬러와 픽셀 정합되는가
- depth 인코딩이 `16UC1`(mm)인지 `32FC1`(m)인지 → `camera.depth_units_m` 확인

---

## 우선순위 요약

```
지금 당장 (외부 의존 없음)
  ├─ TL-9   좌표계 합의            ← 다른 모든 통합의 전제
  ├─ TL-10  대칭 모호성 통지        ← POSE-6 착수 여부 결정
  ├─ INS-3  EfficientAD 학습        ← 데이터셋·내보내기 도구 준비 완료
  └─ INS-7  과검출 원인 분석

하드웨어 도착 시
  ├─ ROS-5  D455 연결 + rosbag 녹화  ← ROS-6, ROS-7의 전제
  └─ ROS-7  실 트리거 연동

Isaac ROS 준비 시
  └─ POSE-4 FoundationPose 브릿지
```
