# RoboWorld — 컨베이어 비전 검사 & 6D 포즈 계산 시스템

컨베이어 정지 스테이션에서 MC 나일론 블록 3종을 검사하고, 양품에 한해 6D 포즈를 계산하여
**ROS 2 토픽으로 로봇 부서에 전달**하는 파이프라인.

```
트리거 → 캡처 → 표면 불량 검사 ─┬─ NG → 결과만 발행 (포즈 생략)
                                └─ OK → 6D 포즈 계산 → 결과 발행
                                                          ↓
                                        /roboworld/part_result  ← 로봇 부서 계약
```

**범위**: 카메라 입력 · 불량 검사 · 6D 포즈 · ROS 2 발행까지.
로봇 제어 · 모션 플래닝 · grasp · pick&place는 로봇 부서 담당 (claude.md §0).

---

## 30초 시작 — 카메라도 GPU도 ROS도 없이

```bash
python3 -m pip install -r requirements.txt
python3 tools/e2e_dryrun.py --inspection statistical --pose icp
```

CAD에서 RGB-D를 렌더링해 전체 사이클을 돌리고, 발행되는 모든 메시지를 ICD 계약에
대조 검증한다.

```
 seq stat    score pose_valid    t_err   r_err     tact  note
   1 OK      0.101       True   0.28mm   0.65d     258ms
   2 OK      0.008       True   1.41mm   0.42d     276ms
   3 NG      0.722      False        -       -     192ms  defect detected
   ...
PASSED: every cycle satisfied the ICD contract
```

---

## 문서

| 문서 | 내용 |
|---|---|
| **[docs/robot_interface_ICD.md](docs/robot_interface_ICD.md)** | **로봇 부서 인터페이스 계약 — 최우선 문서** |
| [docs/architecture.md](docs/architecture.md) | 아키텍처, 데이터 흐름, 알고리즘, 측정 결과 |
| [docs/setup_wsl.md](docs/setup_wsl.md) | WSL2 / CUDA / usbipd / ROS 2 구축 절차 |
| [docs/new_part_registration.md](docs/new_part_registration.md) | **CAD 없이 사진으로 새 부품 등록·검출** |
| [docs/git_strategy.md](docs/git_strategy.md) | 브랜치 · PR · 머지 게이트 |
| [docs/agent_tickets.md](docs/agent_tickets.md) | 에이전트별 작업 티켓 |

---

## 기술 스택 (claude.md §1 고정 — 임의 변경 금지)

| 항목 | 선택 |
|---|---|
| OS / 미들웨어 | Ubuntu 22.04 (WSL2) + ROS 2 Humble |
| 언어 | Python 3.10 |
| GPU | RTX 5070 (Blackwell sm_120), CUDA 12.x |
| 카메라 | Intel RealSense D455 + realsense-ros |
| 불량 검사 | EfficientAD (anomalib) |
| 6D 포즈 | FoundationPose (Isaac ROS) |
| 포인트클라우드 | Open3D |
| 빌드 | colcon |

각 알고리즘에는 **CPU 참조 백엔드**가 함께 구현되어 있다 (`statistical`, `icp`).
GPU·Isaac ROS 없이도 전체 파이프라인이 실제로 동작하며 CI가 이를 검증한다.

---

## 패키지 구성

```
ros2_ws/src/
├── roboworld_interfaces/   msg/srv/action — 계약 (Team Lead 단독 관리)
├── roboworld_core/         ROS 비의존 코어: 설정·기하·목업·백엔드·상태기계
├── roboworld_ros_utils/    ROS 어댑터 (메시지 변환, QoS, 파라미터)
├── roboworld_inspection/   검사 노드
├── roboworld_pose/         포즈 노드
├── roboworld_pipeline/     오케스트레이션 · 프레임 소스 · 목업 카메라 · 트리거
└── roboworld_bringup/      launch + config(단일 진실 원천) + CAD 사본
```

`roboworld_core`는 `rclpy`를 import하지 않는다. 그래서 ROS 미설치 환경에서도
**154개 유닛 테스트와 E2E dry-run이 통과**하며, 4개 에이전트가 하드웨어를 기다리지 않고
병렬 개발할 수 있다 (claude.md §2).

---

## 실행

### ROS 없이 (개발·CI)

```bash
make install     # 의존성 설치
make test        # 유닛 테스트 154개
make dryrun      # E2E dry-run — 계약 검증 게이트
make evaluate    # 검출률·포즈 정확도 (허용치 게이트)
make dataset     # 목업 RGB-D 데이터셋 생성
make symmetry    # 부품 대칭 모호성 리포트
make visualize   # 결과 영상 1장 저장 (컬러 | 깊이 | 분할 | 이상맵)
                 #   make visualize PART=end_stopper DEFECT=scratch
make live        # 실시간 RGB-D 창 (WSLg 필요)
                 #   make live LIVE_ARGS='--inspect --pose'
make check       # 위 전체 (PR 전 게이트)
```

기본 인터프리터는 `python3`이다. conda 등으로 여러 파이썬이 섞여 있으면
`make check PYTHON=/usr/bin/python3.10` 처럼 지정한다 (`make install`에도 같은 값을 쓸 것).

### ROS 2로 (통합)

```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash

# 목업 프레임으로 전체 파이프라인 (하드웨어 불필요)
ros2 launch roboworld_bringup pipeline.launch.py

# 결과 확인
ros2 topic echo /roboworld/part_result
```

주요 launch 인자:

```bash
ros2 launch roboworld_bringup pipeline.launch.py \
    part_id:=spacer_block \
    camera_source:=realsense \        # realsense | rosbag | mock
    inspection_backend:=efficientad \ # efficientad | statistical | stub
    pose_backend:=foundationpose      # foundationpose | icp | stub
```

rosbag 재생 (하드웨어 없는 권장 개발 방식):

```bash
ros2 launch roboworld_bringup rosbag_replay.launch.py bag:=/path/to/station_capture
```

---

### 실시간 영상 보기

```bash
python3 -m pip install --user opencv-python pyrealsense2   # 최초 1회

python3 tools/live_view.py                        # 목업 (CAD 렌더링)
python3 tools/live_view.py --source realsense     # 실제 D455 실시간 영상
python3 tools/live_view.py --inspect --pose       # 검사·포즈 패널 추가
```

WSL2에서는 **WSLg**(Windows 11)가 필요하며 `$DISPLAY`가 설정되어 있으면 이미 사용 가능한
상태다. 창을 띄울 수 없는 환경(SSH·헤드리스)에서는 브라우저로 스트리밍한다:

```bash
python3 tools/live_view.py --backend mjpeg      # http://localhost:8080
```

단축키: `q` 종료 · `space` 일시정지 · `d` 결함 변경 · `p` 부품 변경 ·
`i` 검사 패널 · `o` 포즈 · `s` 스냅샷 저장

실 카메라는 먼저 WSL에 연결해야 한다 (docs/setup_wsl.md §4):

```bash
"/mnt/c/Program Files/usbipd-win/usbipd.exe" list
"/mnt/c/Program Files/usbipd-win/usbipd.exe" attach --wsl --busid <BUSID>
```

> `tools/live_view.py --source realsense`는 **pyrealsense2로 직결**한 경로로, 카메라
> 조준·동작 확인용이다. 운영 경로는 realsense-ros + **RViz2**이며 rosbag 스트림도
> RViz2로 본다.

### CAD 없이 새 부품 등록

사진만으로 새 부품을 등록해 **불량 검사**를 테스트할 수 있다 (6D 포즈는 형상 필요).

```bash
# 검사만 (형상 불필요)
python3 tools/capture_part.py  --part my_part     # 정상품 촬영 (40장 이상 권장)
python3 tools/register_part.py --part my_part --evaluate

# 6D 포즈까지 — depth로 메시 복원
python3 tools/reconstruct_part.py --part my_part
python3 tools/register_part.py --part my_part --mesh data/meshes/my_part.ply

python3 tools/live_view.py --source realsense --part my_part --inspect --pose
```

복원 메시는 실제 CAD와 **동등한 포즈 정확도**(위치오차 1.1~1.9 mm)를 보인다.

절차와 주의사항은 [docs/new_part_registration.md](docs/new_part_registration.md) 참조.

## 설정

모든 운영 값은 `ros2_ws/src/roboworld_bringup/config/`의 YAML에 있다.
코드 하드코딩 금지 (claude.md §4).

| 파일 | 내용 |
|---|---|
| `parts.yaml` | 부품 CAD 경로, 단위, 대칭군 |
| `camera.yaml` | D455 내부 파라미터, 토픽, 프레임 소스 |
| `inspection.yaml` | 백엔드, 임계값, 정규화 파라미터 |
| `pose.yaml` | 백엔드, 수용 게이트, 분할·ICP 파라미터 |
| `pipeline.yaml` | 토픽, QoS, 택트 예산, 분기 규칙 |

이 파일들은 **평문 YAML로도, ROS 2 파라미터 파일로도** 읽힌다.
따라서 유닛 테스트와 노드가 항상 같은 값을 본다.

---

## 현재 성능 (모의 데이터, CPU 백엔드)

`make evaluate` — 부품별 양품 25 / 불량 25, 학습 40프레임

| 부품 | 미검출 | 과검출 | 위치오차(중앙/최대) | 자세오차(중앙/최대) |
|---|---|---|---|---|
| guide_block | 0/25 | 1/25 | 1.36 / 3.69 mm | 0.72 / 1.78° |
| spacer_block | 0/25 | 1/25 | 0.95 / 2.52 mm | 0.58 / 2.29° |
| end_stopper | 0/25 | 1/25 | 0.93 / 2.40 mm | 0.52 / 1.83° |

택트 타임 중앙값 258 ms (예산 1200 ms).

> 실 카메라 · EfficientAD · FoundationPose 전환 후 반드시 재측정한다.
> **자세 정확도는 대칭 보정 기준이다.** 보정 전에는 guide_block 84 %, spacer_block 68 %,
> end_stopper 12 %의 프레임에서 자세가 약 180° 뒤집힌다 — ICD §6.1 참조.

---

## ⚠️ 로봇 부서 확인 필요 사항

1. **좌표계 합의** — `world` 프레임 정의, hand-eye 캘리브레이션 책임 주체 (ICD §4)
2. **자세 대칭 모호성** — 세 부품 모두 55×55 정사각 단면이라 단일 시점 depth로는
   자세가 order-8 대칭군까지만 결정된다. **grasp이 이에 불변인지 확인 필요** (ICD §6.1).
   불변이 아니라면 guide_block / spacer_block은 알고리즘으로 해결 불가이며
   설계 변경(비대칭 마커) 또는 카메라 추가가 필요하다.
3. 요구 택트 타임 · 정확도 목표치 (현재 값은 비전팀 가정)

---

## 라이선스 주의

FoundationPose 공개판은 **비상업(NC)** 라이선스이다. 데모/평가 목적에 한해 사용하며
상업화 시 NGC 상업판 전환이 필요하다 (claude.md §1).
