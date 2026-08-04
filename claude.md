# Claude Code Agent Team 프롬프트 — 컨베이어 비전 검사 & 6D 포즈 계산 시스템

> 아래 내용을 Claude Code에 그대로 붙여넣어 사용하세요. `[ ]` 로 표시된 부분은 환경에 맞게 조정하면 됩니다.

---

## 0. 프로젝트 개요 및 범위

컨베이어 위를 이동하는 부품(MC 나일론 블록 3종)이 정지 스테이션에 멈추면,
① EfficientAD로 표면 불량을 검사하고, ② 불량이면 결과만 통보(포즈 계산 스킵),
③ 양품이면 FoundationPose로 6D 포즈(위치·자세)를 계산하여, ④ **결과(불량 여부 + 6D Pose)를 ROS2로 발행**한다.

**우리 팀의 범위(중요)**: 비전 검사 + 포즈 계산 + **ROS2 전달까지**.
- ✅ 범위 내: 카메라 입력, 불량검사, 6D 포즈, ROS2 메시지 발행(로봇 부서가 구독할 인터페이스)
- ❌ 범위 밖: **로봇 제어, 모션 플래닝, grasp, Isaac Sim, pick&place** — 이는 별도 로봇 부서 담당
- 따라서 로봇 부서와의 **출력 인터페이스(ICD) 합의**가 우리 최종 산출물의 핵심이다.

**동작 방식**: 연속 실시간 추적이 아니라 "트리거 → 캡처 → 검사 → 분기 → (양품) 포즈 → ROS2 발행"의 index(정지) 방식.

## 1. 고정 기술 스택 (에이전트는 임의로 변경 금지)

- **개발 환경**: Windows + **WSL2 Ubuntu 22.04** (24.04에서 22.04로 재구성 — Humble/Isaac ROS 호환 위해), **VSCode + WSL Remote**
- **OS/ROS2**: Ubuntu 22.04 + **ROS 2 Humble (LTS)**
- **언어/런타임**: Python 3.10, C++17(필요 시)
- **GPU/딥러닝**: RTX 5070 12GB, CUDA 12.x + Blackwell(sm_120) 지원 PyTorch 빌드
  - ⚠️ WSL2에서 GPU는 **Windows용 NVIDIA 드라이버 + WSL CUDA 툴킷**으로 동작(리눅스 드라이버 설치 금지). Blackwell 지원 최신 드라이버·PyTorch 필요
- **카메라**: **Intel RealSense D455** (RGB-D) + `librealsense` + `realsense-ros`(Humble)
  - ⚠️ WSL2는 USB 미인식이 기본 → **`usbipd-win`으로 D455를 WSL에 attach** 필요. 초기 개발은 **rosbag 녹화/재생 또는 목업**으로 하드웨어 의존 제거
- **불량검사**: EfficientAD (Intel **anomalib** 라이브러리 기반)
- **6D 포즈**: FoundationPose (**Isaac ROS Pose Estimation** 패키지로 통합 — 22.04/Humble 네이티브)
  - ⚠️ 라이선스: FoundationPose GitHub 공개판은 **비상업(NC)**. 상업화 시 NGC 상업판 전환 필요 — 데모 단계에서만 공개판 사용
- **포인트클라우드/정합**: Open3D
- **컨테이너/재현성**: Docker(선택) + `requirements.txt`/`package.xml`로 환경 고정
- **패키지/빌드**: colcon 워크스페이스

## 2. 최우선 원칙 — "계약 먼저, 그다음 병렬"

에이전트가 병렬로 일하려면 **하드웨어·GPU 없이도** 각자 개발·테스트가 가능해야 한다. 따라서:

1. Team Lead가 **인터페이스 계약(메시지 스키마) + 목업 데이터 + 스텁 노드**를 먼저 확정·제공한다.
2. 각 에이전트는 실제 카메라/모델 대신 **목업(mock)** 으로 개발을 시작하고, 통합 단계에서 실물로 교체한다.
3. 모듈 간 데이터는 **직접 함수 호출이 아니라 ROS2 토픽/서비스/액션으로 디커플**한다. (예: Inspection이 Pose를 직접 호출하지 않음 — 분기는 파이프라인 노드가 담당)

## 3. 팀 구성 및 역할 (4명)

각 에이전트는 (a) 담당 모듈, (b) 산출물, (c) 인터페이스 계약, (d) 테스트, (e) 완료 정의(DoD)를 반드시 만족한다.

### 1) Team Lead Agent (아키텍트 & 통합 책임)
- 전체 아키텍처 설계: 디렉토리 구조, 데이터 흐름, 좌표계(TF2 프레임) 정의
- **공통 메시지/스키마 정의**: 커스텀 msg(불량 결과, 6D Pose, 검사 요청/응답)를 별도 `*_interfaces` 패키지로 선(先) 배포
- **로봇 부서 대상 출력 ICD(Interface Control Document) 작성**: 발행 토픽명, 메시지 필드, 좌표계, 단위, 발행 주기, QoS를 문서화하여 다운스트림(로봇 부서)이 그대로 구독 가능하게 함
- **목업 데이터·스텁 노드 제공**: 샘플 RGB-D 프레임, 더미 anomaly map, 더미 pose 스텁으로 병렬 개발 가능하게 함
- git 브랜치 전략·CI 스켈레톤·통합 테스트 프레임 구축, 각 에이전트 조율 및 최종 통합·E2E 검증
- **DoD**: `docs/architecture.md` + `docs/robot_interface_ICD.md` + interfaces 패키지 + mock 데이터 + CI가 main에 존재하고, 스텁만으로 E2E 파이프라인이 dry-run 된다.

### 2) Inspection Agent (표면 불량 검사)
- anomalib **EfficientAD** 통합, 학습(정상 이미지)·추론 모듈 구현
- anomaly map 생성 + threshold 기반 OK/NG 판정 로직 (임계값은 config로 외부화)
- 결과(불량 여부, anomaly score, 마스크)를 **ROS2 토픽/서비스로 발행** (Pose를 직접 호출하지 않음)
- 실물 없이 개발할 수 있도록 목업 이미지로 유닛 테스트 작성
- **DoD**: 목업 입력에 대해 OK/NG + score + mask를 계약대로 발행하고, 임계값이 config로 조정된다.

### 3) Pose Agent (6D 포즈 추정)
- FoundationPose 연동, CAD(mesh, 기 제작 STEP→OBJ/PLY) 로드·초기화
- **양품 신호를 받은 경우에만** 해당 프레임으로 6D pose(R,t) 계산 (registration 중심, 필요 시 tracking)
- 결과를 `geometry_msgs/PoseStamped` 또는 커스텀 6D Pose msg로 발행, **TF2 프레임·단위(m)·quaternion 규약 준수**
- 목업 pose 스텁으로 다운스트림·통합이 먼저 개발되게 지원
- **DoD**: 목업/샘플 RGB-D + CAD로 pose를 ICD대로 발행하고, 좌표계·단위가 문서와 일치한다.

### 4) ROS2 Agent (미들웨어 · 하드웨어 I/O · 파이프라인 오케스트레이션)
- ROS2 Humble 노드/패키지 구조 설계, **RealSense D455 드라이버(`realsense-ros`) 연동** 및 Image/Depth Subscriber 구현
- WSL2 USB(`usbipd-win`) attach 절차 문서화, 하드웨어 부재 시 **rosbag 재생 경로** 제공
- 정지 스테이션 **트리거(포토센서/토픽)** 수신, 카메라 intrinsics/정렬(align depth to color)/타임스탬프 동기화
- **파이프라인 상태 흐름 담당**: 트리거 → 검사 요청 → (NG면 결과만 발행·포즈 스킵) → (OK면) 포즈 요청 → 최종 결과 발행
- 커스텀 Message/Service/Action 구현(스키마는 Team Lead 계약 준수), QoS 설정, **로봇 부서로의 최종 발행 토픽** 구현
- **DoD**: 실 카메라 또는 rosbag 재생으로 프레임·트리거가 정상 발행되고, 검사/포즈 결과가 ICD대로 최종 토픽에 발행된다.

## 4. 개발 규약

- **출력 인터페이스(로봇 부서와 합의 필요)**: 최종 발행 메시지에는 최소한 `part_id`, `is_good(OK/NG)`, `anomaly_score`, `pose(PoseStamped: frame_id/position(m)/orientation(quat))`, `stamp` 포함. **좌표계(camera 프레임 vs 합의된 world 프레임)와 Hand-eye/캘리브레이션 책임 소재를 로봇 부서와 초기에 확정**한다.
- **좌표/단위**: 길이 단위 미터(m), 회전 quaternion, 모든 pose에 `frame_id`·timestamp 명시. CAD는 mm→m 스케일 변환 규칙 문서화.
- **설정 외부화**: 임계값·카메라 파라미터·경로 등 코드 하드코딩 금지, YAML config로 관리.
- **로깅/진단**: 각 노드는 처리시간(택트 타임)·성공/실패 로깅. 재현용 rosbag 저장 옵션.
- **테스트**: 각 에이전트는 목업 기반 유닛 테스트 필수. 인터페이스는 계약 테스트(contract test)로 검증.

## 5. Git 및 통합 전략

- 각 에이전트는 **feature 브랜치**(`feat/inspection`, `feat/pose`, `feat/ros2`)에서 작업.
- main 직접 push 금지. **PR → 통합 테스트 통과 → Team Lead 리뷰/머지**.
- `*_interfaces` 패키지·ICD는 Team Lead가 관리하며, 변경 시 전 에이전트 및 로봇 부서에 공지(스키마 변경은 계약 재확정 절차).
- 통합 테스트(E2E dry-run: 스텁 파이프라인)가 CI에서 통과해야 머지 가능.

## 6. 첫 번째 작업 지시 (순서 고정)

1. **Team Lead Agent** 먼저 단독 실행:
   - `docs/architecture.md` 작성: 디렉토리 구조, 데이터 흐름도, TF 프레임, 택트 타임 예산
   - `docs/robot_interface_ICD.md` 작성: 로봇 부서가 구독할 최종 출력 토픽/메시지/좌표계 규격
   - `*_interfaces` 패키지에 커스텀 msg/srv/action 정의 (불량 결과 / 6D Pose / 검사·포즈 요청)
   - `docs/setup_wsl.md` 작성: WSL2 Ubuntu 22.04 재구성, NVIDIA 드라이버/WSL CUDA, `usbipd-win`으로 D455 attach, colcon 워크스페이스 초기화 절차
   - **목업 데이터 + 스텁 노드** + colcon 워크스페이스 스켈레톤 + CI(빌드/린트/E2E dry-run) 커밋
   - git 브랜치 전략과 각 에이전트 작업 티켓 문서화
2. 위 산출물이 main에 올라간 뒤, **나머지 3개 에이전트가 각자 feature 브랜치에서 병렬 개발** 시작 (모두 목업 기반).
3. 각 모듈 완성 후 PR → 통합 테스트 → Team Lead가 순차 통합 → 실 하드웨어/모델로 스텁 교체 → E2E 검증(최종 출력 토픽이 ICD를 만족하는지 확인).

---

### 참고: 이 프롬프트가 기존 초안에서 바뀐 점
- 범위를 **비전 검사 + 포즈 + ROS2 전달까지**로 명확화 (로봇 제어·Isaac Sim·grasp는 로봇 부서 담당으로 제외)
- 최종 산출물의 핵심으로 **로봇 부서 대상 출력 ICD(인터페이스 규격 문서)** 를 신설
- Inspection→Pose 강결합을 **ROS2 토픽 디커플 + 파이프라인 노드 분기(ROS2 Agent)** 로 변경
- 병렬 개발을 위한 **인터페이스 계약 + 목업/스텁 우선** 원칙 명문화
- **기술 스택 버전 고정**(ROS2 Humble, CUDA/Blackwell, anomalib, Isaac ROS)
- FoundationPose **라이선스**·RTX 50 **호환성** 제약 명시
- 각 에이전트 **완료 정의(DoD)·테스트·좌표계 규약**, **git/통합 게이트** 추가