# Git 브랜치 전략 및 통합 게이트

claude.md §5를 구체화한 문서.

---

## 1. 브랜치 구조

```
main ──────●──────────●──────────●──────────●─────▶  항상 배포 가능 상태
            \        /          /          /
             \      /          /          /
   feat/inspection ●          /          /
                             /          /
              feat/pose ────●          /
                                      /
              feat/ros2 ─────────────●
```

| 브랜치 | 소유 | 용도 |
|---|---|---|
| `main` | Team Lead | 통합 브랜치. **직접 push 금지** |
| `feat/inspection` | Inspection Agent | EfficientAD, 검사 노드 |
| `feat/pose` | Pose Agent | FoundationPose, 포즈 노드 |
| `feat/ros2` | ROS2 Agent | 드라이버, 트리거, 파이프라인 |
| `feat/<name>` | 누구나 | 그 외 작업 |
| `fix/<name>` | 누구나 | 버그 수정 |
| `docs/<name>` | 누구나 | 문서만 변경 |

---

## 2. 소유권 규칙

| 경로 | 소유 | 변경 시 |
|---|---|---|
| `ros2_ws/src/roboworld_interfaces/**` | **Team Lead 단독** | ICD 개정 + 전원 통지 + 로봇 부서 합의 |
| `docs/robot_interface_ICD.md` | **Team Lead 단독** | 로봇 부서 합의 필수 |
| `ros2_ws/src/roboworld_core/roboworld_core/inspection/**` | Inspection | |
| `ros2_ws/src/roboworld_core/roboworld_core/pose/**` | Pose | |
| `ros2_ws/src/roboworld_pipeline/**`, `roboworld_ros_utils/**` | ROS2 | |
| `ros2_ws/src/roboworld_bringup/config/**` | 해당 섹션 소유자 | 스키마 변경은 Team Lead 승인 |
| `roboworld_core/{config,geometry,types,pipeline,contract}.py` | Team Lead | 리뷰 필수 |

> **인터페이스 변경이 필요하면 코드를 먼저 쓰지 말고 Team Lead에게 요청한다.**
> §2의 계약 우선 원칙이 무너지면 병렬 개발 자체가 성립하지 않는다.

---

## 3. 커밋 규칙

Conventional Commits 형식:

```
<type>(<scope>): <요약>

<본문 — 왜 이렇게 했는지. 무엇을 했는지는 diff가 말해준다>

Refs: <티켓 ID>
```

| type | 용도 |
|---|---|
| `feat` | 기능 추가 |
| `fix` | 버그 수정 |
| `perf` | 성능 개선 |
| `refactor` | 동작 변경 없는 구조 개선 |
| `test` | 테스트만 |
| `docs` | 문서만 |
| `chore` | 빌드·CI·설정 |

scope는 `inspection`, `pose`, `ros2`, `interfaces`, `core`, `bringup`, `docs` 중 하나.

예:

```
fix(pose): 초기 추정 시 부품 중심을 관측면 뒤쪽으로 배치

기존에는 중심을 카메라 쪽으로 half-thickness 이동시켜, 모델의 반대면이
관측 표면에 정합되면서 위치 오차가 정확히 블록 두께(55 mm)만큼 발생했다.
depth는 근접면만 보므로 중심은 관측면 뒤에 있어야 한다.

Refs: POSE-2
```

---

## 4. PR 절차

1. feature 브랜치에서 작업
2. 로컬 게이트 통과 확인 (§5)
3. PR 생성 — 아래 템플릿 사용
4. CI 통과
5. **Team Lead 리뷰 후 머지** (squash merge 권장)
6. 브랜치 삭제

### PR 템플릿

```markdown
## 변경 내용

## 티켓
Refs: INS-3

## 인터페이스 영향
- [ ] `roboworld_interfaces` 변경 없음
- [ ] 변경 있음 → ICD 개정 + 로봇 부서 통지 완료 (링크: )

## 검증
- [ ] `python3 -m pytest ros2_ws/src/roboworld_core/test -q`
- [ ] `python3 tools/e2e_dryrun.py`
- [ ] `python3 tools/evaluate.py --part all`  (정확도에 영향 있는 경우)
- [ ] `colcon build && colcon test`  (ROS 환경 보유 시)

## 측정값 변화
| 항목 | 이전 | 이후 |
|---|---|---|
```

---

## 5. 머지 게이트

CI(`.github/workflows/ci.yml`)가 강제한다. 하나라도 실패하면 머지 불가.

| # | 게이트 | 명령 | 실패 의미 |
|---|---|---|---|
| 1 | 린트 | `ruff check` / `flake8` | 스타일 |
| 2 | 유닛 테스트 | `pytest ros2_ws/src/roboworld_core/test` | 로직 회귀 |
| 3 | **계약 스키마** | `test_contract.py` | `.msg` ↔ 코드 불일치 |
| 4 | **E2E dry-run** | `tools/e2e_dryrun.py` | 파이프라인·계약 위반 |
| 5 | 정확도 | `tools/evaluate.py` | 검출률/포즈 오차 허용치 초과 |
| 6 | ROS 빌드 | `colcon build && colcon test` | ROS 통합 실패 |

게이트 3·4는 **절대 우회 금지**. 이것이 로봇 부서와의 계약을 지키는 유일한 자동 장치다.

### 로컬에서 전체 게이트 실행

```bash
make check        # 1~5 (ROS 불필요)
make ros-check    # 6 (ROS 환경 필요)
```

---

## 6. 통합 순서 (claude.md §6.3)

1. Team Lead 산출물 → `main` (계약·목업·스텁·CI) ← **완료**
2. 3개 에이전트 병렬 개발 (모두 목업 기반)
3. PR → 통합 테스트 → Team Lead 순차 머지
4. 스텁 → 실물 교체 (EfficientAD, FoundationPose, D455)
5. E2E 검증: 최종 출력 토픽이 ICD를 만족하는지 확인

### 스텁 교체 순서 (권장)

리스크가 낮은 순서로, 한 번에 하나씩만 바꾼다.

```
stub          → statistical / icp   (CPU, 검증 가능)   ← 현재 위치
statistical   → efficientad         (GPU 필요)
icp           → foundationpose      (Isaac ROS 필요)
mock          → rosbag              (실 데이터)
rosbag        → realsense           (실 하드웨어)
```

각 단계마다 `tools/evaluate.py`로 수치를 재측정하고 ICD §7을 갱신한다.

---

## 7. 태그 및 버전

`pipeline_version`(`config/pipeline.yaml`)과 git 태그를 일치시킨다.

```bash
git tag -a v0.1.0 -m "계약 확정, 목업 기반 E2E 통과"
git push origin v0.1.0
```

| 변경 | 버전 |
|---|---|
| 메시지 필드 삭제/의미 변경 | MAJOR |
| 메시지 필드 추가, 백엔드 추가 | MINOR |
| 버그 수정, 임계값 조정 | PATCH |

---

## 8. 최초 저장소 초기화

아직 git 저장소가 아닌 경우:

```bash
cd ~/RoboWorld_Demo
git init -b main
git add .
git commit -m "chore: 초기 스캘폴드 — 계약, 목업, 스텁, CI"

git checkout -b feat/inspection    # Inspection Agent
git checkout -b feat/pose          # Pose Agent
git checkout -b feat/ros2          # ROS2 Agent
```

`.gitignore`에 `data/`, `build/`, `install/`, `log/`, `__pycache__/` 가 포함되어 있는지
확인할 것 — 학습된 모델과 rosbag은 저장소에 넣지 않는다.
