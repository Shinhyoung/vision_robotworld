# 로봇 부서 인터페이스 규격서 (ICD)

**Interface Control Document — 비전 검사/6D 포즈 → 로봇 제어**

| 항목 | 내용 |
|---|---|
| 문서 버전 | 0.2.1 |
| 대상 파이프라인 버전 | `pipeline_version = "0.1.0"` |
| 작성 | 비전팀 (Team Lead) |
| 수신 | 로봇 부서 |
| 상태 | **초안 — 로봇 부서 회신 필요 (§4 좌표계, §6.1 대칭, §10 구성)** |

**0.2.x 변경 사항** — 메시지 규격(§3)은 변경되지 않았다. 실측으로 확인된 동작을
추가한 것이므로 기존 구독 코드는 그대로 동작한다.

0.2.1
- §6.2 ⚠️ **실측으로 확인된 불일치 명시** — 부품 미검출 시 현재 구현은
  `STATUS_NO_POSE`가 아니라 `STATUS_NG`를 발행한다. 수정 예정
- §6.2.1 **스테이션 ROI** 추가 — 정지 위치에 있는 부품만 처리한다.
  `STATUS_NO_POSE`의 사유가 하나 늘었고, `message`로 구분된다

0.2.0
- §6.2 **부품 식별** 추가 — 등록된 부품이 아니면 처리하지 않는다
- §6.3 **작업 거리** 추가 — 거리별 정확도와 권장 범위
- §10 **시스템 구성** 추가 — PC 1대/2대 선택과 전달물, 네트워크 요건

---

## 1. 범위

비전팀이 제공하는 것은 **"부품 1개당 검사 결과 1건 + (양품인 경우) 6D 포즈"** 이며,
ROS 2 토픽으로 발행한다. 로봇 제어·모션 플래닝·grasp·pick&place는 로봇 부서 범위이다.

```
[컨베이어 정지] → 트리거 → 캡처 → EfficientAD 검사 ─┬─ NG → 결과만 발행 (포즈 없음)
                                                    └─ OK → FoundationPose 6D 포즈 → 결과 발행
                                                                                        │
                                                            ┌───────────────────────────┘
                                                            ▼
                                              /roboworld/part_result  ← 본 문서의 계약
```

**트리거 1회 = 메시지 정확히 1건.** 검사 실패·포즈 실패·하드웨어 오류에도 반드시 1건이
발행된다. 구독자는 "메시지가 오지 않는 것"을 정상 상태로 취급해서는 안 된다.

---

## 2. 발행 토픽

| 토픽 | 타입 | 성격 | 설명 |
|---|---|---|---|
| **`/roboworld/part_result`** | `roboworld_interfaces/msg/PartResult` | **계약 (필수 구독)** | 사이클당 1건, 최종 결과 |
| `/roboworld/inspection_result` | `roboworld_interfaces/msg/InspectionResult` | 진단용 | anomaly map/mask 포함 |
| `/roboworld/pose_result` | `roboworld_interfaces/msg/PoseResult` | 진단용 | fitness/RMSE 포함 |
| `/roboworld/pipeline_status` | `diagnostic_msgs/msg/DiagnosticArray` | 진단용 | 택트 타임, 단계별 소요 |

진단용 토픽은 **계약에 포함되지 않는다.** 예고 없이 변경될 수 있으므로 로봇 제어 로직이
의존해서는 안 된다.

### 구독 측 서비스 (선택)

요청/응답 방식을 선호하는 경우:

| 인터페이스 | 타입 | 설명 |
|---|---|---|
| `/roboworld/trigger_capture` | `roboworld_interfaces/srv/TriggerCapture` | 1사이클 수동 트리거 |
| `InspectAndLocate` (action) | `roboworld_interfaces/action/InspectAndLocate` | 진행 상황 피드백 포함 |

---

## 3. `PartResult` 메시지 규격

```
std_msgs/Header header
  ├─ stamp     : 해당 결과가 유래한 컬러 프레임의 캡처 시각 (발행 시각 아님)
  └─ frame_id  : 원본 데이터의 카메라 광학 프레임

uint32  sequence            # 트리거 순번, 단조 증가. 유실 감지에 사용
string  part_id             # "guide_block" | "spacer_block" | "end_stopper"

uint8   status              # 아래 §3.1
bool    is_good             # 검사 합격 여부
float32 anomaly_score       # [0.0, 1.0] 정규화된 이상 점수
float32 anomaly_threshold   # 판정에 사용된 임계값 [0.0, 1.0]

bool    pose_valid          # true일 때만 pose 사용 가능
geometry_msgs/PoseStamped pose
  ├─ header.frame_id : 포즈의 기준 좌표계 — **이 필드가 최종 권위**
  ├─ position        : 미터 (m)
  └─ orientation     : 단위 쿼터니언 (x, y, z, w)
float32 pose_fitness        # [0.0, 1.0] 정합 품질. 높을수록 좋음

float32 tact_time_ms        # 트리거 → 발행 소요 시간
string  pipeline_version    # 시맨틱 버전. 변경 시 본 문서 개정
string  message             # 진단 문자열. 정상 OK에서는 빈 문자열
```

### 3.1 `status` 열거값 — **로봇 부서 분기 기준**

| 값 | 상수 | 의미 | **로봇 동작 (권장)** |
|---|---|---|---|
| 0 | `STATUS_OK` | 양품 + 포즈 유효 | **집는다** |
| 1 | `STATUS_NG` | 불량 검출, 포즈 계산 생략 | **불량 배출**. 포즈 필드 무시 |
| 2 | `STATUS_NO_POSE` | 양품이나 포즈 실패 | **집지 않는다.** 재트리거 또는 수동 처리 |
| 3 | `STATUS_ERROR` | 파이프라인/하드웨어 오류 | **집지 않는다.** `message` 확인, 알람 |

> **`status`만으로 분기할 것.** `pose_valid`나 `is_good`을 단독 조건으로 쓰면
> `STATUS_NO_POSE`와 `STATUS_ERROR`가 구분되지 않는다.

### 3.2 필드 유효성 보장

발행 전 다음이 항상 성립함을 파이프라인이 보장한다 (`roboworld_core.contract`가 검증):

- `orientation` 은 단위 쿼터니언 (‖q‖ = 1 ± 1e-6)
- `position` 은 유한값, **미터 단위** (mm 아님)
- `0.0 ≤ anomaly_score ≤ 1.0`, `0.0 ≤ anomaly_threshold ≤ 1.0`
- `status == STATUS_OK` ⟺ `is_good && pose_valid`
- `status ∈ {NG, NO_POSE, ERROR}` ⟹ `pose_valid == false`
- `pose_valid == true` ⟹ `pose.header.frame_id != ""`
- `pipeline_version != ""`

---

## 4. 좌표계 — **로봇 부서 합의 필요 (미결)**

### 4.1 TF 트리

```
world ──(A: 미결)── station_base ──(B: 미결)── camera_link ──(C: 확정)── camera_color_optical_frame
                                                                                    │
                                                                        (D) part_frame (발행 포즈)
```

| 구간 | 내용 | 책임 | 상태 |
|---|---|---|---|
| C | `camera_link` → `camera_color_optical_frame` | realsense-ros (자동) | **확정** |
| D | 부품 모델 프레임 정의 (§4.3) | 비전팀 | **확정** |
| B | 카메라 외부 파라미터 (hand-eye 또는 고정 캘리브레이션) | **미정** | **합의 필요** |
| A | `world` 원점 정의 | 로봇 부서 | **합의 필요** |

### 4.2 현재 발행 프레임 (기본값)

기본 설정은 `pose.header.frame_id = "camera_color_optical_frame"` 이다.
축 규약은 **OpenCV/ROS 광학 프레임**:

| 축 | 방향 |
|---|---|
| +x | 이미지 오른쪽 |
| +y | 이미지 아래쪽 |
| +z | 카메라가 바라보는 방향 (장면 안쪽) |

`world` 프레임으로 발행하려면 `config/pose.yaml`에서:

```yaml
pose:
  output_frame_id: "world"
  transform_to_output_frame: true
```

이 경우 TF `world ← camera_color_optical_frame` 이 **반드시** 존재해야 한다.
**TF가 없으면 포즈는 카메라 광학 프레임 그대로 발행되고 `message`에 fallback 사실이
기록된다.** 조용히 잘못된 프레임으로 발행하지 않는다.
→ 그래서 `pose.header.frame_id`가 최종 권위이다. 하드코딩하지 말고 읽을 것.

### 4.3 부품 모델 프레임 (확정)

CAD(`01_input/*.ply`)의 축 정렬 바운딩 박스 중심을 원점으로 재정렬한다.

| 축 | 정의 | 치수 |
|---|---|---|
| +x | 장축 (긴 방향) | 200 mm |
| +y | | 55 mm |
| +z | | 55 mm |

발행되는 `position`은 **블록의 기하학적 중심**이다(윗면 아님).

### 4.4 단위 (확정)

| 항목 | 단위 |
|---|---|
| 길이 | **미터 (m)** |
| 회전 | 단위 쿼터니언 (x, y, z, w) |
| 시각 | `builtin_interfaces/Time` (ROS) |
| 각도(문서/로그) | degree |

CAD가 mm로 작성된 경우 `parts.yaml`의 `mesh_units: "mm"`로 선언하며 로드 시 m로 변환된다.

### 4.5 캘리브레이션 책임 — **합의 필요**

| 항목 | 제안 |
|---|---|
| 카메라 내부 파라미터 | 비전팀 (realsense-ros `CameraInfo` 사용) |
| 카메라↔로봇 외부 파라미터 (hand-eye) | **미정 — 협의 필요** |
| 캘리브레이션 갱신 주기·검증 방법 | **미정 — 협의 필요** |
| 캘리브레이션 오차의 최종 정확도 기여분 | **미정 — 협의 필요** |

> 이 항목이 미결인 상태에서는 §7의 정확도 수치가 **카메라 프레임 기준**임에 유의.
> 로봇 베이스 기준 정확도는 캘리브레이션 오차가 더해진다.

---

## 5. QoS 프로필

`/roboworld/part_result` 발행 측 설정 (`config/pipeline.yaml`의 `output_qos`):

| 항목 | 값 | 이유 |
|---|---|---|
| Reliability | `RELIABLE` | 결과 유실 불가 |
| Durability | `TRANSIENT_LOCAL` | 늦게 접속한 구독자도 마지막 결과 수신 |
| History | `KEEP_LAST` | |
| Depth | 10 | |

**구독 측은 호환 프로필을 사용해야 한다.** ROS 2 QoS 규칙상 `BEST_EFFORT` 구독자는
`RELIABLE` 발행자와 매칭되지 않아 **메시지를 한 건도 받지 못한다**(에러 없이 조용히).

권장 구독 코드:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
node.create_subscription(PartResult, "/roboworld/part_result", callback, qos)
```

---

## 6. 알려진 제약 — **로봇 부서 확인 필요**

### 6.1 자세 대칭 모호성 ⚠️ **중요 — 로봇 부서 확인 필수**

세 부품 모두 단면이 **55 × 55 mm 정사각형**이다. 가공 피처가 대칭을 깨긴 하지만,
그 차이가 ICP inlier 허용치(6 mm)보다 작으면 단일 시점 depth로는 구분할 수 없다.

**측정 결과** (`tools/check_symmetry.py`, `tools/evaluate.py`, 부품별 25프레임):

| 부품 | 자체 회전 chamfer | 구분 가능성 | **실측 자세 뒤집힘 빈도** |
|---|---|---|---|
| guide_block | 3.2 ~ 4.7 mm | 불가 (< 6 mm) | **21 / 25 (84 %)** |
| spacer_block | 3.2 ~ 3.9 mm | 불가 (< 6 mm) | **17 / 25 (68 %)** |
| end_stopper | 6.8 ~ 12.0 mm | 가능 (> 6 mm) | **3 / 25 (12 %)** |

"자세 뒤집힘"은 **위치는 정확하고(≤ 3.7 mm) 자세만 대칭군의 다른 원소로 보고된 경우**를
뜻한다. 형상적으로는 동일하지만 부품의 앞뒤/상하 라벨이 다르다.

대칭군 (order 8):
- 장축(+x) 기준 0° / 90° / 180° / 270° 회전
- 좌우 뒤집기 (+z 기준 180° 회전)

**위치(position)는 영향을 받지 않는다. 자세(orientation)만 해당된다.**

#### 로봇 부서에 필요한 결정

**Q. grasp이 이 대칭에 대해 불변인가?**
(예: 장축 중앙을 평행 그리퍼로 위에서 잡는 경우 대개 불변)

- **불변이다** → 추가 조치 불필요. 현재 정확도로 진행 가능.
- **불변이 아니다** → 부품별 대응이 다르다.
  - **end_stopper**: 형상만으로 구분 가능하므로 비전팀에서 개선 가능
    (피처 기반 자세 판별, 티켓 POSE-6). 12 % → 수 % 수준으로 낮출 여지 있음.
  - **guide_block / spacer_block**: **형상만으로는 원리적으로 불가능하다.**
    다음 중 하나를 선택해야 한다.
    - 부품에 비대칭 마커/각인 추가 (설계 변경)
    - 2번째 뷰(측면 카메라) 추가 (하드웨어 추가)
    - 컬러/텍스처 기반 판별 (부품에 구분 가능한 외관 차이가 있는 경우에 한함)

> 이 항목은 비전팀 단독으로 해결할 수 없다. **TL-10으로 회신을 요청한다.**

### 6.2 부품 식별 — 등록된 부품만 처리한다

프레임에서 **평면 위의 모든 물체를 찾아 치수를 측정**하고, 등록된 부품의 치수와
가장 잘 맞는 것을 선택한다. 아무것도 맞지 않으면 **처리하지 않는다.**

| 상황 | 발행 결과 (현재 구현) | 의도한 값 |
|---|---|---|
| 부품이 있고 치수가 맞음 | 정상 처리 (`STATUS_OK` / `STATUS_NG`) | 동일 |
| 부품 아닌 물체만 있음 (손, 공구, 다른 부품) | ⚠️ `STATUS_NG` | `STATUS_NO_POSE` |
| 평면 위에 아무것도 없음 | ⚠️ `STATUS_NG` | `STATUS_NO_POSE` |

> ⚠️ **미해결 불일치 (0.2.1에서 실측 확인).** 검사가 포즈보다 먼저 돌고, 분할이
> 아무것도 못 찾으면 검사 점수가 1.0으로 고정되어 **`STATUS_NG`가 발행된다.**
> "집을 것이 없다"가 "불량이다"로 보고되는 것이라, §3.1 권장 동작을 그대로 따르면
> **존재하지 않는 부품을 불량 배출**하게 된다.
> 잘못 집는 것보다는 안전한 방향이지만 **의도한 동작이 아니며 수정 예정이다.**
> 그때까지는 `message` 문자열로 구분할 것 (아래 예시 참조).

`message` 필드에 사유가 담긴다:

```
no object matches the registered part: closest is [404.0 176.0 60.0] mm
vs expected [200.0 55.0 55.0] mm (221% off, tolerance 25%)
```

**이 기능이 없을 때 어떤 일이 벌어지는지 실측했다**: 부품보다 큰 상자를 옆에 두자
그 상자를 부품으로 잡고 **`valid=True`로 150 mm 틀린 포즈를 발행**했다. 로봇이
그대로 집으러 갔을 상황이다. 허용 오차는 `pose.segmentation.size_tolerance`
(기본 25 %)로 조정한다.

> ⚠️ **부품이 다른 물체와 닿아 있으면** 하나의 덩어리로 병합되어 치수가 맞지 않아
> **전체가 거부**된다. 잘못된 포즈보다는 안전하나, 라인에서 부품 간 간격 확보가
> 필요하다.

### 6.2.1 스테이션 ROI — 정지 위치에 있는 부품만 처리한다

§6.2의 치수 식별은 **"이것이 맞는 부품인가"**에는 답하지만
**"이것이 카메라 앞에 정지한 그 부품인가"**에는 답하지 못한다.
벨트 위 다음 부품도 치수가 똑같이 맞기 때문이다.

**실측**: 부품 간격 200 mm에서 **후속 부품(중심 y = +195 mm)이 선택**되어
그 포즈가 발행되었다. 로봇은 정지 스테이션이 아닌 곳을 집으러 갔을 것이다.

그래서 후보의 중심이 **카메라 광학 좌표계 기준 상자** 안에 있어야 선택된다.

```yaml
pose:
  segmentation:
    station_roi:
      enabled: true
      center_m: [0.0, 0.0, 0.60]          # 정지 위치
      half_extents_m: [0.15, 0.15, 0.12]  # ±150 mm (부품 반길이 100 mm의 1.5배)
```

| 상황 | 발행 결과 (현재 구현) | 의도한 값 |
|---|---|---|
| 상자 안에 치수가 맞는 부품이 있음 | 정상 처리 (`STATUS_OK` / `STATUS_NG`) | 동일 |
| 부품이 있으나 전부 상자 밖 | ⚠️ `STATUS_NG` | `STATUS_NO_POSE` (= 아직 도착 안 함) |

> ⚠️ §6.2의 불일치가 이 경우에도 그대로 적용된다. 사유는 `message`로 구분된다.

`message` 예시 — §6.2의 치수 불일치와 **사유가 구분된다**:

```
no object inside the station volume: best of 2 is 45 mm outside,
centred at [0. 195. 549.] mm
```

> ⚠️ **좌표계 주의**: 이 상자는 카메라 좌표계 기준이라 **카메라를 옮기면 같이
> 깨진다.** 이미지 사각형 대신 3D 상자를 쓴 이유는 컨베이어 도면의 mm 값을 그대로
> 넣을 수 있고 해상도·렌즈 변경에 불변이기 때문이지, 카메라 이동에 견고해서가
> 아니다. 진짜 불변인 ROI는 station 프레임 + TF가 필요하며 **§4 미결 사항**이다.

> ❗ **검사 ROI와 혼동 금지.** EfficientAD가 점수를 매기는 픽셀 영역은 depth
> 분할 마스크(부품 실루엣)이며, 여기서 정의하는 상자와 무관하다.

> 🔧 **벨트 진행 방향을 알려주면** 그 축만 좁게 잡아 더 안전하게 만들 수 있다
> (§8 미결 항목).

### 6.3 작업 거리 — **중요**

포즈 정확도는 거리에 강하게 의존한다. depth 노이즈가 거리 제곱에 비례하기 때문이다.

| 카메라~부품 거리 | 위치오차 (중앙/최대) | 자세오차 | 판정 |
|---|---|---|---|
| 0.4 m | 0.7 / 2.7 mm | 0.3° | ✅ |
| 0.6 m | 0.6 / 1.9 mm | 0.3° | ✅ **권장** |
| 0.8 m | 1.3 / 2.8 mm | 0.6° | ✅ |
| 1.0 m | 1.9 / 2.5 mm | 1.2° | ⚠️ 사용 가능 |
| 1.25 m | 6.1 / 9.8 mm | 9.5° | ❌ 정밀 파지 부적합 |
| 1.5 m | 14.2 / 19.3 mm | 3.1° | ❌ |

**설계 작업 거리는 0.6 m이며, 0.8 m 이내를 권장한다.**

기본 설정은 이 범위를 벗어난 포즈를 자동 거부한다:

```yaml
pose:
  valid_z_range_m: [0.30, 1.00]     # 이 밖의 z 는 STATUS_NO_POSE
  max_lateral_offset_m: 0.25
```

1 m 이상에서 운용하려면 위 창과 함께
`pose.segmentation.min_height_above_plane_m`(기본 8 mm)를 depth 노이즈의 약 3배로
올려야 한다. 다만 그보다 얇은 부품은 검출되지 않는다.

> 1280×720 해상도로 올리면 픽셀 수가 4배가 되어 원거리 성능이 개선되나,
> depth 노이즈의 거리 제곱 특성 자체는 변하지 않는다.

### 6.4 기타 제약

| 항목 | 내용 |
|---|---|
| 동시 처리 | 사이클 1건씩 (index 방식). 이전 사이클 진행 중 트리거는 무시되고 경고 로그 |
| 다중 부품 | 프레임 내 1개만 처리. 여러 개면 §6.2.1의 스테이션 ROI 안에서 §6.2의 **치수가 맞는 것**을 선택 |
| 가림(occlusion) | 미지원. 가려지면 치수 불일치로 `STATUS_NO_POSE` |
| 트리거 디바운스 | 300 ms (`pipeline.trigger_debounce_s`) |
| FoundationPose 라이선스 | 공개판은 **비상업(NC)**. 상용화 시 NGC 상업판 전환 필요 |

---

## 7. 성능 (모의 데이터 측정치)

> **주의**: 아래 수치는 CAD 렌더링 기반 모의 데이터에서 CPU 백엔드
> (`statistical` + `icp`)로 측정한 값이다. 실 카메라·EfficientAD·FoundationPose
> 전환 후 반드시 재측정한다. 재측정 명령: `python3 tools/evaluate.py --part all`

### 7.1 검사 (부품별 양품 25 / 불량 25, 학습 40프레임)

| 부품 | 미검출(불량→OK) | 과검출(양품→NG) | 추론 시간 |
|---|---|---|---|
| guide_block | 0 / 25 | 1 / 25 | 62 ms |
| spacer_block | 0 / 25 | 1 / 25 | 62 ms |
| end_stopper | 0 / 25 | 1 / 25 | 62 ms |

- **미검출 0건** (불량이 양품으로 통과된 사례 없음)
- 과검출 4 %는 배치 범위 극단(±35 mm, ±180° yaw)까지 포함한 모의 조건 기준이며,
  실제 정지 스테이션의 기계적 스토퍼는 이보다 훨씬 좁은 반복도를 가진다.

### 7.2 포즈 (부품별 25프레임, 대칭 보정 후, **작업 거리 0.6 m**)

| 부품 | 위치오차 중앙값 | 위치오차 최대 | 자세오차 중앙값 | 자세오차 최대 | 추론 시간 |
|---|---|---|---|---|---|
| guide_block | 1.36 mm | 3.69 mm | 0.72° | 1.78° | 104 ms |
| spacer_block | 0.95 mm | 2.52 mm | 0.58° | 2.29° | 102 ms |
| end_stopper | 0.93 mm | 2.40 mm | 0.52° | 1.83° | 112 ms |

포즈 거부(`STATUS_NO_POSE`) 0건. **거리에 따른 변화는 §6.3 참조** — 위 수치는
설계 작업 거리 0.6 m 기준이며, 1.25 m에서는 위치오차가 6 mm 이상으로 커진다.

> **자세오차는 §6.1의 대칭군으로 보정한 값이다.** 보정 전 원시 자세오차는
> guide_block 84 %, spacer_block 68 %, end_stopper 12 %의 프레임에서 약 180°이다.
> **grasp이 대칭 불변이 아니라면 위 자세오차 수치를 그대로 쓸 수 없다.**

### 7.3 택트 타임

| 구간 | 예산 | 모의 측정 |
|---|---|---|
| 캡처 | 120 ms | (모의 렌더링 ~150 ms — 실 카메라에서는 해당 없음) |
| 검사 | 400 ms | 62 ms |
| 포즈 | 900 ms | ~110 ms |
| **전체 (경고)** | 1200 ms | **중앙값 258 ms** |
| 전체 (하드 리밋) | 5000 ms | 초과 시 `STATUS_ERROR` |

---

## 8. 변경 관리

1. 본 문서와 `roboworld_interfaces` 패키지는 **비전팀 Team Lead가 단독 관리**한다.
2. 스키마 변경 시:
   - `pipeline_version` 상향
   - 본 문서 개정 및 로봇 부서 통지
   - 계약 테스트(`roboworld_core/test/test_contract.py`) 갱신
   - 양 팀 합의 후 머지
3. 필드 **추가**는 하위 호환으로 간주하나, ROS 2 메시지는 필드 추가 시에도 타입이
   달라지므로 양측 동시 재빌드가 필요하다.
4. 필드 **삭제/의미 변경**은 반드시 버전 상향 + 사전 합의.

### 미결 항목 체크리스트

- [ ] §4.1 A: `world` 프레임 원점 정의 (로봇 부서)
- [ ] §4.1 B: 카메라 외부 파라미터 / hand-eye 캘리브레이션 책임 주체
- [ ] §4.5: 캘리브레이션 갱신 주기 및 검증 방법
- [ ] **§6.1: grasp의 대칭 불변성 확인** → 불변이 아니면 대응 방식 결정 (최우선)
- [ ] §6.3: 실제 카메라 설치 거리 확정 (권장 0.6 m, 상한 0.8 m)
- [ ] §6.2.1: 벨트 진행 방향 (카메라 화면의 가로 / 세로) → 스테이션 ROI 비대칭 설정
- [ ] §6.2.1: 정지 위치의 실측 좌표 → `station_roi.center_m` 확정
- [ ] **§10: 시스템 구성 A(PC 2대) / B(PC 1대) 선택** → 전달물이 달라짐
- [ ] §10: `ROS_DOMAIN_ID` 값 합의 (구성 A인 경우)
- [ ] 요구 택트 타임 및 정확도 목표치 (현재 예산은 비전팀 가정값)
- [ ] `STATUS_NO_POSE` / `STATUS_ERROR` 발생 시 라인 처리 절차
- [ ] **§6.2 불일치 수정**: 부품 미검출 시 `STATUS_NG` → `STATUS_NO_POSE`
      (비전팀 작업. 로봇 부서는 "부품 없음"을 불량 배출로 처리할지 재트리거로
      처리할지만 알려주면 된다)

---

## 9. 구독 예제

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from roboworld_interfaces.msg import PartResult


class PartResultSubscriber(Node):
    def __init__(self):
        super().__init__("robot_part_consumer")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(PartResult, "/roboworld/part_result", self.on_result, qos)

    def on_result(self, msg: PartResult):
        # status 만으로 분기한다 (ICD 3.1)
        if msg.status == PartResult.STATUS_OK:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            frame = msg.pose.header.frame_id      # 하드코딩 금지 (ICD 4.2)
            self.get_logger().info(
                f"[{msg.sequence}] {msg.part_id} PICK "
                f"pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) m in '{frame}'"
            )
            # → TF로 로봇 베이스 프레임 변환 후 pick 수행
        elif msg.status == PartResult.STATUS_NG:
            self.get_logger().info(f"[{msg.sequence}] REJECT score={msg.anomaly_score:.3f}")
        else:
            self.get_logger().warning(f"[{msg.sequence}] HOLD: {msg.message}")


def main():
    rclpy.init()
    rclpy.spin(PartResultSubscriber())
```

---

## 10. 시스템 구성 — **로봇 부서 선택 필요**

ROS 2 토픽은 네트워크 메시지 버스다. 발행자가 채널에 내보내면, 같은 네트워크의
구독자가 받는다. 따라서 **비전 프로그램과 로봇 프로그램은 같은 PC에 있어도 되고
다른 PC에 있어도 된다.** 어느 쪽인지에 따라 전달물이 달라진다.

### 구성 A — PC 2대 (권장)

```
   [비전 PC]                                      [로봇 PC]
   D455 + RTX 5070                                로봇 제어
   검사 + 6D 포즈  ──→ /roboworld/part_result ──→  모션 계획
                          (이더넷 / DDS)
```

| 항목 | 내용 |
|---|---|
| **로봇 부서에 전달** | `roboworld_interfaces` 패키지 + 본 문서 |
| 로봇 PC 요구사항 | ROS 2 Humble만 (GPU·카메라 불필요) |
| 장점 | GPU·카메라 요구사항이 로봇 PC로 넘어가지 않음 |

우리 알고리즘 코드는 전달할 필요가 없다. 네트워크로 오는 것은 바이트열이므로
**받는 쪽도 동일한 메시지 정의**만 있으면 해독된다.

### 구성 B — PC 1대

| 항목 | 내용 |
|---|---|
| **로봇 부서에 전달** | 저장소 전체 |
| 로봇 PC 요구사항 | ROS 2 + CUDA 12.x + RTX 5070급 GPU + D455 |
| 장점 | 네트워크 불필요, 지연 최소 |

### 구성 C — 통합 전 선행 개발 (구성 무관)

스테이션 완성 전에 픽 로직을 개발하려면 저장소 전체를 받아 **목업 모드**로 실행한다.
카메라·GPU 없이 `/roboworld/part_result`가 실제로 발행된다.

```bash
ros2 launch roboworld_bringup pipeline.launch.py
```

### 네트워크 요건 (구성 A)

| 항목 | 값 |
|---|---|
| `ROS_DOMAIN_ID` | 양쪽 PC 동일해야 함 (미설정 시 기본 0) |
| 네트워크 | 같은 서브넷. DDS 자동 탐색에 멀티캐스트 필요 |
| 방화벽 | UDP 7400~7500 대역 허용 |

### ⚠️ 메시지 정의 버전 일치

**양쪽의 `.msg` 정의가 한 글자라도 다르면 ROS 2는 서로 다른 타입으로 간주하여,
에러 없이 조용히 아무것도 주고받지 못한다.** 가장 찾기 어려운 장애 유형이다.

- `roboworld_interfaces`는 반드시 **동일한 git 커밋**에서 빌드한다
- 수신한 메시지의 `pipeline_version`을 확인하고, 예상과 다르면 경고를 남길 것
- 인터페이스 변경 시 §8 절차를 따르고 **양측이 동시에 재빌드**한다

---

## 11. 연동 확인 절차

로봇 부서에서 하드웨어 없이 인터페이스를 검증하는 방법:

```bash
# 1. 워크스페이스 빌드
cd ros2_ws && colcon build --symlink-install && source install/setup.bash

# 2. 모의 데이터로 전체 파이프라인 구동 (카메라·GPU 불필요)
ros2 launch roboworld_bringup pipeline.launch.py

# 3. 계약 토픽 확인
ros2 topic echo /roboworld/part_result
ros2 topic info /roboworld/part_result --verbose   # QoS 호환성 확인
ros2 interface show roboworld_interfaces/msg/PartResult
```

3초 간격으로 `STATUS_OK` / `STATUS_NG`가 번갈아 발행된다.
