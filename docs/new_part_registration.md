# CAD 없이 새 부품 등록하기

D455 촬영만으로 새 부품을 등록하고 **불량 검사와 6D 포즈**를 테스트하는 절차.

---

## 0. 어떤 경로를 쓸 것인가

| 목표 | 필요한 것 | 절차 |
|---|---|---|
| **불량 검사만** | 정상품 사진 | §2 촬영 → §3 등록 |
| **불량 검사 + 6D 포즈** | 정상품 사진 + **복원 메시** | §5 복원 → §2 → §3 |

---

## 1. 먼저 알아둘 것 — 무엇이 되고 무엇이 안 되는가

| 기능 | CAD 없이 | 이유 |
|---|---|---|
| **불량 검사 (OK/NG)** | ✅ **동작** | 검사기는 형상을 쓰지 않는다. 정상품 이미지만으로 학습 |
| **부품 분할 (ROI)** | ✅ **동작** | depth로 평면 제거 후 그 위 물체를 잡는다. 형상 무관 |
| **6D 포즈** | ⚠️ **메시 복원 후 동작** | 정합할 모델이 필요 → **§5에서 D455로 복원** |
| 목업 렌더링 | ❌ 불가 | CAD에서 렌더링하므로 |

메시 없이 등록한 부품은 파이프라인을 **OK/NG 판정까지 정상 통과**하고, 그 이후는
`STATUS_NO_POSE`로 보고된다. ICD §3.1이 이미 정의한 "양품이나 포즈 없음 → 집지 말 것"
상태이므로 **로봇 부서가 새로 처리할 케이스는 없다.**

---

## 2. 촬영

```bash
# 카메라를 WSL에 연결 (최초 1회 / 재부팅 후)
"/mnt/c/Program Files/usbipd-win/usbipd.exe" list          # BUSID 확인
"/mnt/c/Program Files/usbipd-win/usbipd.exe" attach --wsl --busid 8-1

# 정상품 촬영
python3 tools/capture_part.py --part my_part
```

실시간 창에 **컬러 | 깊이 | 분할** 3개 패널이 뜬다. 세 번째 패널에서 부품이
초록색으로 잡히는지 확인한 뒤 촬영한다.

| 키 | 동작 |
|---|---|
| `SPACE` / `c` | 현재 프레임 촬영 |
| `a` | 자동 촬영 on/off (`--interval` 초마다) |
| `q` / `ESC` | 종료 |

HUD 상태 표시:

- `READY` — 촬영 가능
- `NO PART FOUND` — 평평한 면 위에 놓고 카메라를 맞출 것
- `SEGMENT TOO LARGE` — 배경이 잡힘. 카메라를 더 가까이

### 촬영 요령 ⚠️

- **정상품만 찍는다.** 검사기는 비지도 학습이라 불량 1장이 섞이면
  그 불량을 "정상"으로 배운다
- **라인에서 변하는 것을 변화시켜 찍는다** — 위치, 회전, 어느 면이 위인지.
  모델은 본 적 있는 변화만 견딘다
- **작업 거리를 유지**하고 평평한 면 위에 놓는다
- **40장 이상** 권장 (최소 10장)

불량 샘플은 검증용으로 따로 찍는다(학습에는 쓰지 않는다):

```bash
python3 tools/capture_part.py --part my_part --defect
```

---

## 3. 등록

```bash
python3 tools/register_part.py --part my_part --evaluate
```

수행 내용:

1. 촬영본으로 검사 모델 학습 → `data/models/statistical_my_part.npz`
2. `config/parts_local.yaml`에 부품 항목 추가
3. `--evaluate` 지정 시 정상 촬영본의 20 %를 떼어 점수 분포 확인

출력 예:

```
part 'my_part': 45 normal, 8 defect captures
  holding out 9 frames for scoring
  fitted on 36 frames -> data/models/statistical_my_part.npz
  threshold 0.500

  normal   n=9   score median=0.118 max=0.331  false rejects=0
  defect   n=8   score median=0.912 min=0.706  missed=0
  margin   +0.375  (양호)
```

**margin을 반드시 볼 것.** 0에 가까우면 실 데이터에서 곧 깨진다
(architecture.md §8.1의 end_stopper 사례 참조).

`parts_local.yaml`은 생성 파일이며, 주석이 있는 `parts.yaml`을 건드리지 않기 위해
분리되어 있다. 로드 시 자동 병합된다.

---

## 4. 검출 테스트

```bash
python3 tools/live_view.py --source realsense --part my_part --inspect
```

실시간으로 OK/NG 판정과 이상맵이 표시된다. `s`로 스냅샷 저장.

파이프라인 전체로 확인하려면:

```bash
ros2 launch roboworld_bringup pipeline.launch.py part_id:=my_part
ros2 topic echo /roboworld/part_result
```

CAD가 없으므로 `status: 2` (`STATUS_NO_POSE`)가 발행되고, `is_good`과
`anomaly_score`는 정상적으로 채워진다.

---

## 5. 6D 포즈까지 하려면 — D455로 메시 복원

```bash
python3 tools/reconstruct_part.py --part my_part      # ① depth로 메시 복원
python3 tools/capture_part.py     --part my_part      # ② 검사용 정상품 촬영
python3 tools/register_part.py    --part my_part --mesh data/meshes/my_part.ply
python3 tools/live_view.py --source realsense --part my_part --inspect --pose
```

### 복원 원리와 정확도

부품이 평면 위에 놓인다는 점을 이용해, 평면 좌표계에서 **높이맵을 만들고 아래로
압출(extrude)**한다. depth는 무늬가 필요 없으므로 무텍스처 부품에 오히려 유리하고,
**실척이 그대로 나온다**(단안 사진 복원에는 없는 장점).

측정값 (200×55×55 mm 블록, 0.6 m 거리, 5뷰 병합, 2 mm 격자):

| 항목 | 결과 |
|---|---|
| 복원 치수 오차 | **+2.0 / +1.0 / +1.1 mm** (약 1 %) |
| 복원 메시로 포즈 추정 | 위치오차 중앙값 **1.1 ~ 1.9 mm** |
| 같은 조건 **실제 CAD** 사용 시 | 1.6 mm |

**복원 메시가 CAD와 동등한 포즈 정확도**를 낸다 (`test_reconstruct.py`가 이를 고정).

### 촬영 방법 ⚠️ **오해하기 쉬운 부분**

| | |
|---|---|
| ✅ **맞음** | 카메라 **고정** + 부품 **고정** → SPACE 를 3~5회 |
| ❌ **틀림** | 부품을 회전시키며 각도별로 촬영 |

**여러 뷰는 depth 노이즈를 평균낼 뿐, 형상을 추가하지 않는다.** 뷰들은 첫 뷰의 평면
좌표계에 그대로 겹쳐 쌓이므로, 부품이 움직이면 다른 위치에 쌓여 메시가 뭉개진다.
(실제로 26뷰를 회전 촬영했더니 200 mm급 부품이 338×322 mm로 복원되었다.)

도구가 두 단계로 막는다:

1. **촬영 중** — 첫 뷰 대비 5 mm 이상 움직이면 그 뷰를 **거부**하고 HUD에 `drift` 표시
2. **복원 시** — 뷰 간 중심 산포·외곽 편차를 계산해 불일치하면 경고

회전시켜 **모든 면**을 담으려면 뷰 간 정합(registration)이 필요한데, 이는 BundleSDF급
문제라 구현되어 있지 않다. 대신 바닥면이 평평하다는 가정으로 압출해 해결한다.

촬영본은 `data/captures/<part>/views/` 에 저장되므로 **재촬영 없이** 다시 복원할 수 있다:

```bash
python3 tools/reconstruct_part.py --part my_part --from-saved --single-view 0
python3 tools/reconstruct_part.py --part my_part --from-saved --cell-size-mm 1
```

### 가정과 한계 ⚠️

- **바닥면이 평평**해야 한다 → 접촉면이므로 보통 성립
- **언더컷 불가** — 위에서 안 보이는 아래쪽은 채워진 것으로 복원된다
- **기울어져 놓이면 안 된다** — 기울기가 메시에 쐐기 모양으로 굳는다.
  도구가 윗면 기울기를 측정해 2° 초과 시 경고한다
- **각기둥형(prismatic) 부품에 정확**하다. 유기적 형상에는 부적합

### 다른 메시 확보 방법

| 방법 | 비고 |
|---|---|
| **CAD 파일 입수** | 가장 정확. 가능하면 이것 |
| **D455 복원** (위) | 무텍스처 부품에 유리, 실척 확보 |
| 스마트폰 스캔 | NVIDIA 권장. iPhone 12 Pro 이상 → `.usdz` → `.obj` (변환 필요) |
| 사진 측량 | **무텍스처·단색 부품에는 부적합.** 특징점 매칭 실패 |

### 주의사항

- **원점이 메시 중심**이어야 한다. 모서리에 있으면 검출 위치가 어긋난다
  (FoundationPose 요구사항). 우리 로더는 로드 시 AABB 중심으로 재정렬한다
- **실척(metric scale)** 확인. 단안 사진 복원에는 절대 크기가 없다.
  `mesh_units`로 보정한다
- 현재 로더는 **PLY만** 지원한다. `.obj`는 변환이 필요하다
- 준대칭 부품이면 `symmetry` 항목을 채운다 — 근거는 `tools/check_symmetry.py`

---

## 6. 참고: model-free 6D 포즈

CAD 없이 참조 이미지만으로 포즈를 추정하는 연구가 있다.

- **FoundationPose model-free** — 소수의 참조 RGB-D + 카메라 상대 포즈,
  BundleSDF로 Neural Object Field 학습
- **Any6D** (CVPR 2025) — RGB-D 1장으로 포즈와 실척 동시 추정

다만 **`isaac_ros_foundationpose`는 메시(.obj)가 필수**이며 model-free 경로를
노출하지 않는다. claude.md §1이 Isaac ROS를 고정 스택으로 정한 이상,
현실적인 경로는 **"스캔 → 메시 → 기존 파이프라인"** 이다. 이 경우 다운스트림이
전혀 바뀌지 않는다는 장점도 있다.
