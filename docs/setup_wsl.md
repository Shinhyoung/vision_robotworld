# WSL2 개발 환경 구축 절차

Windows + WSL2 Ubuntu 22.04 + ROS 2 Humble + RTX 5070(Blackwell) + RealSense D455

> claude.md §1의 고정 스택을 그대로 따른다. 버전을 임의로 바꾸지 말 것.
> Ubuntu 24.04 → 22.04 재구성이 필요한 이유는 Humble/Isaac ROS 호환성이다.

---

## 0. 요약 체크리스트

| # | 항목 | 확인 명령 |
|---|---|---|
| 1 | WSL2 Ubuntu 22.04 | `lsb_release -a` → `22.04` |
| 2 | Windows NVIDIA 드라이버 (Blackwell 지원) | `nvidia-smi` (WSL 안에서) |
| 3 | WSL용 CUDA Toolkit 12.x | `nvcc --version` |
| 4 | PyTorch sm_120 빌드 | §3.3 스니펫 |
| 5 | ROS 2 Humble | `ros2 --version` |
| 6 | colcon 워크스페이스 빌드 | `colcon build` |
| 7 | usbipd-win으로 D455 attach | `rs-enumerate-devices` |

**하드웨어가 없어도 1·5·6만으로 전체 파이프라인이 동작한다.** 2·3·4·7은 EfficientAD /
FoundationPose / 실 카메라를 쓸 때만 필요하다.

---

## 1. WSL2 Ubuntu 22.04 재구성

### 1.1 기존 24.04 배포판 정리 (PowerShell, 관리자)

```powershell
wsl --list --verbose
# 필요한 데이터를 먼저 백업할 것
wsl --export Ubuntu-24.04 D:\backup\ubuntu2404.tar   # 선택
wsl --unregister Ubuntu-24.04                        # 되돌릴 수 없음
```

> **주의**: `--unregister`는 해당 배포판의 파일 시스템을 완전히 삭제한다.
> 실행 전 `\\wsl$\Ubuntu-24.04\home\<user>` 를 열어 필요한 것을 옮겨두자.

### 1.2 22.04 설치

```powershell
wsl --update
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
wsl --set-default Ubuntu-22.04
```

### 1.3 기본 설정 (WSL 셸)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget python3-pip python3-venv \
                    software-properties-common lsb-release gnupg2
lsb_release -a          # Ubuntu 22.04 확인
```

### 1.4 `.wslconfig` (Windows 사용자 폴더 `C:\Users\<user>\.wslconfig`)

```ini
[wsl2]
memory=24GB           # 물리 메모리의 50~75% 권장
processors=8
swap=8GB
# GPU/CUDA 사용 시 필요
gpuSupport=true
```

변경 후 `wsl --shutdown` 으로 재시작.

---

## 2. VSCode + WSL Remote

1. Windows에 VSCode 설치
2. 확장 **WSL** (`ms-vscode-remote.remote-wsl`) 설치
3. WSL 셸에서 `code .` 실행 → 자동 연결
4. 권장 확장: Python, Pylance, ROS (`ms-iot.vscode-ros`), YAML

> 프로젝트는 반드시 **WSL 파일 시스템**(`/home/...`)에 둘 것.
> `/mnt/c/...` 는 I/O가 수십 배 느려 colcon 빌드가 크게 지연된다.

---

## 3. GPU: 드라이버 / CUDA / PyTorch

### 3.1 ⚠️ 가장 흔한 실수

**WSL 안에 리눅스용 NVIDIA 드라이버를 설치하면 안 된다.**
GPU는 Windows 드라이버가 제공하고, WSL은 CUDA 툴킷만 설치한다.

```bash
# 절대 실행하지 말 것
# sudo apt install nvidia-driver-***
```

### 3.2 Windows 드라이버 (Blackwell / RTX 50 시리즈)

Windows에서 NVIDIA 최신 Game Ready 또는 Studio 드라이버 설치
(RTX 5070은 Blackwell, sm_120 — 구버전 드라이버는 인식하지 못한다).

WSL에서 확인:

```bash
nvidia-smi     # GPU 이름과 드라이버 버전이 보이면 정상
```

### 3.3 WSL용 CUDA Toolkit 12.x

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6      # 12.x 계열

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

### 3.4 PyTorch (sm_120 지원 필수)

RTX 5070은 compute capability **12.0**이다. sm_120 커널이 없는 휠을 설치하면
`no kernel image is available for execution on the device` 로 실패한다.

```bash
python3 -m pip install --upgrade pip
# CUDA 12.x 대응 휠. 실패 시 nightly 채널 사용
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

검증:

```bash
python3 - <<'EOF'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")
print("compiled archs:", torch.cuda.get_arch_list())
EOF
```

`capability`가 `(12, 0)`이고 `compiled archs`에 `sm_120`이 있어야 한다.
없으면 nightly 설치:

```bash
python3 -m pip install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu126
```

---

## 4. RealSense D455 — usbipd-win으로 WSL에 연결

WSL2는 기본적으로 USB 장치를 인식하지 못한다.

### 4.1 Windows 측 (관리자 PowerShell)

```powershell
winget install --interactive --exact dorssel.usbipd-win
# 재부팅 또는 새 PowerShell 세션

usbipd list
# BUSID  VID:PID     DEVICE
# 2-3    8086:0b5c   Intel(R) RealSense(TM) Depth Camera 455

usbipd bind   --busid 2-3       # 최초 1회 (관리자 필요)
usbipd attach --wsl --busid 2-3 # WSL 실행 중일 때마다
```

> D455는 여러 USB 인터페이스를 노출한다. `usbipd list`에 복수 항목이 보이면
> 모두 attach 해야 한다. **USB 3.0 포트에 직결**할 것 (허브·USB2 포트에서는
> 대역폭 부족으로 스트림이 끊긴다).

분리:

```powershell
usbipd detach --busid 2-3
```

### 4.2 WSL 측

```bash
sudo apt install -y linux-tools-generic hwdata
lsusb | grep -i intel      # 8086:0b5c 확인
```

### 4.3 librealsense + realsense-ros

```bash
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | \
    sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt update
sudo apt install -y librealsense2-utils librealsense2-dev

rs-enumerate-devices          # 장치가 보이면 성공
sudo apt install -y ros-humble-realsense2-camera
```

### 4.4 WSL2에서 안 될 때 — rosbag 우회 (권장 개발 방식)

USB 패스스루는 커널 버전에 따라 불안정할 수 있다. **claude.md §1이 명시한 대로,
개발은 rosbag 재생 또는 목업으로 진행하고 통합 단계에서 실물로 교체한다.**

Windows 또는 별도 리눅스 PC에서 1회 녹화:

```bash
ros2 launch roboworld_bringup realsense.launch.py
ros2 bag record -o station_capture \
    /camera/camera/color/image_raw \
    /camera/camera/aligned_depth_to_color/image_raw \
    /camera/camera/color/camera_info \
    /roboworld/trigger
```

WSL에서 재생:

```bash
ros2 launch roboworld_bringup rosbag_replay.launch.py bag:=/path/to/station_capture
```

**녹화 없이도 개발 가능**하다 — `camera.source: mock`이 CAD에서 RGB-D를 렌더링한다.

---

## 5. ROS 2 Humble 설치

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-vcstool
sudo rosdep init && rosdep update

echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc
ros2 --version
```

### 5.1 필수 ROS 패키지

```bash
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-message-filters \
    ros-humble-tf2-ros \
    ros-humble-diagnostic-msgs \
    ros-humble-rosbag2-storage-mcap
```

---

## 6. 워크스페이스 빌드

```bash
cd ~/RoboWorld_Demo/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install`은 파이썬 파일 수정 시 재빌드 없이 반영되게 한다(개발 편의).
`msg`/`srv` 변경 시에는 반드시 재빌드해야 한다.

### 6.1 동작 확인 (하드웨어 없이)

```bash
ros2 launch roboworld_bringup pipeline.launch.py
# 다른 터미널
ros2 topic echo /roboworld/part_result
```

### 6.2 ROS 없이 확인 (빌드 전에도 가능)

```bash
cd ~/RoboWorld_Demo
python3 -m pip install -r requirements.txt
python3 tools/e2e_dryrun.py
python3 -m pytest ros2_ws/src/roboworld_core/test -q
```

---

## 7. 선택: anomalib (EfficientAD)

```bash
python3 -m pip install -r requirements-gpu.txt
python3 -c "import anomalib; print(anomalib.__version__)"
```

학습:

```bash
python3 tools/train_inspection.py --part guide_block --backend efficientad
```

---

## 8. 선택: Isaac ROS FoundationPose

> ⚠️ **라이선스**: FoundationPose 공개판은 **비상업(NC)** 라이선스이다.
> 데모/평가 목적에 한해 사용하고, 상업화 시 NGC 상업판으로 전환해야 한다 (claude.md §1).

Isaac ROS는 컨테이너 실행을 전제로 한다.

```bash
sudo apt install -y nvidia-container-toolkit
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
cd isaac_ros_common && ./scripts/run_dev.sh
# 컨테이너 내부
sudo apt install -y ros-humble-isaac-ros-foundationpose
```

CAD 준비: FoundationPose는 OBJ/PLY 메시와 텍스처를 요구한다.
`01_input/*.ply` 를 그대로 사용할 수 있으나, 단위가 미터인지 확인할 것
(본 저장소의 메시는 미터 기준, 200×55×55 mm).

---

## 9. 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| `nvidia-smi` 실패 | Windows 드라이버 미설치 또는 구버전. WSL 안에 리눅스 드라이버를 설치했다면 제거 |
| `no kernel image is available` | PyTorch가 sm_120 미지원. §3.4 nightly 설치 |
| `rs-enumerate-devices` 빈 출력 | usbipd attach 누락, 또는 USB2 포트 연결. §4.1 |
| depth가 전부 0 | `align_depth.enable`이 false. ICD는 정합된 depth를 요구 |
| 토픽이 안 보임 | QoS 불일치. ICD §5 확인. `ros2 topic info --verbose` |
| colcon 빌드가 매우 느림 | 프로젝트가 `/mnt/c` 에 있음 → `/home` 으로 이동 |
| `roboworld_interfaces` import 실패 | `source install/setup.bash` 누락, 또는 msg 변경 후 미재빌드 |
| 노드가 서로 못 찾음 | `ROS_DOMAIN_ID` 불일치. 모든 터미널에서 동일하게 설정 |

---

## 10. 환경 변수 (선택)

```bash
export ROS_DOMAIN_ID=42               # 같은 네트워크의 다른 팀과 충돌 방지
export RCUTILS_COLORIZED_OUTPUT=1
export ROBOWORLD_CONFIG_DIR=~/RoboWorld_Demo/ros2_ws/src/roboworld_bringup/config
export ROBOWORLD_ASSETS_DIR=~/RoboWorld_Demo/01_input
export ROBOWORLD_DATA_DIR=~/RoboWorld_Demo/data
```

`ROBOWORLD_*` 변수는 설정하지 않아도 자동 탐색되므로 보통 불필요하다.
컨테이너나 비표준 배치에서만 사용한다.
