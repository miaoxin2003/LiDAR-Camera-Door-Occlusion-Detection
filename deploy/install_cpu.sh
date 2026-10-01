#!/usr/bin/env bash
# Run as the normal ROS user: bash deploy/install_cpu.sh
set -eo pipefail

fusion_bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fusion_workspace="$HOME/fusion_ws"
fusion_venv="$HOME/venvs/fusion_cpu"

if [[ ! -f /opt/ros/noetic/setup.bash ]] || [[ "$(uname -m)" != x86_64 ]]; then
    echo '此安装脚本适用于 Ubuntu 20.04 / ROS Noetic / x86_64。当前系统不匹配。'
    exit 1
fi
if ! /usr/bin/python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 8) else 1)'; then
    echo '此依赖组合要求系统 Python 3.8；请先确认小车系统版本。'
    exit 1
fi
if [[ ! -f "$fusion_bundle/image_pointcloud_fusion/package.xml" ]] || \
   [[ ! -f "$fusion_bundle/best.pt" ]]; then
    echo '缺少项目或当前门框 YOLO 权重 best.pt，请先完整解压部署包。'
    exit 1
fi

echo '[1/3] 安装 ROS 和系统依赖（可能需要输入 sudo 密码）'
source /opt/ros/noetic/setup.bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake python3-venv python3-pip \
    python3-numpy python3-opencv python3-yaml python3-rospkg \
    python3-catkin-pkg python3-empy \
    ros-noetic-catkin ros-noetic-rospy ros-noetic-sensor-msgs ros-noetic-std-msgs \
    ros-noetic-cv-bridge ros-noetic-message-filters \
    ros-noetic-geometry-msgs ros-noetic-visualization-msgs \
    ros-noetic-jsk-recognition-msgs ros-noetic-rviz \
    libgl1 libglib2.0-0 libgomp1

echo '[2/3] 安装 CPU Python 环境（首次需要下载 PyTorch / Open3D）'
/usr/bin/python3 -m venv --system-site-packages "$fusion_venv"
"$fusion_venv/bin/python" -m pip install --upgrade \
    pip==25.0.1 setuptools==69.5.1 wheel==0.45.1
"$fusion_venv/bin/python" -m pip install numpy==1.24.4
"$fusion_venv/bin/python" -m pip install \
    torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
"$fusion_venv/bin/python" -m pip install \
    -r "$fusion_bundle/deploy/requirements-noetic-cpu.txt" \
    -c "$fusion_bundle/deploy/constraints-noetic-cpu.txt"

echo '[3/3] 安装项目并编译'
fusion_package="$fusion_workspace/src/image_pointcloud_fusion"
mkdir -p "$fusion_package"
fusion_saved_config=''
if [[ -f "$fusion_package/config/c32_oak.yaml" ]]; then
    mkdir -p "$fusion_workspace/config_backups"
    fusion_saved_config="$(mktemp --suffix=.yaml "$fusion_workspace/config_backups/c32_oak_XXXXXX")"
    cp "$fusion_package/config/c32_oak.yaml" "$fusion_saved_config"
fi
if [[ "$(realpath "$fusion_bundle/image_pointcloud_fusion")" != "$(realpath "$fusion_package")" ]]; then
    cp -a "$fusion_bundle/image_pointcloud_fusion/." "$fusion_package/"
fi
if [[ -n "$fusion_saved_config" ]]; then
    cp "$fusion_saved_config" "$fusion_package/config/c32_oak.yaml"
    echo "已保留现有 c32_oak.yaml，备份位于 $fusion_saved_config"
fi
chmod +x "$fusion_package/scripts/"*.py
cd "$fusion_workspace"
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

echo ''
echo '安装完成。先按原来的方式启动相机和雷达驱动，再在小车桌面终端执行：'
printf 'bash "%s/deploy/run_live.sh"\n' "$fusion_bundle"
