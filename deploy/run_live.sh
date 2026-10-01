#!/usr/bin/env bash
# Start detection and RViz against the live driver's ROS master.
set -eo pipefail

fusion_bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$HOME/fusion_ws/devel/setup.bash" ]] || \
   [[ ! -x "$HOME/venvs/fusion_cpu/bin/python" ]]; then
    echo '请先执行部署包中的 deploy/install_cpu.sh。'
    exit 1
fi
source /opt/ros/noetic/setup.bash
source "$HOME/fusion_ws/devel/setup.bash"
export FUSION_PYTHON="$HOME/venvs/fusion_cpu/bin/python"
export ROS_MASTER_URI="${FUSION_ROS_MASTER_URI:-http://127.0.0.1:11311}"
if ! rosparam list >/dev/null 2>&1; then
    echo "无法连接 $ROS_MASTER_URI，请先用原来的方式启动相机和雷达驱动。"
    exit 1
fi
rosparam set /use_sim_time false
echo "启动实时检测和 RViz，ROS master: $ROS_MASTER_URI"
if [[ -z "${DISPLAY:-}" ]]; then
    echo '当前终端没有图形显示环境；RViz 请在小车桌面终端运行此脚本。'
    echo '如只需在 SSH 中启动检测：在命令末尾加 rviz:=false。'
fi
for fusion_arg in "$@"; do
    if [[ "$fusion_arg" == model:=* ]]; then
        exec roslaunch image_pointcloud_fusion c32_live.launch "$@"
    fi
done
if [[ ! -f "$fusion_bundle/best.pt" ]]; then
    echo "缺少门框模型：$fusion_bundle/best.pt"
    exit 1
fi
exec roslaunch image_pointcloud_fusion c32_live.launch \
    "model:=$fusion_bundle/best.pt" "$@"
