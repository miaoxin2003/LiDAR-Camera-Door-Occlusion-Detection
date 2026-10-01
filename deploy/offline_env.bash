# Source this file in EACH terminal used for bag replay in the ROS container.
# Keep simulated time away from the live robot's usual ROS master.
source /opt/ros/noetic/setup.bash
fusion_persistent_root="$HOME/catkin_ws"
if [ -d "$fusion_persistent_root/fusion_inbox/vendor/ros_python" ]; then
    export PYTHONPATH="$fusion_persistent_root/fusion_inbox/vendor/ros_python:${PYTHONPATH:-}"
fi
if [ -f "$fusion_persistent_root/fusion_ws/devel/setup.bash" ]; then
    source "$fusion_persistent_root/fusion_ws/devel/setup.bash"
elif [ -f "$HOME/fusion_ws/devel/setup.bash" ]; then
    source "$HOME/fusion_ws/devel/setup.bash"
elif [ -d "$fusion_persistent_root/fusion_inbox/image_pointcloud_fusion" ]; then
    # Docker may be recreated after a reboot. The bind-mounted source still
    # exists even when the container's old ~/fusion_ws has disappeared.
    export ROS_PACKAGE_PATH="$fusion_persistent_root/fusion_inbox:${ROS_PACKAGE_PATH:-}"
fi
if [ -x "$fusion_persistent_root/venvs/fusion_cpu/bin/python" ]; then
    export FUSION_PYTHON="$fusion_persistent_root/venvs/fusion_cpu/bin/python"
elif [ -x "$HOME/venvs/fusion_cpu/bin/python" ]; then
    export FUSION_PYTHON="$HOME/venvs/fusion_cpu/bin/python"
fi
export ROS_MASTER_URI=http://127.0.0.1:11321
export ROS_IP=127.0.0.1
unset ROS_HOSTNAME
