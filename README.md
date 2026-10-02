Camera LIDAR pre-fusion, post-fusion  
method：pointcloud cluster+yolo11  
video：https://www.bilibili.com/video/BV1fwMwzMEFW/?spm_id_from=333.1387.upload.video_card.click

## C32 + OAK / ROS1 CPU 复现资料

- **当前门框模型的真机部署：[真机实时部署教程](真机实时部署教程.md)**
- 当前部署包：`c32_oak_ros1_bundle.zip`，含 `best.pt` 和最新遮挡率代码，不含 bag。
- Docker 关机重开后按原命令回放：[重启后回放](deploy/重启后回放.md)
- 早期通用 YOLO11s 资料：[实时检测精简手册](实时检测精简手册.md)
- 自动安装：`bash ~/fusion_inbox/deploy/install_cpu.sh`
- 驱动启动后，检测与 RViz 一起运行：`bash ~/fusion_inbox/deploy/run_live.sh`
- [完整操作手册](小车复现操作手册.md)
- [下午执行速查](下午执行速查.md)
- 实测参数与投影对比：`inspection/`
- 配置：`image_pointcloud_fusion/config/c32_oak.yaml`
- 启动入口：`c32_projection.launch`、`c32_fusion.launch`
- 可传输代码包：`c32_oak_ros1_bundle.zip`（bag 单独传输）

标定 TXT 的矩阵方向与当前 bag 的直接投影存在矛盾；默认求逆只是候选配置，先按手册验收静态投影。

## 门框上沿模型回放

`my_30s_all.bag` 回放默认使用当前目录的 `best.pt`，模型训练尺寸为 640。
停止原有检测节点和 RViz，保持回放 master，重新启动：

```bash
source ~/catkin_ws/fusion_inbox/deploy/offline_env.bash
rosparam set /use_sim_time true
roslaunch image_pointcloud_fusion c32_live.launch max_hz:=1.0
```

以后换权重，可以把新的 `.pt` 文件放到容器能访问的路径，并在启动时传入
`model:=/容器内绝对路径/新权重.pt`；也可以用新文件替换当前目录的 `best.pt`，
保留以上默认命令。两种方式都需要重启检测节点才能加载新权重。若新模型的
训练尺寸不同，可同时传入 `imgsz:=尺寸`。

待 `CPU fusion ready` 后，再播放 bag。RViz 的 `Live detection` 显示模型直接
检测到的二维框；`YOLO point support` 的青色三维框只在检测框内部有足够且
深度一致的实际雷达点时显示。若某个检测框内没有足够雷达点，只有二维框、
没有青色三维框是正常结果。

## 门框上沿线与门位置面

检测框内的实测雷达点经过三维直线拟合后，距离直线不超过 `0.01 m`
（1 cm）的点用于确定黄色门框上沿线。按图像横向顺序，相邻拟合点的
相机深度跳变超过 `0.12 m` 时会分段，只保留有足够跨度的一段。
相邻帧的线段端点按 `0.35` 的新帧权重平滑；检测中断、时间倒退或
门位置明显变化时重置。线段只覆盖实测内点，不延长到图像框两端。
这些参数位于 `image_pointcloud_fusion/config/c32_oak.yaml`。
RViz 中启用 `Measured door top line` 查看，话题为
`/c32_fusion/door_top_line_markers`。

浅蓝色的 `Door position plane` 从黄色线的两个端点沿雷达坐标系 `-Z`
方向延伸到 `door_ground_z`，当前 bag 使用 `-1.17 m`；它是构造的门位置面，
不依赖面前是否有点云遮挡物，话题为 `/c32_fusion/door_plane_markers`。
雷达安装高度或地面高度变化时，应重新测定并修改
`image_pointcloud_fusion/config/c32_oak.yaml` 中的 `door_ground_z`。
原 `menkuang ROI (...pts)` 文字已移除，青色实测点框仍保留。

## 门面遮挡率

检测到门框并生成门面后，程序用**原始雷达点云**从雷达原点向各回波发射射线，
计算射线与虚拟门面的交点。交点落在门内、且回波比门面近 `0.15 m` 以上时，
该网格视为遮挡；回波到达门面或在门面后方时视为通畅。没有回波的网格保持
“未观测”，不会被当作通畅。门框和地面附近各 `0.10 m` 不参加统计。
网格大小为 `0.15 m`，可在 `config/c32_oak.yaml` 修改。

`/c32_fusion/door_occlusion_ratio` 为遮挡网格数除以**已观测**网格数，
`/c32_fusion/door_occlusion_coverage` 为已观测网格数除以参与统计的全部
网格数；二者均为 `0–1`。覆盖率低于 `0.05` 或未检测到门时，遮挡率发布
`NaN`，表示证据不足。RViz 的 `Door occlusion` 显示红色遮挡网格和
“Occlusion / observed”文字。这个比例描述**本帧雷达可观测射线**，不能
把未观测区域推断为真实无遮挡面积；地面高度、门面位置和点云密度会影响结果。

111

222

333

444
