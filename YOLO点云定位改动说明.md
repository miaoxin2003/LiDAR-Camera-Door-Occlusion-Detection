# YOLO 检测驱动的点云定位：改动说明

## 目标与结果

实时融合节点现在保留原有的“点云聚类框与 YOLO 框按 IoU 匹配”路径，同时增加一条独立路径：对**未匹配聚类**的 YOLO 检测，直接在同一时刻的有效雷达点中寻找投影落入检测框的点。满足点数和深度一致性条件时，在 RViz 点云场景中显示青色三维支持区域框。

这个框表示“该 YOLO 图像区域内有一组雷达测得的三维点”，并不证明这些点构成与类别同形状的实体。例如打印在纸上的猫可以定位到纸张表面，但不能据此说雷达识别出了实体猫。

## 修改的文件

| 文件 | 修改内容 |
|---|---|
| `image_pointcloud_fusion/scripts/c32_common.py` | 新增 `localize_detection_points()`：根据已投影点的像素位置选取 YOLO 框内点；在相机深度上寻找一致的点层；点数不足或存在两个相近支持量的不同深度层时返回无结果；用所选点的中位数和 5%～95% 分位数形成三维位置及范围。 |
| `image_pointcloud_fusion/scripts/c32_oak_fusion.py` | 对聚类前的有效点云做一次投影并保留点、像素、深度的对应关系；继承的聚类关联完成后，只对未匹配的检测调用新定位方法；聚类抛出异常时记录错误并继续尝试新路径；发布青色 Marker 和数值三维盒，并记录 `ROI3D` 数量。 |
| `image_pointcloud_fusion/scripts/imgpc_fusion_detection.py` | 原聚类融合方法在保持原有发布行为的同时返回已匹配 YOLO 检测的身份，用于避免新路径重复标记。 |
| `image_pointcloud_fusion/config/c32_oak.yaml` | 增加新路径的开关和三个阈值。 |
| `image_pointcloud_fusion/rviz/c32_live.rviz` | 默认启用新的青色 YOLO 点云支持区域显示。 |
| `tools/test_roi_localization.py` | 增加不依赖 ROS 的定位测试，覆盖有单一表面、前后两层歧义和点数不足。 |
| `tools/build_robot_bundle.py` | 支持原始 bag 不在当前精简工作目录时，保留已记录的 bag 校验值并重建部署包；把本说明放入部署包。 |

部署包 `c32_oak_ros1_bundle.zip` 与 `bundle_contents.sha256` 也已随上述文件重建。`c32_oak_ros1_bundle（加速）.zip` 没有更新，请传输标准文件名的部署包。

## 新增话题与参数

| 项目 | 用途 |
|---|---|
| `/c32_fusion/yolo_point_markers` | RViz `MarkerArray`；青色三维支持区域框及类别、支持点数文字。每次处理会清除上一帧标记。 |
| `/c32_fusion/yolo_point_boxes3d` | `BoundingBoxArray`；未匹配检测对应的数值三维范围。 |
| `roi_localization_enabled: true` | 启用新路径；设为 `false` 可停用。 |
| `roi_min_points: 8` | 同一深度层至少需要的框内雷达点数。 |
| `roi_depth_band: 0.4` | 同一深度层允许的相机深度跨度，单位米。 |
| `roi_ambiguity_ratio: 0.7` | 另一独立深度层的点数达到主层的这一比例时，放弃定位。 |

新路径使用聚类之前的有效点云，所以不受 `z_axis_min/z_axis_max`、`voxel_size` 或 DBSCAN 是否形成独立簇的限制；仍受 `max_range`、相机视野、内外参和时间配对影响。旧的 `/c32_fusion/cluster_markers`、`/c32_fusion/bounding_boxes3d` 和红色聚类投影保持原有含义。

## 小车部署与观察

1. 将本目录更新后的 `c32_oak_ros1_bundle.zip` 传到小车，按《实时检测精简手册.md》解压并重新运行 `deploy/install_cpu.sh`，然后启动原有两路驱动与 `deploy/run_live.sh`。安装脚本会保留小车上已存在的 `c32_oak.yaml`；即使旧 YAML 尚无新参数，节点也使用上表默认值。
2. RViz 点云视图中，原聚类匹配框与新青色框分开展示。图像中的绿色框仍是 YOLO 检测，红框仍是聚类投影；没有红框但有足够可信的框内雷达点时，可以出现青色框。
3. 运行日志的 `ROI3D=N` 表示本帧由新路径生成的框数。若绿色框存在但没有青色框，可能是框内点不足、前后深度层有歧义、距离超出 `max_range` 或投影标定不准。不要通过无限降低点数阈值来强制出框。

## 当前边界

- 青色框是雷达支持点的轴对齐范围，不是物体真实轮廓；打印图案、邻近墙面和目标表面可能处于同一深度层。
- 当前没有实现门框上沿端点、三维直线、门洞面或障碍物遮挡率。门框细窄且框内没有足够雷达点时，新路径会明确不给三维框；下一步应使用门柱或墙面支撑点做受约束的几何推算。
- 如果障碍物完全占据 YOLO 框且只有障碍物有雷达回波，单帧框内点无法证明这些点属于门框；青色框不能作为门位置的最终测量依据。
- 外参仍需在实际小车上核对。离线单元测试验证了点层选择逻辑，不等于验证现场定位精度或 ROS 实时性能。

## 验证

- `py tools/test_roi_localization.py`：3 个测试通过。
- 修改过的 Python 文件通过 AST 语法解析。
- 部署包重建后通过 ZIP CRC 检查。
