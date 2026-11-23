# CMD 赛道二运行说明

这套基线在官方 CenterPoint 上增加两项跨体制泛化措施：

1. 同时使用 Ouster OS128 和 Hesai XT32 两个有标签机械式激光雷达源域训练。
2. 训练时随机删除俯仰扫描线，并随距离增加点丢弃比例，模拟固态雷达非均匀、低密度的采样模式。

## 1. 检查下载完整性

在仓库根目录执行：

```bash
python tools/cmd_competition.py check-data --data-root . --strict
```

下载尚未完成时，该命令会列出缺失的序列、传感器目录或帧。不要在检查通过前生成 info 文件。

如果 `hf download` 在大型仓库树分页时反复出现 HTTP 429，可绕过仓库树接口，按比赛所需文件直接续传：

```bash
python tools/download_cmd_required.py --data-root .
```

脚本只下载 Ouster/Hesai 的 train、val，Robosense 的 test，以及对应标签和标定文件；已有非空文件会跳过，未完成文件使用 `.part` 断点续传。

## 2. 生成数据索引

```bash
python tools/cmd_competition.py make-infos --data-root . --workers 8
```

会生成训练实际需要的五个索引：Ouster/Hesai 的 train、val，以及 Robosense 的 test。

## 3. 训练

8 卡训练：

```bash
bash tools/scripts/train_cmd_da.sh 8 "$(pwd)" source_mix_sparse
```

显存不足时，修改 `tools/cfgs/xmu_da_models/centerpoint_source_mix.yaml` 中的 `BATCH_SIZE_PER_GPU`，或通过命令行传入较小 batch size。

## 4. 隐藏目标域推理与打包

```bash
bash tools/scripts/infer_cmd_test.sh 4 /absolute/path/to/checkpoint_epoch_50.pth "$(pwd)"
```

脚本将：

- 使用无标签 Robosense 测试配置推理；
- 生成 `result.pkl`；
- 检查帧数、重复帧、字段、类别、box 维度和 NaN/Inf；
- 在仓库根目录生成仅包含 `prediction.pkl` 的 `submission.zip`。

也可以单独打包：

```bash
python tools/cmd_competition.py package \
  --result /path/to/result.pkl \
  --output submission.zip \
  --data-root .
```

## 5. 进一步提分建议

- 分别训练 Ouster、Hesai、双源三个模型，在目标域做 WBF/旋转 NMS 融合。
- 根据 CodaBench 分距离结果调低远距离小目标阈值，类别分别设置 score threshold。
- 使用源域验证集扫描稀疏化强度；不要使用隐藏测试标签。
- 在显存允许时尝试仓库自带 VoxelNeXt，并保留相同双源和稀疏化策略。
