# CMD competition training artifacts

该目录保存 2026-11-15 在 `lab` 服务器完成的 CMD 跨机制域适应训练产物，不包含 CMD 数据集。

## 模型权重

- `checkpoint_epoch_46.pth`：用于当前隐藏测试集提交的第 46 轮权重。
- `checkpoint_epoch_50.pth`：训练结束时的第 50 轮权重。
- `latest_model.pth`：训练程序保存的最新权重。

权重路径：

`training_output/xmu_da_models/centerpoint_source_mix/source_mix_fp32_1gpu/ckpt/`

## 日志和结果

- 正式训练及前期测试的 `train_*.log`。
- TensorBoard `events.out.tfevents.*`。
- 训练时使用的 `centerpoint_source_mix.yaml`。
- 第 40 至 50 轮验证集 `result.pkl`。
- Robosense 隐藏测试集推理结果与 `log_eval_*.txt`。
- `submissions/submission_epoch46.zip`：可上传 Codabench 的提交压缩包。

这些产物从 `/home/hyy/src/CMD/output` 和 `/home/hyy/src/CMD/submissions` 同步得到。
