# 实验进度记录

## 2026-08-27

### ✅ Step 1: RVQ Codebook训练成功

**配置**:
- 8层residual quantization
- 每层256个codes
- Hidden dim: 2048 (Qwen2.5-3B)
- 训练样本: 2000个合成KV对

**关键发现** 🎯:

| 保留层数 | Keys RMSE | Values RMSE | 误差增加 |
|---------|-----------|-------------|----------|
| 8层(完整) | 0.0081 | 0.0082 | baseline |
| 6层 | 0.0113 | 0.0113 | +38% |
| 4层 | 0.0144 | 0.0144 | +77% |
| 2层 | 0.0173 | 0.0173 | +112% |

**核心洞察**:
1. ✅ **渐进式精度降级work了！** 从8层→4层，误差只增加77%，不是灾难性的
2. ✅ **位置编码问题完全避免** - token不删除，只是精度降低
3. ✅ **可以trade off显存和质量** - 4层用一半显存，质量保持合理

**显存计算**:
- 完整KV (FP16): 2048 dim × 2 bytes = 4KB/token
- RVQ 8层: 8 × log2(256) = 64 bits = 8 bytes/token → **压缩512倍**
- RVQ 4层: 4 bytes/token → **压缩1024倍**

**理论vs实际**:
- 理论上4层应该保留50%信息
- 实际误差增加77%，略高于理论值
- 但仍然在可接受范围内

### 下一步: Step 2 - 实际推理测试

现在需要：
1. 在真实生成中使用这些codebook
2. 对比3个方法的生成质量：
   - Baseline (无驱逐)
   - Token驱逐 (H2O-style, 删除50% token)
   - RVQ驱逐 (保留所有token，降到4层)

3. 评估指标：
   - 困惑度 (Perplexity)
   - 生成质量 (ROUGE)
   - 显存占用
   - 推理速度

**预期结果**:
如果RVQ方法在相同显存下能保持更低的困惑度 → 论文核心贡献确立

**挑战**:
- 需要hook到模型的attention计算
- 或者实现一个简化的"伪驱逐"测试
  - 先正常生成
  - 再用RVQ压缩KV cache
  - 重新计算困惑度

---

## 文件清单

### 已完成
- [x] `src/rvq_kv_cache.py` - RVQ KV cache实现框架
- [x] `src/baseline_cache.py` - Baseline实现
- [x] `experiments/quick_test.py` - 基础测试
- [x] `experiments/step1_train_codebooks_simple.py` - RVQ训练
- [x] `checkpoints/rvq_codebooks_L8_C256.pt` - 训练好的codebook ✓

### 待完成
- [ ] `experiments/step2_comparison.py` - 完整对比实验
- [ ] `src/model_wrapper.py` - 模型Hook (可选，如果直接集成)
- [ ] `experiments/analyze_results.py` - 结果分析和可视化

---

## 论文相关

### 当前创新点
1. **渐进式精度降级** - 不是二元的删除/保留，而是连续的精度空间
2. **完全绕过位置编码问题** - token位置索引保持连续
3. **与现有eviction策略正交** - 可以叠加到任何方法上

### 实验证据 (目前)
- ✅ RVQ重建误差测试完成
- ⏳ 生成质量对比 (待做)
- ⏳ 显存-质量trade-off曲线 (待做)
- ⏳ 多模型验证 (待做)

### 发表可行性评估
- CCF B: 需要完成Step 2-3的对比实验
- SCI 3-4区: 目前进度已经可以开始写作框架

---

最后更新: 2026-08-27 17:30
