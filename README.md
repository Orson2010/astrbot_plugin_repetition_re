# AstrBot 增强版复读机插件

一个功能强大的增强版复读插件，支持倒读、乱序、图片处理（翻转、缩放、GIF支持、反色、GIF速度调整）和打断施法功能。

## 功能特点

### 核心复读功能
- **智能复读检测**: 检测连续相同消息并自动复读（可配置触发阈值）
- **打断施法**: 有概率在复读前发送打断施法消息
- **多会话支持**: 群聊/私聊独立计数，互不干扰
- **命令过滤**: 自动跳过以 "/" 开头的命令消息

### 文本处理功能
- **倒读复读**: 将消息内容反转后复读（如"你好"→"好你"）
- **乱序复读**: 随机重新排序文本字符后复读
- **智能分块**: 可按配置的分块大小进行局部乱序

### 图片处理功能（支持静态图片和GIF）
- **水平翻转**: 随机水平翻转图片
- **垂直翻转**: 随机垂直翻转图片
- **缩放处理**: 随机缩放图片（等比或独立缩放）
- **反色处理**: 将图片颜色反转（黑色变白色等）
- **GIF优化**: 正确处理GIF动画，保持动画效果
- **GIF速度调整**: 智能调整GIF播放速度（抽帧加速/重复帧减速）

### 高级特性
- **概率控制**: 所有处理功能均有独立可配置的概率
- **透明度过滤**: 可配置丢弃图片透明度（简化处理）
- **最小变化率**: 确保缩放操作有实质性的变化
- **调试模式**: 详细的调试日志便于问题排查

## 指令说明

- `/repetition on`: 在当前群开启复读功能
- `/repetition off`: 在当前群关闭复读功能
- `/repetition`: 查看帮助

## 配置项

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `break_spell_probability` | float | 0.3 | 打断施法的触发概率（0-1） |
| `break_spell_text` | string | "打断施法！" | 打断施法时发送的文本内容 |
| `repeat_threshold` | int | 2 | 触发复读所需的最小连续相同消息数 |
| `reverse_probability` | float | 0.1 | 触发倒读复读的概率（0-1） |
| `reorder_probability` | float | 0.1 | 触发随机重新排序复读的概率（0-1） |
| `min_chunk_size` | int | 2 | 文本分块的最小字数 |
| `max_chunk_size` | int | 5 | 文本分块的最大字数 |
| `reorder_strength` | float | 0.5 | 随机重新排序的强度（0-1，0不排序，1完全随机） |
| `reorder_per_chunk_times` | int | 1 | 每个文本分块内的重新排序次数 |
| `image_process_probability` | float | 0.3 | 处理图片的概率（0-1） |
| `flip_horizontal_probability` | float | 0.1 | 水平翻转图片的概率（0-1） |
| `flip_vertical_probability` | float | 0.1 | 垂直翻转图片的概率（0-1） |
| `scale_probability` | float | 0.1 | 缩放图片的概率（0-1） |
| `min_scale_percent` | float | 0.5 | 图片缩放的最小百分比（0.1-5.0，如0.5表示50%） |
| `max_scale_percent` | float | 1.5 | 图片缩放的最大百分比（0.1-5.0，如1.5表示150%） |
| `invert_color_probability` | float | 0.05 | 反色处理图片的概率（0-1） |
| `gif_frame_change_probability` | float | 0.1 | GIF播放速度改变的概率（0-1） |
| `gif_frame_change_min_rate` | float | 0.5 | GIF速度改变的最小倍率（如0.5表示加速到2倍速） |
| `gif_frame_change_max_rate` | float | 2.0 | GIF速度改变的最大倍率（如2.0表示减速到0.5倍速） |
| `min_scale_threshold` | float | 0.1 | 缩放的最小变化率（相对于原图的变化百分比，如0.1表示至少变化10%） |
| `preserve_aspect_ratio` | bool | true | 缩放时是否保持宽高比 |

## 使用示例

### 基本复读
```
用户A: 早上好！
用户B: 早上好！
机器人: 早上好！  （触发复读）
```

### 倒读复读
```
用户A: 你好世界
用户B: 你好世界
机器人: 界世好你  （触发倒读复读）
```

### 图片处理
- 发送相同图片时，可能被水平/垂直翻转
- 可能被随机缩放（放大或缩小）
- 可能被反色处理（颜色反转）
- GIF动图可能被加速或减速播放

### 打断施法
```
用户A: 测试消息
用户B: 测试消息
机器人: 打断施法！  （触发打断施法）
```

## 安装方法

1. 将本插件文件夹 `astrbot_plugin_repetition_re` 放入 AstrBot 的插件目录（`data/plugins/`）
2. 重启 AstrBot 或重新加载插件
3. 在配置界面调整各项参数（可选）

## 依赖要求

- Python 3.7+
- AstrBot 框架
- 可选但推荐：PIL/Pillow（用于图片处理）
- 可选但推荐：aiohttp（用于网络图片下载）

如果缺少可选依赖，图片处理功能将自动跳过。

## 注意事项

1. **性能考虑**: GIF处理可能消耗较多资源，建议在配置中合理设置概率
2. **透明度处理**: 默认会丢弃图片透明度以简化处理逻辑
3. **缩放保护**: 通过`min_scale_threshold`确保缩放有实质性变化，避免微调
4. **GIF速度调整**: 采用智能算法，加速时抽帧，减速时重复帧
5. **错误处理**: 图片处理失败时会自动回退到原始图片
6. **调试模式**: 设置`DEBUG = True`可在控制台查看详细处理日志

## 高级配置建议

### 轻度娱乐模式
```json
{
  "image_process_probability": 0.2,
  "flip_horizontal_probability": 0.05,
  "flip_vertical_probability": 0.05,
  "scale_probability": 0.05,
  "invert_color_probability": 0.02,
  "gif_frame_change_probability": 0.05
}
```

### 高度随机模式
```json
{
  "image_process_probability": 0.5,
  "flip_horizontal_probability": 0.15,
  "flip_vertical_probability": 0.15,
  "scale_probability": 0.15,
  "invert_color_probability": 0.1,
  "gif_frame_change_probability": 0.2,
  "reverse_probability": 0.2,
  "reorder_probability": 0.2
}
```

### 保守模式
```json
{
  "image_process_probability": 0.1,
  "gif_frame_change_probability": 0,
  "break_spell_probability": 0.1
}
```

## 技术实现

### 图片处理流程
1. 检测图片格式（静态图片或GIF）
2. 下载图片数据（支持URL和base64）
3. 随机选择处理方式（翻转、缩放、反色）
4. 处理图片并转换为RGB模式（丢弃透明度）
5. 保存为base64数据URI返回

### GIF速度调整算法
- **加速（<0.8x）**: 智能抽帧，保留首尾帧，等间隔选择中间帧
- **减速（>1.2x）**: 重复帧插值，保持动画流畅性
- **轻微调整（0.8-1.2x）**: 调整每帧的`duration`时间

### 文本处理算法
- **倒读**: 简单字符串反转
- **乱序**: 分块随机交换字符，可控制强度

## 故障排除

### 常见问题
1. **图片不处理**: 检查PIL/Pillow和aiohttp是否安装
2. **GIF不动**: 确保PIL版本支持GIF动画处理
3. **处理速度慢**: 降低`image_process_probability`或关闭GIF处理
4. **内存占用高**: 限制最大GIF帧数或图片尺寸

### 调试方法
在`main.py`中设置`DEBUG = True`，查看控制台日志了解处理过程。

## 更新日志

### v1.1.0 (最新)
- 新增反色处理功能
- 新增GIF速度调整功能（智能抽帧/重复帧）
- 新增最小缩放变化率配置
- 优化GIF处理逻辑，修复动画显示问题
- 改进缩放算法，确保实质性变化
- 更新配置文件说明

### v1.0.x
- 基础复读功能
- 倒读和乱序复读
- 图片翻转和缩放
- 打断施法功能

## 关于

- **插件名称**: astrbot_plugin_repetition_re
- **版本**: v1.1.0
- **作者**: Aug (基于FengYing1314的原始版本)
- **项目地址**: https://github.com/FengYing1314/astrbot_plugin_repetition
- **许可证**: 根据原始项目许可证

## 支持与反馈

如果遇到问题或有改进建议，欢迎：
1. 在GitHub仓库提交Issue
2. 查看调试日志定位问题
3. 调整配置参数适应需求

---

**提示**: 本插件旨在增加聊天趣味性，请合理使用，避免滥用影响正常聊天体验。# astrbot_plugin_repetition_re
