# KeepColor Mobile

Android 与 iPhone 离线开发版。使用 Capacitor 和 ONNX Runtime，在设备上执行 DRUNet，无远程处理接口，不含 APISR。模型在构建阶段下载并打入应用资源。

## 构建

在仓库根目录创建并激活 Python 虚拟环境，执行：

```sh
python -m pip install -r skills/keepcolor-denoise/scripts/requirements.txt
python -m pip install -r mobile/tools/requirements.txt
python mobile/tools/export_model.py
cd mobile
npm ci
npm test
npm run build
npx cap sync
```

`export_model.py` 校验下载权重，导出 ONNX 并比较 PyTorch / ONNX 数值结果。模型约 125 MiB，不提交到 Git；缺少模型时正式构建会失败，避免生成不能去噪的空壳。

- Android：用 Android Studio 打开 `android/`。需要 Android SDK 36 与 Capacitor 8 对应的 JDK；发行时使用作者自己的签名。
- iPhone：在 macOS / Xcode 打开 `ios/App/App.xcodeproj`，配置开发团队和签名后构建。

## 功能与边界

支持选择图片、DRUNet 强度调整、按住查看原图、PNG 导出 / 系统分享。图片只在设备中处理；导出会产生临时缓存文件。使用 128 像素重叠分块与 2×2 像素重排，输入限制 400 万像素，不自动缩小图片。

暂未移植电脑版 L0、保护笔刷、批处理。Canvas 解码不保留原始 ICC 配置；透明通道虽被保留，但颜色转换和分块大小可能导致与电脑版结果不同。尚未完成原生编译、真机性能、视觉质量和分享功能测试，不是正式发行版。

已通过技能格式校验、实际 Python DRUNet 图片处理、透明度 / 尺寸验证、四组像素重排往返测试、ONNX 数值对照、前端生产构建以及原生工程资源同步。

DRUNet 许可见 `DRUNET-LICENSE.txt`；项目代码使用根目录 MIT 许可。
