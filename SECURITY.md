# Security Policy

## 运行与隐私边界

- 本项目为纯本地运行工具：本地 Web 服务仅监听 `127.0.0.1`，图片不会上传至任何远程服务器。
- 模型权重在首次使用时从公开上游地址（或哈希一致的镜像）下载，并执行 SHA-256 完整性校验。
- 官方 `Denoise.exe` **不存放在源码树中**，仅通过本仓库 GitHub Actions 从公开源码 `launcher/Denoise.cs` 构建并发布至 [GitHub Releases](https://github.com/juanhanyu/keepcolor-denoise/releases)，附带 `SHA256SUMS.txt` 供校验。

## 漏洞反馈 / Reporting a Vulnerability

如发现安全隐患（例如本地服务越权访问、命令注入、依赖供应链风险等），请勿直接公开在 Issue 中披露利用细节，请通过 GitHub Security Advisory（仓库 **Security → Advisories → Report a vulnerability**）私密上报，或联系仓库维护者处理。
