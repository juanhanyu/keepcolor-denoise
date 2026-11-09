# 技能与插件

将 `skills/keepcolor-denoise` 复制到支持本地技能的宿主的技能目录，例如项目 `.agents/skills/`。选择该技能并提供图片路径。需要 Python 环境，首次运行需要下载 DRUNet 模型，图片不上传。

建议使用虚拟环境：

```sh
python -m venv .venv
# 激活 .venv 后执行
python -m pip install -r skills/keepcolor-denoise/scripts/requirements.txt
python skills/keepcolor-denoise/scripts/denoise.py input.png --out results
```

默认保留原尺寸，输出 PNG。不同目录中的同名输入应使用不同输出目录。支持可选 `--l0`，不含 APISR。

插件 ZIP 的根目录放 `plugin.json` 和 `skills/`。不要打包模型缓存、输出图片、虚拟环境或凭证。该配置为开发包；公开投稿前还需完善图标、目录信息和发布者资料，并通过平台检查。
