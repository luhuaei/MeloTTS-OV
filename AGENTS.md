# AGENTS.md

本文件定义本仓库自动化构建与 Jetson 联调测试的硬性约束。后续任何代理或脚本执行都必须遵守。

## 1. Jetson 自动化测试流程（必须按顺序）

1. 代码同步到 Jetson（默认机器）  
   - 目标：`nvidia@192.168.1.230`
   - 同步命令基线：
     - `rsync -rP --exclude .venv . nvidia@192.168.1.230:~/MeloTTS-OV/`
   - 建议额外排除：`.git`、`logs`、`__pycache__`

2. 在 Jetson 上构建镜像  
   - 必须通过 SSH 远程执行：
     - `make build_jetson`

3. 在 Jetson 上运行 Docker 并启动服务  
   - 默认命令：
     - `docker run --rm -ti -p 8888:8000 --runtime nvidia registry.lazycat.cloud/x/lzc-aipod-melotts-ov:0.1.2`
   - 自动化脚本中允许使用 `--network host` 或其他端口，但必须可配置。

4. 通过 HTTP 请求验证 TTS 可用  
   - 必须请求：`/v1/audio/speech`
   - 示例：
     - `curl -X POST http://<jetson-ip>:<port>/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"tts-1","input":"长时间。肥差","voice":"alloy","response_format":"mp3","speed":1.0}' --output speech.mp3`
   - 判定条件：HTTP 200 且输出音频文件非空。

5. 自动收集日志与产物  
   - 必须收集：构建日志、容器日志、健康检查日志、HTTP 请求与响应头、生成音频文件、总结文件。
   - 必须给出最终 PASS/FAIL 和日志目录。

## 2. ASR 回归校验（必须支持）

1. 自动化测试需要支持把生成音频提交到 ASR 服务做转写验证：  
   - `https://asr-ai.13gxg.heiyu.space/v1/audio/transcriptions`
2. 至少支持参数：
   - `file`（音频文件）
   - `model`（如 `whisper-1` / `paraformer-large`）
   - `response_format`（默认 `json`）
3. 必须比较“期望文本 vs ASR 文本”，输出相似度分数和阈值判断（可配置）。
4. ASR 严格模式下，不达阈值必须返回 FAIL。

## 3. 离线化约束（强制）

1. 默认使用离线模式（`OFFLINE_MODE=1`）。  
2. 运行阶段禁止联网下载模型/词典/分词器。  
3. 切换语言时禁止重新下载资源。  
4. 必须在镜像构建阶段预拉取并缓存多语言所需资源（含语音/文本处理相关依赖）。  
5. 需要设置离线环境变量并确保代码使用本地缓存（例如 HF/NLTK 离线模式）。

## 4. Dockerfile.jetson 构建顺序约束（强制）

必须遵循以下层级顺序以兼顾离线化和构建速度：

1. 先放模型与下载相关内容（例如 `COPY onnx_models ...`、预下载脚本）。  
2. 再安装 `torch`/`torchvision`/`triton` 等重依赖。  
3. 再安装 `requirements` 等 Python 依赖。  
4. 最后再 `COPY . /app`（业务代码放最后，最大化缓存命中）。  

补充：允许使用 BuildKit 缓存优化，例如：

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install triton torchvision -i http://wa.lan:10608/simple --trusted-host wa.lan
```

额外优化约束：

1. `onnx_models` 中重复的 `bert_multilingual.onnx` 和 tokenizer 文件必须在镜像内单副本复用（例如 `shared_bert` + 软链接）。  
2. 镜像构建必须支持按语言裁剪（`MELO_LANGS`），避免始终打包全部语言模型。  
3. 容器默认启动路径应避免重复模块导入和不必要初始化，优先提升启动与首包响应速度。

## 5. 自动化脚本约束

1. 默认入口脚本：`scripts/jetson_e2e_test.sh`。  
2. 脚本必须支持：
   - `SKIP_SYNC=1`（跳过同步）
   - `SKIP_BUILD=1`（跳过构建）
   - `OFFLINE_MODE=1` 默认启用（可显式设为 `0` 关闭）
   - 自定义 `DOCKER_RUN_ARGS`、`SERVICE_PORT`
   - 可选 `ASR_VERIFY_URL` 与阈值配置
   - 默认执行一次 `voice=ZH_MIX_EN` 的混合语言请求用于离线回归
   - 默认执行 voice 矩阵测试（OpenAI 别名 + `ZH_MIX_EN,ZH,EN,JP,KR,ES,FR`）
   - voice 矩阵必须输出汇总文件 `voice_matrix/summary.txt`
   - 支持 `VOICE_MATRIX_STRICT`，默认严格模式（任一 voice 失败则 FAIL）
   - 支持扫描容器日志中的离线下载错误并在发现时 FAIL
3. 每次执行必须生成独立日志目录，且包含 `summary.txt`。  
4. `summary.txt` 至少包含：
   - `PASS/FAIL`
   - 远端机器
   - 镜像名
   - 容器名
   - 日志目录

## 6. 变更验证标准（提交前）

1. 至少完成一次全流程（sync + build + run + HTTP）PASS。  
2. 至少完成一次 ASR 验证 PASS（可用跳过构建快速回归）。  
3. 如出现端口冲突，必须在日志中明确记录并给出替代端口执行结果。  
4. 任何新增功能都不能破坏离线运行约束。
5. 发布前必须验证 `ZH_MIX_EN` 在离线模式下可用（不能触发 HuggingFace 在线拉取）。
6. 发布前必须验证 voice 矩阵内所有 voice 均可生成非空音频文件。
