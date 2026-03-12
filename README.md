# ASRTTS

本项目提供一个本地化的 ASR + TTS 服务：

- ASR 支持 4 种模式
  - `whisper`：`faster-whisper-large-v3-turbo`
  - `whisper_small`：`faster-whisper-small`
  - `paraformer`：`speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
  - `funasr_nano`：`Fun-ASR-Nano-2512`
- TTS 预留 `MeloTTS` 本地目录接入
- ASR 后处理支持监所业务词表、短语纠错、混淆词纠错，以及可开关的大模型纠错

注意：

- 仓库默认**不包含** 4 种 ASR 模式对应的模型和附属资源
- 用哪个 ASR 模式，就执行对应的下载脚本
- 下载后的模型都会放在项目目录内，方便后续整体迁移
- Hugging Face 下载脚本默认走 `https://hf-mirror.com`
- 各下载脚本会在运行时打印镜像页面地址，失败后可按提示手工下载
- 所有下载脚本统一按“文件存在且大小大于 0 才跳过”的规则执行；若文件大小为 `0`，会视为损坏并重新覆盖下载

## 目录说明

- `app/`：Web 服务、ASR/TTS 运行代码
- `scripts/`：环境和模型下载脚本
- `domain_profiles.yaml`：基础监所词表、短语规则、混淆规则
- `security_terms.yaml`：常用安防词汇
- `config.json`：运行配置
- `res/uploads/`：上传音频目录
- `res/tts_output/`：TTS 输出目录

## 创建项目内虚拟环境

```bat
cd /d 你的项目目录
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 四种 ASR 模式的下载脚本

### 1. whisper

下载 `faster-whisper-large-v3-turbo`：

```bat
.venv\Scripts\python.exe scripts\download_whisper_large_v3_turbo.py
```

下载后模型目录：

```text
faster-whisper-large-v3-turbo
```

### 2. whisper_small

下载 `faster-whisper-small`：

```bat
.venv\Scripts\python.exe scripts\download_whisper_small.py
```

下载后模型目录：

```text
faster-whisper-small
```

### 3. paraformer

下载 `speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`：

```bat
.venv\Scripts\python.exe scripts\download_paraformer.py
```

下载后模型目录：

```text
models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

### 4. funasr_nano

下载 `Fun-ASR-Nano-2512` 主模型、runtime 代码，以及 `VAD` 模型：

```bat
.venv\Scripts\python.exe scripts\download_funasr_nano.py
```

如果只缺 `VAD` 模型，也可以单独执行：

```bat
.venv\Scripts\python.exe scripts\download_funasr_vad.py
```

下载后模型目录：

```text
models/Fun-ASR-Nano-2512
models/speech_fsmn_vad_zh-cn-16k-common-pytorch
```

## 切换 ASR 后端

可通过页面切换，也可以直接改 `config.json`。

### whisper

```json
{
  "asr": {
    "backend": "whisper",
    "model_path": "faster-whisper-large-v3-turbo"
  }
}
```

### whisper_small

```json
{
  "asr": {
    "backend": "whisper",
    "model_path": "faster-whisper-small"
  }
}
```

### funasr_nano

```json
{
  "asr": {
    "backend": "funasr_nano",
    "funasr_nano": {
      "model_path": "models/Fun-ASR-Nano-2512",
      "vad_model_path": "models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
      "release_after_inference": false,
      "use_prompt_terms_as_hotwords": true,
      "hotword_mode": "full",
      "max_hotwords": 200,
      "hotwords": []
    }
  }
}
```

其中：

- `release_after_inference = false` 表示 `funasr_nano` 默认常驻内存/显存，只在切换后端时释放
- 如果你更看重显存回收，也可以改成 `true`，让它每次识别完成后释放模型和显存

### paraformer

```json
{
  "asr": {
    "backend": "paraformer",
    "paraformer": {
      "model_path": "models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
      "use_prompt_terms_as_hotwords": true,
      "hotword_mode": "full",
      "max_hotwords": 200,
      "hotwords": []
    }
  }
}
```

## GPU 说明

- `whisper/faster-whisper` 的 GPU 依赖是 `ctranslate2 + CUDA/cuDNN`
- `funasr_nano` 的 GPU 依赖是 `PyTorch CUDA 版`
- 如果环境不满足，系统会自动回退到 CPU，或在日志里给出明确错误

## 启动服务

```bat
cd /d 你的项目目录
.venv\Scripts\python.exe -m app
```

页面地址：

```text
http://127.0.0.1:8000/
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

## 后处理链路

```text
音频 -> ASR -> 原始文本 -> 短语纠错 -> 混淆词纠错 -> 大模型纠错（可关） -> 最终文本
```

约束：

- 基础监所词表、短语规则、混淆规则维护在 `domain_profiles.yaml`
- 常用安防词汇维护在 `security_terms.yaml`
- 业务词表不硬编码在代码中
- 当前固定业务域为 `prison`

## 主要接口

### `GET /api/health`

返回服务状态、当前 ASR 后端、模型路径、TTS 状态。

### `GET /api/asr/backend-options`

返回可选 ASR 模式：

- `whisper`
- `whisper_small`
- `paraformer`
- `funasr_nano`

### `POST /api/asr/backend`

切换当前 ASR 模式。

请求体示例：

```json
{
  "backend": "funasr_nano"
}
```

### `POST /api/asr/transcribe`

普通识别接口。

表单参数：

- `file`
- `language`

### `POST /api/asr/transcribe-stream`

流式识别接口，按阶段返回：

- `raw_text`
- `phrase_text`
- `final_text`
- `complete`

### `POST /api/tts/synthesize`

TTS 合成接口。

### `GET /media/{file_name}`

播放或下载 TTS 输出音频。

## 打包和迁移

为了减小仓库体积，以下内容默认不入仓：

- `.venv`
- `faster-whisper-large-v3-turbo`
- `faster-whisper-small`
- `models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- `models/Fun-ASR-Nano-2512`
- `models/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- `res/uploads/*`
- `res/tts_output/*`

迁移到新机器时：

1. 拷贝项目目录
2. 在项目内创建 `.venv`
3. 安装 `requirements.txt`
4. 按需执行对应 ASR 下载脚本
5. 启动服务
