const $ = (id) => document.getElementById(id);

let mediaRecorder = null;
let mediaStream = null;
let recordChunks = [];
let latestAudioUrl = "";
const HEALTH_TIMEOUT_MS = 5000;
const ASR_TIMEOUT_MS = 180000;
const TTS_TIMEOUT_MS = 180000;
const RECORD_START_WARMUP_MS = 600;
const RECORD_STOP_GUARD_MS = 350;

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function playReadyTone() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) {
    return;
  }

  const ctx = new AudioCtx();
  try {
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.12, ctx.currentTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.16);
    oscillator.connect(gain);
    gain.connect(ctx.destination);
    oscillator.start();
    oscillator.stop(ctx.currentTime + 0.18);
    await sleep(220);
  } catch (_) {
    // 忽略提示音失败，不影响录音。
  } finally {
    if (typeof ctx.close === "function") {
      ctx.close().catch(() => {});
    }
  }
}

function getRequestedAudioConstraints() {
  return {
    channelCount: { ideal: 1 },
    sampleRate: { ideal: 48000 },
    sampleSize: { ideal: 16 },
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  };
}

function formatCaptureEnhancement(track) {
  if (!track || typeof track.getSettings !== "function") {
    return "";
  }

  const settings = track.getSettings();
  const parts = [
    `AEC:${settings.echoCancellation === true ? "开" : "关"}`,
    `NS:${settings.noiseSuppression === true ? "开" : "关"}`,
    `AGC:${settings.autoGainControl === true ? "开" : "关"}`,
  ];

  if (settings.sampleRate) {
    parts.push(`${settings.sampleRate}Hz`);
  }
  if (settings.channelCount) {
    parts.push(`${settings.channelCount}ch`);
  }
  return parts.join(" | ");
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(new Error(`请求超时: ${timeoutMs}ms`)), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("请求超时，请检查服务是否仍在运行。");
    }
    if (err instanceof TypeError) {
      throw new Error("无法连接到服务，请确认后端仍在运行。");
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

async function fetchHealth() {
  const res = await fetchWithTimeout("/api/health", {}, HEALTH_TIMEOUT_MS);
  const data = await res.json();
  const healthBox = $("health_box");
  if (!healthBox) {
    return;
  }
  healthBox.textContent =
    `ASR 后端: ${data.asr_backend}\n` +
    `ASR 模型: ${data.asr_model}\n` +
    `固定业务域: ${data.asr_default_domain}\n` +
    `ASR 大模型纠错: ${data.asr_correction_enabled ? "是" : "否"}\n` +
    `TTS 启用: ${data.tts_enabled ? "是" : "否"}\n` +
    `MeloTTS 目录: ${data.tts_vendor_dir}\n` +
    `TTS 缓存目录: ${data.tts_cache_dir}`;
}

async function fetchBackendOptions() {
  const res = await fetchWithTimeout("/api/asr/backend-options", {}, HEALTH_TIMEOUT_MS);
  const data = await res.json();
  const select = $("asr_backend");
  if (!select) {
    return;
  }
  if (Array.isArray(data.options) && data.options.length) {
    select.innerHTML = data.options.map((item) => `<option value="${item.value}">${item.label}</option>`).join("");
  }
  select.value = data.current || "whisper";
}

async function applyBackendSelection() {
  const select = $("asr_backend");
  if (!select) {
    return;
  }

  const backend = select.value;
  setRecordStatus(`正在切换 ASR 后端到 ${backend} ...`);
  const res = await fetchWithTimeout("/api/asr/backend", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ backend }),
  }, 30000);
  const data = await res.json();
  if (!res.ok || !data.ok) {
    throw new Error(data.detail || data.error || "切换 ASR 后端失败");
  }
  await fetchBackendOptions();
  await fetchHealth();
  setAsrStageLabels({});
  setAsrResult(`已切换到 ${data.backend}`, "", "", "");
  setRecordStatus(`ASR 后端已切换到 ${data.backend}`);
}

function setRecordStatus(text) {
  $("record_status").textContent = text;
}

function setAsrResult(meta, rawText, phraseText, finalText) {
  $("asr_meta").textContent = meta;
  $("asr_raw_text").value = rawText || "";
  $("asr_phrase_text").value = phraseText || "";
  $("asr_final_text").value = finalText || "";
}

function formatMs(value) {
  const ms = Number(value || 0);
  if (!Number.isFinite(ms) || ms <= 0) {
    return "--";
  }
  return `${ms.toFixed(ms >= 100 ? 0 : 1)} ms`;
}

function setAsrStageLabels(state) {
  $("asr_raw_label").textContent = `原始文本（${formatMs(state.asr_elapsed_ms)}）`;
  $("asr_phrase_label").textContent = `短语纠错后（${formatMs(state.phrase_elapsed_ms)}）`;
  $("asr_final_label").textContent = `大模型纠正后（${formatMs(state.llm_elapsed_ms)}）`;
}

function buildAsrMeta(data) {
  const ruleSummary = Array.isArray(data.applied_rules) ? data.applied_rules.map((item) => item.rule).join(", ") : "";
  const fallbackInfo = data.used_fallback_language ? ` | 回退: ${data.used_fallback_language}` : "";
  const llmInfo = data.llm_correction_applied ? " | 大模型纠错: 已执行" : "";
  const correctionError = data.correction_error ? ` | 大模型纠错异常: ${data.correction_error}` : "";
  const backendInfo = data.backend ? `后端: ${data.backend} | ` : "";
  const hotwordsInfo = Array.isArray(data.hotwords) && data.hotwords.length ? ` | 热词: ${data.hotwords.length}` : "";
  return `${backendInfo}业务域: ${data.domain || "unknown"} | 语言: ${data.language || "unknown"} | 置信: ${(Number(data.language_probability || 0) * 100).toFixed(1)}% | 时长: ${Number(data.duration_s || 0).toFixed(2)}s${fallbackInfo}${llmInfo}${correctionError}${hotwordsInfo}${ruleSummary ? ` | 规则: ${ruleSummary}` : ""}`;
}

function setTtsMeta(text) {
  $("tts_meta").textContent = text;
}

function toggleRecording(active) {
  $("start_record").disabled = active;
  $("stop_record").disabled = !active;
}

async function transcribeBlob(blob, filename) {
  const fd = new FormData();
  fd.append("file", blob, filename);
  fd.append("language", $("asr_language").value);

  setRecordStatus("识别中...");
  setAsrResult("正在调用 ASR", "", "", "");

  const res = await fetchWithTimeout("/api/asr/transcribe-stream", {
    method: "POST",
    body: fd,
  }, ASR_TIMEOUT_MS);
  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || "识别失败");
  }
  await consumeAsrStream(res);
}

async function consumeAsrStream(response) {
  if (!response.body) {
    throw new Error("浏览器不支持流式识别响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let currentState = {
    domain: "unknown",
    language: "unknown",
    language_probability: 0,
    duration_s: 0,
    used_fallback_language: "",
    raw_text: "",
    text_after_phrase: "",
    final_text: "",
    applied_rules: [],
    llm_correction_applied: false,
    correction_error: "",
    backend: "",
    hotwords: [],
    asr_elapsed_ms: 0,
    phrase_elapsed_ms: 0,
    confusion_elapsed_ms: 0,
    llm_elapsed_ms: 0,
    postprocess_elapsed_ms: 0,
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) {
        const event = JSON.parse(line);
        currentState = handleAsrStreamEvent(currentState, event);
      }
      newlineIndex = buffer.indexOf("\n");
    }

    if (done) {
      break;
    }
  }
}

function handleAsrStreamEvent(currentState, event) {
  if (event.event === "error") {
    throw new Error(event.detail || "识别失败");
  }

  const nextState = {
    ...currentState,
    ...event,
    raw_text: event.raw_text ?? currentState.raw_text,
    text_after_phrase: event.text_after_phrase ?? currentState.text_after_phrase,
    final_text: event.final_text ?? currentState.final_text,
    applied_rules: event.applied_rules ?? currentState.applied_rules,
    correction_error: event.correction_error ?? currentState.correction_error,
    llm_correction_applied: event.llm_correction_applied ?? currentState.llm_correction_applied,
    backend: event.backend ?? currentState.backend,
    hotwords: event.hotwords ?? currentState.hotwords,
    asr_elapsed_ms: event.asr_elapsed_ms ?? currentState.asr_elapsed_ms,
    phrase_elapsed_ms: event.phrase_elapsed_ms ?? currentState.phrase_elapsed_ms,
    confusion_elapsed_ms: event.confusion_elapsed_ms ?? currentState.confusion_elapsed_ms,
    llm_elapsed_ms: event.llm_elapsed_ms ?? currentState.llm_elapsed_ms,
    postprocess_elapsed_ms: event.postprocess_elapsed_ms ?? currentState.postprocess_elapsed_ms,
  };

  if (event.event === "raw_text") {
    setAsrStageLabels(nextState);
    setAsrResult(buildAsrMeta(nextState), nextState.raw_text, nextState.text_after_phrase, nextState.final_text);
    setRecordStatus("已收到原始文本，正在执行短语纠错...");
  } else if (event.event === "phrase_text") {
    setAsrStageLabels(nextState);
    setAsrResult(buildAsrMeta(nextState), nextState.raw_text, nextState.text_after_phrase, nextState.final_text);
    setRecordStatus("已完成短语纠错，正在执行混淆词和大模型纠错...");
  } else if (event.event === "final_text") {
    setAsrStageLabels(nextState);
    setAsrResult(buildAsrMeta(nextState), nextState.raw_text, nextState.text_after_phrase, nextState.final_text);
    setRecordStatus(nextState.correction_error ? "识别完成，已回退到规则纠错结果。" : "识别完成");
  } else if (event.event === "complete") {
    setAsrStageLabels(nextState);
    setAsrResult(buildAsrMeta(nextState), nextState.raw_text, nextState.text_after_phrase, nextState.final_text || nextState.text);
    setRecordStatus(nextState.correction_error ? "识别完成，已回退到规则纠错结果。" : "识别完成");
  }

  return nextState;
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("当前浏览器不支持录音。");
  }

  setRecordStatus("正在打开麦克风...");
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: getRequestedAudioConstraints(),
  });
  recordChunks = [];
  mediaRecorder = new MediaRecorder(mediaStream);
  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      recordChunks.push(event.data);
    }
  };
  mediaRecorder.start(250);
  toggleRecording(true);
  setRecordStatus("麦克风已打开，正在预热，请在提示音后开始说话...");
  await sleep(RECORD_START_WARMUP_MS);
  await playReadyTone();
  const audioTrack = mediaStream.getAudioTracks()[0];
  const enhancementInfo = formatCaptureEnhancement(audioTrack);
  const enhancementSuffix = enhancementInfo ? `（${enhancementInfo}）` : "";
  setRecordStatus(`已开始录音，请开始说话，点击“停止并识别”提交。${enhancementSuffix}`);
}

async function stopRecordingAndTranscribe() {
  if (!mediaRecorder) return;

  const done = new Promise((resolve) => {
    mediaRecorder.onstop = resolve;
  });

  $("stop_record").disabled = true;
  setRecordStatus("正在收尾录音，请稍等...");
  await sleep(RECORD_STOP_GUARD_MS);
  mediaRecorder.stop();
  await done;

  mediaStream.getTracks().forEach((track) => track.stop());
  const blob = new Blob(recordChunks, { type: mediaRecorder.mimeType || "audio/webm" });

  mediaRecorder = null;
  mediaStream = null;
  toggleRecording(false);

  await transcribeBlob(blob, "record.webm");
}

async function transcribeSelectedFile(file) {
  if (!file) return;
  await transcribeBlob(file, file.name);
}

async function synthesize() {
  const text = $("tts_text").value.trim();
  if (!text) {
    throw new Error("请输入要合成的文本。");
  }

  const fd = new FormData();
  fd.append("text", text);
  fd.append("language", $("tts_language").value);
  fd.append("speed", $("tts_speed").value || "1.0");

  setTtsMeta("合成中...");
  const res = await fetchWithTimeout("/api/tts/synthesize", {
    method: "POST",
    body: fd,
  }, TTS_TIMEOUT_MS);
  const data = await res.json();
  if (!res.ok || !data.ok) {
    const message = data.detail || data.error || "合成失败";
    throw new Error(message);
  }

  latestAudioUrl = data.audio_url;
  $("tts_player").src = latestAudioUrl;
  $("tts_download").href = latestAudioUrl;
  $("tts_download").classList.remove("disabled");
  $("play_latest").disabled = false;
  setTtsMeta(`语言: ${data.language} | 说话人: ${data.speaker || "default"} | 文件: ${data.file_name}`);
}

window.addEventListener("load", () => {
  Promise.all([fetchHealth(), fetchBackendOptions()]).catch((err) => {
    const healthBox = $("health_box");
    if (healthBox) {
      healthBox.textContent = `服务检查失败: ${err.message}`;
    }
  });

  $("start_record").addEventListener("click", async () => {
    try {
      await startRecording();
    } catch (err) {
      setRecordStatus(`录音失败: ${err.message}`);
    }
  });

  $("stop_record").addEventListener("click", async () => {
    try {
      await stopRecordingAndTranscribe();
    } catch (err) {
      setRecordStatus(`识别失败: ${err.message}`);
      toggleRecording(false);
    }
  });

  $("audio_file").addEventListener("change", async (event) => {
    try {
      const file = event.target.files && event.target.files[0];
      await transcribeSelectedFile(file);
    } catch (err) {
      setRecordStatus(`文件识别失败: ${err.message}`);
    } finally {
      event.target.value = "";
    }
  });

  $("clear_asr").addEventListener("click", () => {
    setAsrStageLabels({});
    setAsrResult("等待识别", "", "", "");
    setRecordStatus("可直接录音，或上传本地音频文件后识别。");
  });

  $("asr_backend").addEventListener("change", async () => {
    try {
      await applyBackendSelection();
    } catch (err) {
      setRecordStatus(`切换后端失败: ${err.message}`);
    }
  });

  $("tts_submit").addEventListener("click", async () => {
    try {
      await synthesize();
    } catch (err) {
      setTtsMeta(`合成失败: ${err.message}`);
    }
  });

  $("play_latest").addEventListener("click", () => {
    if (!latestAudioUrl) return;
    $("tts_player").play().catch(() => {});
  });

  setAsrStageLabels({});
});
