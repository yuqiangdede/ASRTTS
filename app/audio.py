from __future__ import annotations

import logging
import wave
from pathlib import Path

import av
import numpy as np


HEAD_SILENCE_MS = 320
TAIL_SILENCE_MS = 900
SPLIT_FRAME_MS = 20
SPLIT_MIN_SILENCE_MS = 220
SPLIT_MIN_SEGMENT_MS = 900
SPLIT_MAX_SEGMENTS = 4
SPLIT_KEEP_SILENCE_MS = 120
SPLIT_SILENCE_THRESHOLD = 520
logger = logging.getLogger(__name__)


def decode_audio_to_pcm16(audio_path: str | Path, *, sample_rate: int = 16000) -> tuple[bytes, int, float]:
    path = str(Path(audio_path).resolve())
    container = av.open(path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        raise RuntimeError("未找到音频流。")

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks: list[bytes] = []
    skipped_packets = 0

    def append_frame(frame) -> None:
        for out_frame in resampler.resample(frame):
            arr = out_frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr[0]
            chunks.append(np.asarray(arr, dtype=np.int16).tobytes())

    try:
        for packet in container.demux(stream):
            try:
                for frame in packet.decode():
                    append_frame(frame)
            except av.error.InvalidDataError:
                skipped_packets += 1
                continue

        append_frame(None)
    finally:
        container.close()

    pcm = b"".join(chunks)
    if skipped_packets:
        logger.warning("音频解码跳过了损坏的数据包：path=%s skipped_packets=%s", path, skipped_packets)
    if not pcm:
        raise RuntimeError("音频解码失败：未能从文件中提取有效音频帧。")
    duration = (len(pcm) / 2.0 / sample_rate) if pcm else 0.0
    if pcm:
        head_padding = np.zeros(int(sample_rate * HEAD_SILENCE_MS / 1000.0), dtype=np.int16).tobytes()
        tail_padding = np.zeros(int(sample_rate * TAIL_SILENCE_MS / 1000.0), dtype=np.int16).tobytes()
        pcm = head_padding + pcm + tail_padding
    return pcm, sample_rate, duration


def write_pcm16_wav(
    audio_path: str | Path,
    pcm16_bytes: bytes,
    *,
    sample_rate: int = 16000,
    suffix: str = ".normalized.wav",
    output_path: str | Path | None = None,
) -> Path:
    src_path = Path(audio_path).resolve()
    target = Path(output_path).resolve() if output_path is not None else src_path.with_suffix(suffix)
    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_bytes)
    return target


def split_pcm16_by_silence(
    pcm16_bytes: bytes,
    *,
    sample_rate: int = 16000,
    max_segments: int = SPLIT_MAX_SEGMENTS,
) -> list[bytes]:
    samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
    if samples.size == 0:
        return []

    frame_samples = max(1, int(sample_rate * SPLIT_FRAME_MS / 1000.0))
    min_silence_frames = max(1, int(SPLIT_MIN_SILENCE_MS / SPLIT_FRAME_MS))
    min_segment_samples = max(frame_samples, int(sample_rate * SPLIT_MIN_SEGMENT_MS / 1000.0))
    keep_silence_samples = int(sample_rate * SPLIT_KEEP_SILENCE_MS / 1000.0)
    usable = (samples.size // frame_samples) * frame_samples
    if usable < frame_samples * 2:
        return [pcm16_bytes]

    framed = samples[:usable].reshape(-1, frame_samples).astype(np.int32)
    energy = np.mean(np.abs(framed), axis=1)
    silent = energy <= SPLIT_SILENCE_THRESHOLD

    candidates: list[tuple[int, int]] = []
    idx = 0
    while idx < silent.size:
        if not silent[idx]:
            idx += 1
            continue
        start = idx
        while idx < silent.size and silent[idx]:
            idx += 1
        end = idx
        if (end - start) < min_silence_frames:
            continue
        split_sample = ((start + end) // 2) * frame_samples
        if split_sample < min_segment_samples or (samples.size - split_sample) < min_segment_samples:
            continue
        candidates.append(((end - start) * frame_samples, split_sample))

    if not candidates:
        return [pcm16_bytes]

    selected: list[int] = []
    for _, split_sample in sorted(candidates, key=lambda item: item[0], reverse=True):
        tentative = sorted([*selected, split_sample])
        prev = 0
        valid = True
        for point in [*tentative, samples.size]:
            if (point - prev) < min_segment_samples:
                valid = False
                break
            prev = point
        if not valid:
            continue
        selected = tentative
        if len(selected) >= max_segments - 1:
            break

    if not selected:
        return [pcm16_bytes]

    segments: list[bytes] = []
    start = 0
    boundaries = [*selected, samples.size]
    for end in boundaries:
        left = max(0, start - keep_silence_samples)
        right = min(samples.size, end + keep_silence_samples)
        segment = samples[left:right]
        if segment.size > 0:
            segments.append(segment.astype(np.int16).tobytes())
        start = end

    if len(segments) < 2:
        return [pcm16_bytes]
    return segments[:max_segments]
