"""Benchmark script measuring CPU performance, latency, RAM usage, and RTF."""

import os
import sys
import time
import numpy as np
import psutil

from app.config import DSPConfig
from app.conversion.dsp_converter import DSPConverter


def run_benchmark(num_chunks: int = 100, chunk_duration_ms: int = 300, sample_rate: int = 16000):
    print("=" * 65)
    print("  VaniFlow Audio Engine Benchmark — Local Windows 10 (CPU)")
    print("=" * 65)

    proc = psutil.Process(os.getpid())
    ram_initial_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"Initial Process RAM: {ram_initial_mb:.1f} MB")

    chunk_samples = int(sample_rate * (chunk_duration_ms / 1000.0))
    pad_samples = int(sample_rate * 0.15)  # 150ms padding
    window_samples = chunk_samples + pad_samples

    print(f"Sample Rate:          {sample_rate} Hz")
    print(f"Chunk Duration:       {chunk_duration_ms} ms ({chunk_samples} samples)")
    print(f"Analysis Window:      {window_samples / sample_rate * 1000:.1f} ms ({window_samples} samples)")
    print(f"Test Iterations:      {num_chunks} chunks ({num_chunks * chunk_duration_ms / 1000:.1f} seconds of audio)")
    print("-" * 65)

    # Instantiate converter
    cfg = DSPConfig(engine="world", pitch_factor=1.60, formant_shift=1.15)
    converter = DSPConverter(cfg, sample_rate=sample_rate)

    # Generate synthetic male voice test signal: fundamental at 125 Hz + harmonics
    t = np.arange(window_samples) / float(sample_rate)
    test_signal = (
        0.4 * np.sin(2 * np.pi * 125 * t)
        + 0.3 * np.sin(2 * np.pi * 250 * t)
        + 0.2 * np.sin(2 * np.pi * 375 * t)
        + 0.1 * np.sin(2 * np.pi * 500 * t)
    ).astype(np.float32)

    # Warmup
    for _ in range(5):
        converter.process(test_signal)

    inference_times_ms = []

    print("Executing benchmark stream...")
    t_start_total = time.perf_counter()

    for i in range(num_chunks):
        # Alternate harmonic speech and conversational pauses (silence)
        if i % 10 == 9:
            inp = np.zeros(window_samples, dtype=np.float32)
        else:
            inp = test_signal

        t0 = time.perf_counter()
        out = converter.process(inp)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        inference_times_ms.append(elapsed_ms)

        # Sanity assertions
        assert len(out) == len(inp), "Length mismatch!"
        assert np.all(np.isfinite(out)), "Non-finite output detected!"

    t_end_total = time.perf_counter()
    total_time_s = t_end_total - t_start_total

    ram_final_mb = proc.memory_info().rss / (1024 * 1024)
    ram_diff_mb = ram_final_mb - ram_initial_mb

    avg_proc_ms = np.mean(inference_times_ms)
    p50_proc_ms = np.percentile(inference_times_ms, 50)
    p95_proc_ms = np.percentile(inference_times_ms, 95)
    p99_proc_ms = np.percentile(inference_times_ms, 99)
    min_proc_ms = np.min(inference_times_ms)
    max_proc_ms = np.max(inference_times_ms)

    rtf = avg_proc_ms / chunk_duration_ms
    total_audio_simulated_s = (num_chunks * chunk_duration_ms) / 1000.0

    print("=" * 65)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 65)
    print(f"Total Audio Simulated:     {total_audio_simulated_s:.2f} seconds")
    print(f"Total Wall-clock Time:     {total_time_s:.2f} seconds")
    print(f"Chunk Duration (Budget):   {chunk_duration_ms:.1f} ms")
    print(f"Avg Processing Time:       {avg_proc_ms:.2f} ms")
    print(f"Median Processing Time:    {p50_proc_ms:.2f} ms")
    print(f"95th Percentile Latency:   {p95_proc_ms:.2f} ms")
    print(f"99th Percentile Latency:   {p99_proc_ms:.2f} ms")
    print(f"Min / Max Processing Time: {min_proc_ms:.2f} ms / {max_proc_ms:.2f} ms")
    print(f"Real-Time Factor (RTF):    {rtf:.3f}x (< 1.0 = Sustainable Real-Time)")
    print(f"Headroom Ratio:            {1.0 / rtf:.2f}x faster than real-time")
    print(f"Final Process RAM:         {ram_final_mb:.1f} MB (Delta: +{ram_diff_mb:.1f} MB)")
    print("=" * 65)

    if rtf < 1.0:
        print("  VERDICT: PASS - Audio engine achieves sustainable low-latency real-time conversion on CPU!")
    else:
        print("  VERDICT: FAIL - Processing exceeds real-time audio budget.")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
