"""Real-time audio processing pipeline coordinating capture, conversion, and playback."""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.audio.capture import AudioCapture
from app.audio.playback import AudioPlayback
from app.config import AppConfig
from app.conversion.base import BaseConverter
from app.conversion.dsp_converter import DSPConverter
from app.conversion.rvc_converter import RVCConverter
from app.conversion.voice_cloning_converter import VoiceCloningConverter

logger = logging.getLogger(__name__)


@dataclass
class PipelineStatus:
    """Current health, configuration, and performance metrics of the pipeline."""

    is_running: bool = False
    active_mode: str = "dsp"
    converter_name: str = "None"
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = 16000
    chunk_duration_ms: int = 300
    processing_time_ms: float = 0.0
    buffer_delay_ms: float = 0.0
    total_latency_ms: float = 0.0
    real_time_factor: float = 0.0
    input_rms: float = 0.0
    output_rms: float = 0.0
    overflow_count: int = 0
    underrun_count: int = 0
    chunks_processed: int = 0
    overload: bool = False
    overload_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_running": self.is_running,
            "active_mode": self.active_mode,
            "converter_name": self.converter_name,
            "input_device": self.input_device,
            "output_device": self.output_device,
            "sample_rate": self.sample_rate,
            "chunk_duration_ms": self.chunk_duration_ms,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "buffer_delay_ms": round(self.buffer_delay_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "real_time_factor": round(self.real_time_factor, 3),
            "input_rms": round(self.input_rms, 4),
            "output_rms": round(self.output_rms, 4),
            "overflow_count": self.overflow_count,
            "underrun_count": self.underrun_count,
            "chunks_processed": self.chunks_processed,
            "overload": self.overload,
            "overload_count": self.overload_count,
            "last_error": self.last_error,
        }


class AudioPipeline:
    """Thread-safe real-time audio pipeline managing streaming capture, conversion, and playback."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.status = PipelineStatus(
            active_mode=config.mode,
            sample_rate=config.audio.internal_sample_rate,
            chunk_duration_ms=config.audio.chunk_duration_ms,
        )

        self.capture: Optional[AudioCapture] = None
        self.playback: Optional[AudioPlayback] = None
        self.converter: Optional[BaseConverter] = None

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # Overlap & crossfade state
        self._prev_pad: Optional[np.ndarray] = None
        self._prev_tail: Optional[np.ndarray] = None

        # Hann crossfade table
        self._crossfade_samples = int(
            config.audio.internal_sample_rate * (config.audio.crossfade_ms / 1000.0)
        )
        if self._crossfade_samples > 0:
            fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(self._crossfade_samples) / self._crossfade_samples))
            self._fade_in = fade_in.astype(np.float32)
            self._fade_out = (1.0 - fade_in).astype(np.float32)
        else:
            self._fade_in = np.zeros(0, dtype=np.float32)
            self._fade_out = np.zeros(0, dtype=np.float32)

    def _init_converter(self) -> BaseConverter:
        """Instantiate the active converter based on configuration."""
        if self.config.mode == "dsp":
            return DSPConverter(
                config=self.config.dsp,
                sample_rate=self.config.audio.internal_sample_rate,
            )
        elif self.config.mode == "rvc":
            return RVCConverter(
                config=self.config.rvc,
                sample_rate=self.config.audio.internal_sample_rate,
            )
        elif self.config.mode == "voice_cloning":
            return VoiceCloningConverter(
                config=self.config.voice_cloning,
                sample_rate=self.config.audio.internal_sample_rate,
            )
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    def start(self) -> None:
        """Start the audio capture, processing worker, and playback stream."""
        with self._lock:
            if self.status.is_running:
                logger.warning("Pipeline is already running.")
                return

            self.status.last_error = None
            try:
                # 1. Initialize converter
                self.converter = self._init_converter()
                self.status.converter_name = self.converter.name

                sr = self.config.audio.internal_sample_rate
                chunk_ms = self.config.audio.chunk_duration_ms

                # 2. Initialize capture
                self.capture = AudioCapture(
                    device_id=self.config.audio.input_device,
                    target_sample_rate=sr,
                    chunk_duration_ms=chunk_ms,
                    max_queue_size=self.config.audio.queue_max_size,
                    blocksize=self.config.audio.blocksize,
                )

                # 3. Initialize playback
                self.playback = AudioPlayback(
                    device_id=self.config.audio.output_device,
                    source_sample_rate=sr,
                    max_queue_size=self.config.audio.queue_max_size,
                    blocksize=self.config.audio.blocksize,
                )

                # Reset state
                self._stop_event.clear()
                self._prev_pad = None
                self._prev_tail = None
                self.status.chunks_processed = 0
                self.status.overflow_count = 0
                self.status.underrun_count = 0

                # Start hardware streams
                self.capture.start()
                self.playback.start()

                # Start background processing thread
                self._worker_thread = threading.Thread(
                    target=self._processing_loop,
                    name="AudioPipelineWorker",
                    daemon=True,
                )
                self._worker_thread.start()

                self.status.is_running = True
                self.status.input_device = self.capture.device_id if self.capture else None
                self.status.output_device = self.playback.device_id if self.playback else None
                self.status.active_mode = self.config.mode

                logger.info("Pipeline started successfully in %s mode.", self.config.mode)

            except Exception as e:
                self.status.last_error = str(e)
                logger.exception("Failed to start pipeline: %s", e)
                self._cleanup_streams()
                raise

    def stop(self) -> None:
        """Stop processing worker and close audio streams cleanly."""
        with self._lock:
            if not self.status.is_running:
                return

            logger.info("Stopping audio pipeline...")
            self._stop_event.set()

            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

            self._cleanup_streams()

            if self.converter is not None:
                try:
                    self.converter.cleanup()
                except Exception as e:
                    logger.warning("Error during converter cleanup: %s", e)
                self.converter = None

            self.status.is_running = False
            self.status.input_rms = 0.0
            self.status.output_rms = 0.0
            logger.info("Audio pipeline stopped.")

    def _cleanup_streams(self) -> None:
        """Safely close capture and playback streams."""
        if self.capture is not None:
            try:
                self.capture.stop()
            except Exception as e:
                logger.warning("Error stopping capture: %s", e)
            self.capture = None

        if self.playback is not None:
            try:
                self.playback.stop()
            except Exception as e:
                logger.warning("Error stopping playback: %s", e)
            self.playback = None

    def _processing_loop(self) -> None:
        """Background thread worker fetching chunks, converting, and feeding playback."""
        try:
            sr = self.config.audio.internal_sample_rate
            pad_samples = int(sr * (self.config.audio.pad_duration_ms / 1000.0))
            chunk_samples = int(sr * (self.config.audio.chunk_duration_ms / 1000.0))
            cf_samples = self._crossfade_samples

            while not self._stop_event.is_set():
                if self.capture is None or self.playback is None or self.converter is None:
                    break

                chunk = self.capture.get_chunk(timeout=0.1)
                if chunk is None:
                    continue

                t_start = time.perf_counter()

                # Input RMS level
                in_rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) if chunk.size > 0 else 0.0
                self.status.input_rms = in_rms

                # Construct analysis window with left context padding
                if self._prev_pad is not None and len(self._prev_pad) == pad_samples:
                    window = np.concatenate([self._prev_pad, chunk])
                    pad_offset = pad_samples
                else:
                    window = np.pad(chunk, (pad_samples, 0), mode="reflect")
                    pad_offset = pad_samples

                # Save tail for next window's padding
                if len(chunk) >= pad_samples:
                    self._prev_pad = chunk[-pad_samples:].copy()
                else:
                    self._prev_pad = np.pad(chunk, (pad_samples - len(chunk), 0), mode="constant")

                # Execute voice conversion
                try:
                    processed_window = self.converter.process(window)
                except Exception as e:
                    logger.exception("Error in converter: %s", e)
                    self.status.last_error = f"Converter error: {e}"
                    processed_window = window

                # Extract current chunk segment
                if len(processed_window) >= pad_offset + len(chunk):
                    processed_chunk = processed_window[pad_offset : pad_offset + len(chunk)]
                else:
                    processed_chunk = processed_window[-len(chunk) :]

                # Apply Crossfade with previous chunk boundary to eliminate discontinuities
                if self._prev_tail is not None and cf_samples > 0 and len(processed_chunk) >= cf_samples:
                    processed_chunk[:cf_samples] = (
                        processed_chunk[:cf_samples] * self._fade_in
                        + self._prev_tail * self._fade_out
                    )

                if len(processed_chunk) >= cf_samples:
                    self._prev_tail = processed_chunk[-cf_samples:].copy()
                else:
                    self._prev_tail = None

                # Dry / Wet mixing
                wet = self.config.dsp.dry_wet_mix if self.config.mode == "dsp" else 1.0
                if wet < 0.999:
                    mixed = (1.0 - wet) * chunk + wet * processed_chunk
                else:
                    mixed = processed_chunk

                # Output gain
                gain = self.config.dsp.output_gain if self.config.mode == "dsp" else 1.0
                out_chunk = (mixed * gain).astype(np.float32)

                # Output RMS level
                out_rms = float(np.sqrt(np.mean(out_chunk.astype(np.float64) ** 2))) if out_chunk.size > 0 else 0.0
                self.status.output_rms = out_rms

                # Queue chunk for playback
                self.playback.put_chunk(out_chunk)

                # Performance telemetry
                t_elapsed = (time.perf_counter() - t_start) * 1000.0  # ms
                chunk_ms = float(self.config.audio.chunk_duration_ms)
                rtf = t_elapsed / chunk_ms if chunk_ms > 0 else 0.0

                # Estimate buffer delays
                in_q_size = self.capture.output_queue.qsize() if self.capture else 0
                out_q_size = self.playback.input_queue.qsize() if self.playback else 0
                buf_delay = (in_q_size + out_q_size) * chunk_ms
                tot_latency = buf_delay + t_elapsed

                # Update metrics
                self.status.processing_time_ms = t_elapsed
                self.status.buffer_delay_ms = buf_delay
                self.status.total_latency_ms = tot_latency
                self.status.real_time_factor = rtf
                self.status.chunks_processed += 1
                if rtf > 1.0:
                    self.status.overload = True
                    self.status.overload_count += 1
                else:
                    self.status.overload = False
                if self.capture:
                    self.status.overflow_count = self.capture.overflow_count
                if self.playback:
                    self.status.underrun_count = self.playback.underrun_count
        except Exception as e:
            logger.exception("Audio pipeline worker crashed: %s", e)
            self.status.last_error = f"Worker crash: {e}"

    def update_settings(self, new_config: AppConfig) -> None:
        """Update settings dynamically while running if possible."""
        with self._lock:
            need_restart = (
                new_config.mode != self.config.mode
                or new_config.audio.input_device != self.config.audio.input_device
                or new_config.audio.output_device != self.config.audio.output_device
                or new_config.audio.internal_sample_rate != self.config.audio.internal_sample_rate
                or new_config.audio.chunk_duration_ms != self.config.audio.chunk_duration_ms
            )

            self.config = new_config

            if self.status.is_running:
                if need_restart:
                    logger.info("Configuration change requires pipeline restart.")
                    self.stop()
                    self.start()
                else:
                    # Update live DSP parameters without restarting streams
                    if self.converter is not None and isinstance(self.converter, DSPConverter):
                        self.converter.config = new_config.dsp
                        self.converter._engine = new_config.dsp.engine
                        logger.info("Updated DSP converter live settings.")
