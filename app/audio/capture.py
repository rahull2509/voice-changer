"""Audio capture module using sounddevice InputStream with resampling and ring buffer."""

import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from app.audio.devices import resolve_input_device, validate_device

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures microphone audio, handles mono-downmix, resampling, and chunking."""

    def __init__(
        self,
        device_id: Optional[int] = None,
        target_sample_rate: int = 16000,
        chunk_duration_ms: int = 300,
        max_queue_size: int = 20,
        blocksize: int = 1024,
    ) -> None:
        self.device_id = resolve_input_device(device_id)
        self.target_sample_rate = target_sample_rate
        self.chunk_size = int(target_sample_rate * (chunk_duration_ms / 1000.0))
        self.max_queue_size = max_queue_size
        self.blocksize = blocksize

        self.stream: Optional[sd.InputStream] = None
        self.device_sample_rate: int = target_sample_rate
        self.input_channels: int = 1
        self.is_running: bool = False

        self.output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self._buffer: list[np.ndarray] = []
        self._buffer_len: int = 0
        self._lock = threading.Lock()

        # Stats
        self.total_frames_captured: int = 0
        self.overflow_count: int = 0
        self.last_rms: float = 0.0

    def _determine_device_settings(self) -> None:
        """Query device capabilities and select appropriate sample rate and channels."""
        try:
            if self.device_id is None:
                self.device_id = resolve_input_device(None)
            if self.device_id is None:
                raise ValueError("No valid audio input device detected on this system.")

            info = sd.query_devices(self.device_id)

            dev_channels = int(info["max_input_channels"])
            dev_sr = int(info["default_samplerate"])

            # Check if target sample rate and mono are supported directly
            try:
                sd.check_input_settings(
                    device=self.device_id,
                    channels=1,
                    samplerate=self.target_sample_rate,
                )
                self.input_channels = 1
                self.device_sample_rate = self.target_sample_rate
            except Exception:
                # Use native device parameters and resample/downmix in software
                self.input_channels = dev_channels
                self.device_sample_rate = dev_sr

            logger.info(
                "AudioCapture setup: device='%s' (id=%s), hw_sr=%d, target_sr=%d, channels=%d",
                info["name"],
                self.device_id,
                self.device_sample_rate,
                self.target_sample_rate,
                self.input_channels,
            )
        except Exception as e:
            logger.warning("Could not probe device settings directly (%s); using defaults", e)
            self.device_sample_rate = self.target_sample_rate
            self.input_channels = 1

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags
    ) -> None:
        """sounddevice real-time input callback."""
        if status:
            if status.input_overflow:
                self.overflow_count += 1
                logger.warning("Audio capture input overflow detected (%d total)", self.overflow_count)

        if not self.is_running:
            return

        # Downmix to mono if stereo/multi-channel
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = np.mean(indata, axis=1, dtype=np.float32)
        else:
            mono = indata.flatten().astype(np.float32)

        # Measure RMS level for VU / silence gating
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size > 0 else 0.0
        self.last_rms = rms

        # Resample if hardware rate != target rate
        if self.device_sample_rate != self.target_sample_rate:
            from math import gcd
            common = gcd(self.target_sample_rate, self.device_sample_rate)
            up = self.target_sample_rate // common
            down = self.device_sample_rate // common
            processed = resample_poly(mono, up, down).astype(np.float32)
        else:
            processed = mono

        self.total_frames_captured += len(processed)

        # Accumulate samples in internal buffer until we have a full chunk
        with self._lock:
            self._buffer.append(processed)
            self._buffer_len += len(processed)

            while self._buffer_len >= self.chunk_size:
                # Merge accumulated blocks
                merged = np.concatenate(self._buffer)
                chunk = merged[: self.chunk_size]
                remaining = merged[self.chunk_size :]

                if len(remaining) > 0:
                    self._buffer = [remaining]
                    self._buffer_len = len(remaining)
                else:
                    self._buffer = []
                    self._buffer_len = 0

                # Push chunk to output queue (drop oldest if full)
                try:
                    self.output_queue.put_nowait(chunk)
                except queue.Full:
                    try:
                        self.output_queue.get_nowait()
                        self.overflow_count += 1
                    except queue.Empty:
                        pass
                    self.output_queue.put_nowait(chunk)

    def start(self) -> None:
        """Start capturing microphone audio."""
        if self.is_running:
            return

        self._determine_device_settings()

        with self._lock:
            self._buffer.clear()
            self._buffer_len = 0

        # Drain queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        self.stream = sd.InputStream(
            device=self.device_id,
            channels=self.input_channels,
            samplerate=self.device_sample_rate,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.is_running = True
        self.stream.start()
        logger.info("Audio capture started successfully.")

    def stop(self) -> None:
        """Stop capturing microphone audio and close stream."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning("Error stopping capture stream: %s", e)
            finally:
                self.stream = None
        logger.info("Audio capture stopped.")

    def get_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Fetch the next audio chunk from the capture queue."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
