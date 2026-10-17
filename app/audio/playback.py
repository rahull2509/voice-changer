"""Audio playback module using sounddevice OutputStream with resampling, channel duplication, and soft limiting."""

import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from app.audio.devices import resolve_output_device, validate_device

logger = logging.getLogger(__name__)


class AudioPlayback:
    """Consumes audio chunks, applies limiter, resamples, and outputs to speakers/headphones."""

    def __init__(
        self,
        device_id: Optional[int] = None,
        source_sample_rate: int = 16000,
        max_queue_size: int = 30,
        blocksize: int = 1024,
    ) -> None:
        self.device_id = resolve_output_device(device_id)
        self.source_sample_rate = source_sample_rate
        self.max_queue_size = max_queue_size
        self.blocksize = blocksize

        self.stream: Optional[sd.OutputStream] = None
        self.device_sample_rate: int = source_sample_rate
        self.output_channels: int = 2
        self.is_running: bool = False

        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self._playback_buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

        # Stats
        self.total_frames_played: int = 0
        self.underrun_count: int = 0
        self.overflow_count: int = 0

    def _determine_device_settings(self) -> None:
        """Query output device capabilities and select appropriate settings."""
        try:
            if self.device_id is None:
                self.device_id = resolve_output_device(None)
            if self.device_id is None:
                raise ValueError("No valid audio output device detected on this system.")

            info = sd.query_devices(self.device_id)

            max_out = int(info["max_output_channels"])
            if max_out <= 0:
                raise ValueError(f"Selected output device '{info['name']}' has 0 output channels.")

            self.output_channels = min(max_out, 2)
            dev_sr = int(info["default_samplerate"])

            # Test target sample rate directly
            try:
                sd.check_output_settings(
                    device=self.device_id,
                    channels=self.output_channels,
                    samplerate=self.source_sample_rate,
                )
                self.device_sample_rate = self.source_sample_rate
            except Exception:
                self.device_sample_rate = dev_sr

            logger.info(
                "AudioPlayback setup: device='%s' (id=%s), hw_sr=%d, src_sr=%d, channels=%d",
                info["name"],
                self.device_id,
                self.device_sample_rate,
                self.source_sample_rate,
                self.output_channels,
            )
        except Exception as e:
            logger.warning("Could not probe output device (%s); using defaults", e)
            self.device_sample_rate = self.source_sample_rate
            self.output_channels = 2

    @staticmethod
    def _soft_limit(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Apply smooth non-linear limiting to prevent digital clipping without harsh distortion."""
        peak = np.max(np.abs(audio)) if audio.size > 0 else 0.0
        if peak > threshold:
            # Soft-knee tanh compression on peaks
            scale = threshold / np.tanh(peak / threshold)
            return np.tanh(audio / threshold) * threshold
        return audio

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags
    ) -> None:
        """sounddevice real-time output callback."""
        if status:
            if status.output_underflow:
                self.underrun_count += 1

        if not self.is_running:
            outdata.fill(0)
            return

        with self._lock:
            # While buffer doesn't have enough frames for this callback, pull from input_queue
            while len(self._playback_buffer) < frames:
                try:
                    chunk = self.input_queue.get_nowait()
                except queue.Empty:
                    break

                # Ensure mono 1D float32
                if chunk.ndim > 1:
                    chunk = chunk.mean(axis=1)
                chunk = chunk.astype(np.float32)

                # Resample from source rate to device rate if needed
                if self.device_sample_rate != self.source_sample_rate:
                    from math import gcd
                    common = gcd(self.source_sample_rate, self.device_sample_rate)
                    up = self.device_sample_rate // common
                    down = self.source_sample_rate // common
                    chunk = resample_poly(chunk, up, down).astype(np.float32)

                # Append to playback buffer
                if len(self._playback_buffer) == 0:
                    self._playback_buffer = chunk
                else:
                    self._playback_buffer = np.concatenate([self._playback_buffer, chunk])

            # Provide samples to sounddevice
            avail = len(self._playback_buffer)
            if avail >= frames:
                take = self._playback_buffer[:frames]
                self._playback_buffer = self._playback_buffer[frames:]
            elif avail > 0:
                take = np.pad(self._playback_buffer, (0, frames - avail), mode="constant")
                self._playback_buffer = np.zeros(0, dtype=np.float32)
                self.underrun_count += 1
            else:
                take = np.zeros(frames, dtype=np.float32)
                self.underrun_count += 1

        # Apply soft limiting to output chunk
        take = self._soft_limit(take)

        # Multi-channel routing: duplicate to stereo if needed
        if self.output_channels == 2:
            outdata[:, 0] = take
            outdata[:, 1] = take
        else:
            outdata[:, 0] = take

        self.total_frames_played += frames

    def put_chunk(self, chunk: np.ndarray) -> bool:
        """Queue a converted audio chunk for playback."""
        if not self.is_running:
            return False
        try:
            self.input_queue.put_nowait(chunk)
            return True
        except queue.Full:
            try:
                self.input_queue.get_nowait()  # Drop oldest
                self.overflow_count += 1
            except queue.Empty:
                pass
            self.input_queue.put_nowait(chunk)
            return True

    def start(self) -> None:
        """Start audio output stream."""
        if self.is_running:
            return

        self._determine_device_settings()

        with self._lock:
            self._playback_buffer = np.zeros(0, dtype=np.float32)

        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break

        self.stream = sd.OutputStream(
            device=self.device_id,
            channels=self.output_channels,
            samplerate=self.device_sample_rate,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.is_running = True
        self.stream.start()
        logger.info("Audio playback started successfully.")

    def stop(self) -> None:
        """Stop audio output stream and clear buffers."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning("Error stopping playback stream: %s", e)
            finally:
                self.stream = None
        logger.info("Audio playback stopped.")
