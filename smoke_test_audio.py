"""Microphone-to-Headphones live hardware smoke test."""

import sys
import time
import numpy as np

from app.audio.devices import get_input_devices, get_output_devices
from app.audio.pipeline import AudioPipeline
from app.config import AppConfig


def run_smoke_test(duration_seconds: int = 5):
    print("=" * 65)
    print("  VaniFlow Live Hardware Audio Smoke Test")
    print("=" * 65)

    inputs = get_input_devices()
    outputs = get_output_devices()

    print(f"Detected {len(inputs)} input device(s) and {len(outputs)} output device(s).")
    if not outputs:
        print("ERROR: No output devices available for playback.")
        sys.exit(1)

    cfg = AppConfig()
    cfg.mode = "dsp"
    cfg.dsp.pitch_factor = 1.60
    cfg.dsp.formant_shift = 1.15
    cfg.dsp.output_gain = 0.85
    cfg.dsp.dry_wet_mix = 1.0

    print(f"Active Mode: DSP (WORLD Vocoder)")
    print(f"Target Voice: Indian Female (Pitch: {cfg.dsp.pitch_factor}x, Formant: {cfg.dsp.formant_shift}x)")
    print(f"Starting pipeline for {duration_seconds} seconds...")

    pipeline = AudioPipeline(cfg)
    try:
        pipeline.start()
        print("Pipeline running! Processing live chunks...")

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            time.sleep(0.5)
            st = pipeline.status
            print(
                f"  Status: running={st.is_running} | "
                f"proc={st.processing_time_ms:.1f}ms | "
                f"rtf={st.real_time_factor:.2f} | "
                f"in_rms={st.input_rms:.4f} | "
                f"out_rms={st.output_rms:.4f} | "
                f"chunks={st.chunks_processed}"
            )

        print("\nStopping pipeline...")
        pipeline.stop()
        print("Pipeline stopped cleanly.")
        print("=" * 65)
        print("  SMOKE TEST RESULT: PASSED")
        print("=" * 65)

    except Exception as e:
        print(f"\nSmoke test error: {e}")
        try:
            pipeline.stop()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_smoke_test(dur)
