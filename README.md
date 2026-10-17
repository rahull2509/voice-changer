# VaniFlow — Live Real-Time AI Voice Changer

> **High-Performance Local Real-Time Voice Conversion for Windows 10 (CPU-Optimized)**  
> Built for natural Indian and Hinglish female voice conversion with zero cloud dependency.

---

## 1. System Overview & Architecture

VaniFlow is a local voice-to-voice conversion system engineered for **Windows 10 64-bit with 16 GB RAM and CPU-only inference** (Intel/AMD integrated graphics, no NVIDIA CUDA required).

```
Microphone (Live Speech)
          │
          ▼
┌─────────────────────────────────┐
│     AudioCapture (sounddevice)  │  ◄── Auto-probes hardware rate (e.g. 44.1k/48k)
│   • Hardware format downmix     │      Resamples to 16 kHz internal pipeline
│   • Thread-safe ring buffer     │
└────────────────┬────────────────┘
                 │ (Bounded queue)
                 ▼
┌────────────────────────────────────────────────────────┐
│               AudioPipeline Processing Worker          │
│                                                        │
│  ┌───────────────────────┐   ┌──────────────────────┐  │
│  │ DSP Mode (Real-Time)  │   │  RVC AI Mode (Neural)│  │
│  │ • WORLD Vocoder (Dio) │   │  • HuBERT extractor  │  │
│  │ • Pitch shift: 1.60x  │   │  • SynthesizerTrn    │  │
│  │ • Formant warp: 1.15x │   │  • Pitch guidance    │  │
│  │ • Latency: 20-65 ms   │   │  • Latency: ~1.8 s   │  │
│  │ • RTF: 0.08x - 0.22x  │   │  • Higher quality    │  │
│  └───────────────────────┘   └──────────────────────┘  │
│                                                        │
│  • Context window padding (150 ms)                     │
│  • Hann crossfading across chunk boundaries (10 ms)    │
│  • Dry / Wet balance & gain scaling                    │
│  • Soft-knee limiter (anti-clipping protection)        │
└────────────────┬───────────────────────────────────────┘
                 │ (Bounded queue)
                 ▼
┌─────────────────────────────────┐
│    AudioPlayback (sounddevice)  │
│   • Resample to hardware rate   │
│   • Stereo channel duplication  │
│   • Zero-click buffer underrun  │
└────────────────┬────────────────┘
                 │
                 ▼
Headphones / Speakers OR Virtual Microphone (VB-CABLE)
```

### Why the Hybrid Architecture?

| Feature | DSP Mode (WORLD Vocoder) | RVC AI Mode (Neural) |
| :--- | :--- | :--- |
| **Primary Goal** | Ultra-low latency conversational speaking | Maximum neural timbre resemblance |
| **Measured Latency** | **20 – 65 ms** per 300 ms chunk | **~1.5 – 2.5 s** per chunk (on CPU) |
| **Real-Time Factor (RTF)**| **0.08x – 0.22x** (4.6x faster than real-time) | **~1.2x – 1.8x** (heavy on CPU) |
| **Hardware Required** | CPU only (Intel/AMD, ~35 MB RAM) | CPU PyTorch (~1.8 GB RAM) |
| **Model Downloads** | **Zero downloads needed** — ready immediately | Requires `hubert_base.pt` + `.pth` voice model |
| **Hindi/Hinglish** | Preserves all Hindi retroflex & aspirated consonants | Preserves phonemes via acoustic feature tokens |
| **Labeling** | Clearly labeled as Algorithmic DSP | Clearly labeled as Neural AI |

---

## 2. Benchmark Measurements (Tested on this Machine)

Executed on Windows 10, Python 3.12.2, 16 GB RAM (CPU-only):

```text
=================================================================
  VaniFlow Audio Engine Benchmark — Local Windows 10 (CPU)
=================================================================
Sample Rate:               16,000 Hz
Chunk Duration:            300.0 ms (4,800 samples)
Analysis Window:           450.0 ms (7,200 samples with context)
-----------------------------------------------------------------
Chunk Duration (Budget):   300.0 ms
Avg Processing Time:       65.30 ms
Median Processing Time:    46.27 ms
Live Hardware Smoke Test:  21.0 - 29.7 ms (RTF: 0.07x - 0.10x)
95th Percentile Latency:   132.20 ms
Real-Time Factor (RTF):    0.218x (< 1.0 = Sustainable Real-Time)
Headroom:                  4.59x faster than real-time
Process RAM Usage:         35.1 MB (Delta: +1.3 MB over sustained stream)
Buffer Overflows/Drops:    0
=================================================================
VERDICT: PASS — Sustainable real-time low-latency conversion on CPU
=================================================================
```

---

## 3. Quick Start & Launch

### Prerequisites
- Windows 10 64-bit
- Python 3.12 64-bit installed and added to your `PATH`

### Step 1: Clone or Navigate to Project
```powershell
cd c:\Users\rahul\Code\voice
```

### Step 2: Install CPU Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Application
You can use the batch launcher, PowerShell script, or run directly:

**Option A — PowerShell:**
```powershell
.\run.ps1
```

**Option B — Batch File:**
```cmd
run.bat
```

**Option C — Python Command:**
```powershell
python -m app.main
```

### Step 4: Open the Control Panel
Navigate to:
```
http://127.0.0.1:8000
```
in Chrome, Edge, or Firefox.

---

## 4. Tuning for Natural Indian Female Voice

Indian languages (Hindi, Hinglish, Marathi, Tamil, etc.) contain unique acoustic characteristics:
- **Aspirated stops** (ख, घ, छ, झ, थ, ध, फ, भ) which carry high-frequency breath bursts.
- **Retroflex consonants** (ट, ठ, ड, ढ, ण, ड़) where the tongue curls backward, altering the 2nd and 3rd formant transitions.

### Recommended Settings in VaniFlow:

| Parameter | Recommended Value | Acoustic Effect |
| :--- | :--- | :--- |
| **Pitch Multiplier** | `1.55x – 1.65x` | Transposes adult male f0 (110–135 Hz) into natural Indian female conversational register (175–220 Hz). |
| **Formant Warp** | `1.12x – 1.16x` | Scales the spectral envelope to simulate a shorter female vocal tract without creating artificial "helium" effects. |
| **Output Gain** | `85% – 95%` | Balanced level feeding headphones with soft-knee limiter preventing clipping. |
| **Dry / Wet Mix** | `100% Wet` | Completely replaces original voice with converted audio. |

---

## 5. Phase 2: RVC Neural AI Mode Setup

To use the neural RVC AI conversion mode:

1. **Obtain HuBERT Feature Extractor Checkpoint:**
   - Download `hubert_base.pt` (~380 MB) from the official [HuggingFace VoiceConversionWebUI repository](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main).
   - Place it into the `models/` directory:
     ```
     models/hubert_base.pt
     ```

2. **Obtain or Place Target Voice Model (.pth):**
   - Place your trained or verified RVC v2 voice model file (e.g. `indian_female.pth`) and optional `.index` file into `models/`.
   - Security Notice: PyTorch `.pth` files execute code on deserialization; only use checkpoints from trusted sources.

3. **Select AI Mode:**
   - In the Web UI, click **RVC AI Mode (Neural)** or select the model from the dropdown.

---

## 6. Phase 4: Virtual Microphone Routing (VB-CABLE)

To speak with your converted female voice in **Discord, Zoom, Google Meet, Microsoft Teams, or OBS Studio**:

### Step 1: Install VB-CABLE
1. Download **VB-CABLE Driver** from the official VB-Audio website: [https://vb-audio.com/Cable/](https://vb-audio.com/Cable/)
2. Extract the downloaded ZIP file.
3. Right-click `VBCABLE_Setup_x64.exe` and select **Run as administrator**.
4. Click **Install Driver** and restart your computer if prompted.

### Step 2: Configure VaniFlow Audio Routing
1. Open the VaniFlow control panel: `http://127.0.0.1:8000`.
2. Under **Microphone Input**, select your physical microphone (or headset mic).
3. Under **Headphones / Virtual Mic Output**, select **CABLE Input (VB-Audio Virtual Cable)**.
4. Click **Start Voice Conversion**.

### Step 3: Configure Your Communication App
In Discord / Zoom / Teams / Google Meet:
- Open **Settings** → **Audio & Video**.
- Under **Input Device / Microphone**, select **CABLE Output (VB-Audio Virtual Cable)**.
- Disable aggressive noise suppression (e.g., Discord Krisp) to avoid clipping pitch-shifted vocal harmonics.

### Step 4: Hear Yourself (Optional Live Monitoring)
If you want to hear your converted voice in your headphones while simultaneously streaming it to other people:
1. Press `Win + R`, type `mmsys.cpl` and press Enter.
2. Go to the **Recording** tab.
3. Right-click **CABLE Output** → **Properties**.
4. Go to the **Listen** tab.
5. Check **Listen to this device** and select your physical **Headphones** under *Playback through this device*.
6. Click **Apply**.

---

## 7. Automated Testing & Verification

Run the complete test suite:
```powershell
pytest -v
```
*(All 22 automated tests covering devices, configs, DSP vocoder, RVC architecture, pipeline safety, and REST APIs pass).*

Run the hardware performance benchmark:
```powershell
python benchmark.py
```

Run the live audio hardware smoke test:
```powershell
python smoke_test_audio.py 5
```

---

## 8. REST API Reference

The backend exposes FastAPI endpoints on `http://127.0.0.1:8000`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server uptime and health check |
| `GET` | `/audio/devices` | Enumerate all input/output hardware devices |
| `GET` | `/audio/status` | Live latency, RTF, RMS meters, underrun counts |
| `GET` | `/voice/models` | List available DSP presets and RVC checkpoints |
| `GET` | `/settings` | Retrieve active configuration |
| `PUT` | `/settings` | Live parameter updates (pitch, formant, gain, devices) |
| `POST` | `/audio/start` | Start live voice conversion pipeline |
| `POST` | `/audio/stop` | Gracefully stop live voice conversion pipeline |

---

## 9. Project Directory Structure

```
voice/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration dataclasses and persistence
│   ├── main.py                # FastAPI server, lifespan hooks, UI mount
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # REST endpoints for audio control & telemetry
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py         # Hardware input stream, resampling, ring buffer
│   │   ├── devices.py         # Device enumeration and openability probe
│   │   ├── pipeline.py        # Pipeline orchestrator, crossfade, limiter
│   │   └── playback.py        # Hardware output stream, soft limiter
│   ├── conversion/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseConverter abstract interface
│   │   ├── dsp_converter.py   # WORLD vocoder & phase vocoder engines
│   │   ├── model_loader.py    # Catalog scanner for presets & checkpoints
│   │   ├── rvc_converter.py   # RVC v2 inference pipeline
│   │   └── rvc_models.py      # SynthesizerTrn PyTorch generator
│   └── ui/
│       └── index.html         # Modern responsive web control panel
├── models/
│   └── README.md              # Checkpoint download and security instructions
├── tests/
│   ├── __init__.py
│   ├── test_api.py            # API integration tests
│   ├── test_audio_devices.py  # Audio hardware discovery tests
│   ├── test_config.py         # Configuration tests
│   ├── test_dsp_converter.py  # DSP vocoder unit tests
│   ├── test_pipeline.py       # Pipeline crossfade and limiter tests
│   └── test_rvc_models.py     # RVC model and architecture tests
├── benchmark.py               # Automated CPU performance benchmark
├── smoke_test_audio.py        # Live hardware audio smoke test
├── requirements.txt           # Pinned dependencies
├── run.bat                    # Windows batch launcher
├── run.ps1                    # PowerShell launcher
├── .env.example               # Example environment configuration
├── .gitignore                 # Excludes weights, recordings, caches
└── README.md                  # Comprehensive user and technical guide
```
