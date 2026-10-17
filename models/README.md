# Models Directory
# ================
#
# Place voice conversion model files here.
#
# ## DSP Mode
# No model files needed — DSP mode uses algorithmic processing.
#
# ## RVC AI Mode (Phase 2)
# You will need:
#   1. hubert_base.pt  — HuBERT feature extractor (~380 MB)
#      Source: https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main
#      License: MIT (Facebook Research)
#
#   2. A .pth voice model file (e.g., female_voice.pth)
#      Source: Train your own or find community models on Hugging Face
#      ⚠️ Verify licensing before using any third-party voice model
#
#   3. (Optional) A .index file for retrieval-based matching
#
# ## Security Note
# Never run model files from untrusted sources. PyTorch .pth files
# can contain arbitrary Python code that executes on load.
# Only use models from sources you trust.
