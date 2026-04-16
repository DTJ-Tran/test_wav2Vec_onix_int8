from transformers import Wav2Vec2Processor
import onnxruntime as ort
from datasets import load_dataset
import numpy as np
import librosa

# ----------------------------
# 1. Load processor + ONNX
# ----------------------------
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-lv-60-espeak-cv-ft"
)

onnx_path = "production_rag_v2/resources/llm_edge/wav2vec2-lv-60-espeak-cv-ft-ONNX/onnx-community_wav2vec2-lv-60-espeak-cv-ft-_int8.onnx"

session = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)

# ----------------------------
# 2. Load audio
# ----------------------------
# ds = load_dataset("patrickvonplaten/audio_samples", split="train")

# audio = ds[1]["audio"]["array"]
# sr = ds[1]["audio"]["sampling_rate"]

audio, sr = librosa.load("arctic_a0001.wav", sr=16000)

# FORCE correct sampling rate
# audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# ----------------------------
# 3. Preprocess
# ----------------------------
inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="np",
    padding=True
)

input_values = inputs["input_values"].astype(np.float32)

# ----------------------------
# 4. Run ONNX
# ----------------------------
outputs = session.run(
    None,
    {"input_values": input_values}
)

logits = outputs[0]

# ----------------------------
# 5. DEBUG CHECKS (IMPORTANT)
# ----------------------------

print("\n===== SHAPES =====")
print("logits shape:", logits.shape)

pred_ids = np.argmax(logits, axis=-1)

print("\n===== PRED SHAPE =====")
print("pred_ids shape:", pred_ids.shape)

print("\n===== BLANK TOKEN CHECK =====")
blank_ratio = np.mean(pred_ids == 0)
print("blank ratio:", blank_ratio)

print("\n===== SAMPLE IDS =====")
print(pred_ids[0][:50])

# ----------------------------
# 6. Decode
# ----------------------------
text = processor.batch_decode(pred_ids)

print("\n===== TRANSCRIPTION =====")
print(text)