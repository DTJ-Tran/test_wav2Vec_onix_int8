
# wav2vec2-lv-60-espeak-cv-ft (ONNX)


This is an ONNX version of [facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft). It was automatically converted and uploaded using [this Hugging Face Space](https://huggingface.co/spaces/onnx-community/convert-to-onnx).

In this version I'm using the int-8 quantization (you can download it in [here](https://huggingface.co/onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX/tree/main/onnx)


# Usage

To transcribe audio files the model can be used as a standalone acoustic model as follows: (Check the code in test_ASR.py) - on your local machine with CPU

What you need to install: 

torchcodec==0.11.1, 

onnx==1.21.0

optimum-onnx==0.1.0

onnxruntime==1.24.4

transformers==4.57.6

espeak-ng (use brew / apt is ok - [brew](https://formulae.brew.sh/formula/espeak-ng)

