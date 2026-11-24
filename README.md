# Deep-Fake Detection

This project contains a baseline PyTorch classifier for deciding whether an image is probably real or AI-generated.

## Recommended dataset download

For this image-level project, start with the official [GenImage repository](https://github.com/GenImage-Dataset/GenImage). Its README links to the [Google Drive download](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS?usp=sharing). Do not download the entire benchmark initially; use one generator subset such as Stable Diffusion V1.4 for a first training run.

GenImage uses the labels `ai` and `nature`. Copy or rename them into this project's labels:

```text
GenImage/.../train/ai     -> data/train/fake
GenImage/.../train/nature -> data/train/real
GenImage/.../val/ai       -> data/val/fake
GenImage/.../val/nature   -> data/val/real
```

For face-specific deepfakes, request [FaceForensics++](https://github.com/ondyari/FaceForensics) through its official access form. It is distributed under dataset terms of use and is not an instant public download.

## Dataset layout

Put images into these folders:

```text
data/
├── train/
│   ├── fake/
│   └── real/
├── val/
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
```

Keep the classes balanced. Do not place images from the same original source in both training and validation. For a meaningful research evaluation, hold out at least one generator or manipulation type for testing.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python train.py --epochs 10
```

The best checkpoint is saved as `best_model.pt` according to validation ROC-AUC.

## Download and prepare CIFAKE automatically

```powershell
python prepare_cifake.py
```

For a quick smoke test, use `python prepare_cifake.py --limit-per-class 2000`. The full export creates the train, validation, and test folders and may take several minutes.

## Predict an image

```powershell
python predict.py path\to\image.jpg
```

## Launch the upload application

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit and upload a JPG, PNG, or WEBP image.

## Linux/WSL environment used for the demo

The trained checkpoint in this folder was created in Ubuntu WSL with an NVIDIA RTX 3050 GPU. The full CIFAKE export is in `data/`; the runnable demo checkpoint was trained on the balanced `data_quick/` subset containing 20,000 training images, 4,000 validation images, and 8,000 test images.

To reproduce the Linux run:

```bash
python3 -m venv /home/$USER/deepfake-venv
/home/$USER/deepfake-venv/bin/pip install -r requirements.txt
/home/$USER/deepfake-venv/bin/python train.py --data-dir data_quick --epochs 2 --batch-size 128 --image-size 128 --workers 4
```

The measured test result for the saved checkpoint was 95.56% accuracy and 0.9926 ROC-AUC. These numbers describe CIFAKE-style images and should not be presented as guaranteed accuracy on arbitrary internet images.

The predictor uses three outcomes: `AI-generated`, `Probably real`, and `Uncertain`. The last category is intentional because an image detector is not a proof of authenticity.

The app also performs a local provenance scan for surviving C2PA/Content Credentials, EXIF/XMP text, and provider terms associated with OpenAI/ChatGPT/DALL-E or Google/Gemini/Imagen. These are hints only: metadata can be removed, and no marker does not mean an image is real. OpenAI images may carry C2PA and SynthID signals, while Google describes SynthID as an invisible watermark for supported generated content.

Because the initial model was trained on 32×32 CIFAKE images, the app now warns when a high-resolution upload is outside that training domain. It also flags conflicts when camera metadata is present but the visual model predicts AI. This prevents an overconfident model score from being presented as proof.

## Recommended research datasets

- GenImage: general AI-generated image detection.
- FaceForensics++: manipulated facial images and videos.
- CNN-generated images paper: a useful research baseline for cross-generator generalization.

