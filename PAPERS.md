# Research papers and implementation path

## 1. CNN-generated images are surprisingly easy to spot... for now

- Paper: https://arxiv.org/abs/1912.11035
- Official project: https://peterwang512.github.io/CNNDetection/
- Why it matters: establishes a strong CNN baseline and studies cross-generator generalization.
- Implementation step: compare this project's ResNet baseline against the paper's preprocessing and augmentation choices.

## 2. FaceForensics++: Learning to Detect Manipulated Facial Images

- Paper: https://arxiv.org/abs/1901.08971
- Official repository: https://github.com/ondyari/FaceForensics
- Why it matters: standard benchmark for face manipulation methods such as DeepFakes, Face2Face, FaceSwap, and NeuralTextures.
- Implementation step: use extracted face crops and test whether the classifier generalizes across manipulation types.

## 3. GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image

- Paper: https://arxiv.org/abs/2306.08571
- Official repository: https://github.com/GenImage-Dataset/GenImage
- Why it matters: evaluates AI-generated image detection across multiple generators and degraded images.
- Implementation step: train on selected generators, then hold out another generator for cross-generator testing.

## Reproducibility checklist

- Record the dataset version, generator names, image count, and split seed.
- Keep near-duplicates and images from the same source in one split only.
- Report ROC-AUC, precision, recall, F1-score, and confusion matrix.
- Evaluate separately on untouched images, resized images, blurred images, and JPEG-compressed images.
