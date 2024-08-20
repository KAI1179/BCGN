# BCGN

## Environment (primary)
torch                     1.13.1+cu117

torchaudio                0.13.1

torchsummary              1.5.1

torchvision               0.14.1+cu117

transformers              4.26.1


## Dataset

Please download the VMRD dataset from the [URL](https://www.dropbox.com/s/ff0f4bqw4s1pxa2/VMRD%20V2%20fixed.tar.gz?dl=0) and organize it into the following format: (ImageSets and Instructions have been provided in this repo.)
```
├── LRGD
│   ├── Annotations
│   ├── Grasps
│   ├── ImageSets
│   │   └── Main
│   ├── Instructions
│   ├── JPEGImages
```

## Training
  ```bash
  # Full model
  python train_BCGN.py
  # Ablation of Cross-modal Fusion
  python train_abla_noFusion.py
  ## Ablation of ResNet Encoder
  python train_abla_noResNet.py
  ```
## Test
  ```bash
  # The pre-trained modals (full and ablation) are provided in Google drive(https://drive.google.com/file/d/15ETcqYkG3x1zX-2fRzJcP2ChA-mGesw3/view?usp=drive_link)
  python test.py
  ```

