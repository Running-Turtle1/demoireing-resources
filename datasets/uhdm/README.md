# UHDM

Expected layout:

```text
UHDM/
├── train/**/*_{moire,gt}.jpg
└── test/**/*_{moire,gt}.jpg
```

Training uses aligned random 768×768 crops without resizing. Evaluation uses
the complete test images, either with 768-pixel overlap-add tiles or one padded
rectangular forward pass. The dataset is not redistributed.
