# FHDMi

Expected layout:

```text
FHDMi/{train,test}/
├── source/src_XXXXX.png
└── target/tar_XXXXX.png
```

Training uses aligned random 512×512 crops without resizing or geometric
augmentation. Evaluation restores complete images with overlap-add tiles. The
dataset is not redistributed.
