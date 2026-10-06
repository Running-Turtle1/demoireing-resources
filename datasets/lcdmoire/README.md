# LCDMoire

Expected paired layout:

```text
LCDMoire/
├── train/{moire,clear}/*.jpg
└── val/{moire,clear}/*.png
```

Training uses aligned random 512×512 crops. The paired validation set is used
for evaluation because the challenge test split has no public targets. The
canonical evaluation performs one full 1024×1024 forward pass. The dataset is
not redistributed.
