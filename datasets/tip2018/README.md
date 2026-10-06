# TIP2018

Expected layout:

```text
TIP2018/
├── trainData/{source,target}/
└── testData/{source,target}/
```

Files are paired as `*_source.png` and `*_target.png`. The Uformer paper uses a
central `[0.15, 0.85]` crop followed by bilinear resize to 256×256. The released
checkpoint in this repository uses the explicitly labeled UHDM-compatible
variant: central `1/6..5/6`, paired ±6-pixel training offsets, then bilinear
resize to 256×256. These protocols must not be compared as identical.

The dataset is not redistributed by this repository.
