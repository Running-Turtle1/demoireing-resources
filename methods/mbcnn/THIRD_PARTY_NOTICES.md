# Third-party notices

## UniDemoire PyTorch port

The PyTorch MBCNN model and loss in this repository were extracted and adapted
from [4DVLab/UniDemoire](https://github.com/4DVLab/UniDemoire) at commit
`4221a0e98f0078c3f72ef1da8128642e7f02662a`. UniDemoire is distributed under
the MIT License reproduced in [`LICENSE`](LICENSE).

The extraction removes UniDemoire's Lightning wrapper, moire synthesis and
blending pipeline, datasets, and other restoration networks. Device handling,
imports, packaging, and tests were adapted for standalone use.

## MBCNN architecture

MBCNN was introduced in *Image Demoireing with Learnable Bandpass Filters*
(CVPR 2020). The authors' reference implementation is available at
[zhenngbolun/Learnbale_Bandpass_Filter](https://github.com/zhenngbolun/Learnbale_Bandpass_Filter)
and uses Keras/TensorFlow. This repository is not an official release by the
paper authors.

