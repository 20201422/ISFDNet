# Identity and Style Feature Decoupling Network for Cross-domain Palmprint Recognition

This repository is a PyTorch implementation of ISFDNet (accepted by IEEE Transactions on Information Forensics and Security). This paper can be downloaded at [this link](https://doi.org/10.1109/TIFS.2026.3675481).

## Abstract
Palmprint recognition systems experience a significant performance decline in cross-domain scenarios due to
domain shift caused by non-identity factors such as capture
devices and lighting conditions. To address this issue, this paper
introduces a novel deep decoupling framework, the Identity
and Style Feature Decoupling Network (ISFDNet), designed
to improve the model’s cross-domain generalization. ISFDNet
explicitly separates stable identity-related information from variable domain-related style information within palmprint features.
The framework incorporates two innovative mechanisms: at the
feature level, the Spatially-Aware Separation Module (SASM)
adaptively produces complementary spatial attention masks to
decouple mixed features into identity and style components; at
the image level, the Low-Frequency Disturbance Module (LFDM)
creates stylized training samples by perturbing the low-frequency
parts of images, encouraging the network to learn identity representations that are insensitive to style variations. Additionally,
a carefully designed collaborative supervision strategy combines
multiple losses to ensure effective decoupling. Extensive experiments on four publicly available palmprint datasets demonstrate
that ISFDNet achieves top performance in both cross-domain and
in-domain tests, while significantly enhancing the generalization
capabilities of existing networks.

## Citation
If our work is valuable to you, please cite our work:
```
@ARTICLE{
  author={Liu, Yunlong and Leng, Lu and Chu, Jun and Teoh, Andrew Beng Jin and Zhang, Bob and Yang, Ziyuan},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Identity and Style Feature Decoupling Network for Cross-domain Palmprint Recognition}, 
  year={2026},
  volume={21},
  pages={3066-3079},
  doi={10.1109/TIFS.2026.3675481}}
```

## Requirements

Our codes were implemented by ```PyTorch 2.4.1``` and ```12.1``` CUDA version. If you wanna try our method, please first install necessary packages as follows:

```
pip install requirements.txt
```

## Data Preprocessing
To help readers to reproduce our method, we also release our training and testing lists (including PolyU, Tongji, IITD, Multi-Spectrum datasets). If you wanna try our method in other datasets, you need to generate training and testing texts as follows:

#### Be careful to modify the data set path in the file!

```
python ./data/PolyU/get_data_text_for_PolyU.py
```

## Quick Start for Training and Testing
If you wanna try our method quickly, you can directly run our code as follows:

#### Check the order in which the data sets are trained in the ```auto_run.py``` file before you begin!

```
python auto_run.py
```

## Training and Testing
If you need to customize the parameters, look like this:

```
python main.py
```

## Acknowledgments
Thanks to my all cooperators, they contributed so much to this work.

## Contact
If you have any question or suggestion to our work, please feel free to contact me. My email is 24.yunongliu@gmail.com.

