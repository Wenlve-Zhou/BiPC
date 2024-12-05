## BiPC：Bidirectional Probability Calibration for Domain Adaption

**by Wenlve Zhou, Zhiheng Zhou, Junyuan Shang, Chang Niu, Mingyue Zhang, Xiyuan Tao, Tianlei Wang**

**[[ESWA 2024 Paper]](https://www.sciencedirect.com/science/article/pii/S0957417424023273)**
**[[pdf]](https://arxiv.org/abs/2409.19542)**


## Overview

Unsupervised Domain Adaptation (UDA) leverages a labeled source domain to solve tasks in an unlabeled target domain. 
While Transformer-based methods have shown promise in UDA, their application is limited to plain Transformers, excluding 
Convolutional Neural Networks (CNNs) and hierarchical Transformers. To address this issues, we propose Bidirectional Probability 
Calibration (BiPC) from a probability space perspective. We demonstrate that the probability outputs from a pre-trained head, 
after extensive pre-training, are robust against domain gaps and can adjust the probability distribution of the task head. Moreover, 
the task head can enhance the pre-trained head during adaptation training, improving model performance through bidirectional complementation. 
Technically, we introduce Calibrated Probability Alignment (CPA) to adjust the pre-trained head's probabilities, 
such as those from an ImageNet-1k pre-trained classifier. Additionally, we design a Calibrated Gini Impurity (CGI) loss to refine the task head,
with calibrated coefficients learned from the pre-trained classifier. BiPC is a simple yet effective method applicable to various networks, 
including CNNs and Transformers. 
Experimental results demonstrate its remarkable performance across multiple UDA tasks. 
![UDA over time](resources/Methods.jpg)

## Training

You can employ various networks to train on different datasets according to your objectives, for example: 

#ResNet-50-BiPC
```shell
python main.py --config configs/office_home.yaml --data_dir ["root of dataset"] --src_domain ["source domain"] --tgt_domain ["target domain"] --model_name resnet50
```

#ResNet-50-Baseline
```shell
python main.py --config configs/office_home.yaml --data_dir ["root of dataset"] --src_domain ["source domain"] --tgt_domain ["target domain"] --model_name resnet50 --transfer_loss 0.0
```

#deit_base-BiPC
```shell
python main.py --config configs/office_home.yaml --data_dir ["root of dataset"] --src_domain ["source domain"] --tgt_domain ["target domain"] --model_name deit_base
```

#ResNet-deit_base-Baseline
```shell
python main.py --config configs/office_home.yaml --data_dir ["root of dataset"] --src_domain ["source domain"] --tgt_domain ["target domain"] --model_name deit_base --transfer_loss 0.0
```
