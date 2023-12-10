## EVCap: Retrieval-Augmented Image Captioning with External Visual-Name Memory for Open-World Comprehension
<a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Partition-And-Debias_Agnostic_Biases_Mitigation_via_a_Mixture_of_Biases-Specific_Experts_ICCV_2023_paper.pdf"><img src="https://img.shields.io/static/v1?label=Paper&message=PnD&color=red" height=20.5></a> 
<a href="https://arxiv.org/abs/2311.15879"><img src="https://img.shields.io/badge/arXiv-2308.10005-b31b1b.svg" height=20.5></a>
<a href="https://jiaxuan-li.github.io/EVCap/"><img src="https://img.shields.io/badge/WEB-Page-159957.svg" height=20.5></a>

[**Jiaxuan Li**](https://jiaxuan-li.github.io/)<sup>1</sup>, [**Minh-Duc Vo**](https://vmdlab.github.io/)<sup>1</sup>, [**Akihiro Sugimoto**](http://research.nii.ac.jp/~sugimoto/index.html)<sup>2</sup>, [**Hideki Nakayama**](http://www.nlab.ci.i.u-tokyo.ac.jp/index-e.html)<sup>1</sup>

<sup>1</sup>The University of Tokyo, <sup>2</sup>National Institute of Informatics

> Large language models (LLMs)-based image captioning has the capability of describing objects not explicitly observed in training data; yet novel objects occur frequently, necessitating the requirement of sustaining up-to-date object knowledge for open-world comprehension. Instead of relying on large amounts of data and scaling up network parameters, we introduce a highly effective retrieval-augmented image captioning method that prompts LLMs with object names retrieved from External Visual-name memory (EVCAP). We build ever-changing object knowledge memory using objects’ visuals and names, enabling us to (i) update the memory at a minimal cost and (ii) effortlessly augment LLMs with retrieved object names utilizing a lightweight and fast-to-train model. Our model, which was trained only on the COCO dataset, can be adapted to out-domain data without additional fine-tuning or retraining. Our comprehensive experiments conducted on various benchmarks and synthetic commonsense-violating data demonstrate that EVCAP, comprising solely 3.97M trainable parameters, exhibits superior performance compared to other methods of equivalent model size scale. Notably, it achieves competitive performance against specialist SOTAs with an enormous number of parameters.

<div align=center><img width="80%" src="./static/images/model.png"/></div>


## Setup
Install the required packages using conda with the provided [environment.yaml](environment.yaml) file.

## Training
coming soon~

## Evaluation
coming soon~

## Acknowledgements
This repo is built on [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4), we thank the authors for their great effort.

## Citation
If you find our work helpful for your research, please kindly consider citing:

    @article{li2023evcap,
      title={EVCap: Retrieval-Augmented Image Captioning with External Visual-Name Memory for Open-World Comprehension}, 
      author={Jiaxuan Li and Duc Minh Vo and Akihiro Sugimoto and Hideki Nakayama},
      journal={arXiv preprint arXiv:2311.15879},
      year={2023},
    }

