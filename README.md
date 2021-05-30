# MSLPN
## Few-Shot Image Classification Based on Multi-Scale Label Propagation
  Under the condition of few-shot, due to the problem of low data, in other words, the labeled data is rare and difficult to gather, it is very difficult to train a good classifier by traditional deep learning. In recent researches, the method based on measuring low level local information and Transductive Propagation Network (TPN) has achieved good classification results. Moreover, local information can measure the relation between features well, but the problem of low data still exists. In order to solve the issue of low data, a Multi-Scale Label Propagation Network (MSLPN) based on TPN is proposed in this paper. Firstly, generate multiple image feature information in different scales by using multi-scale feature generator. And then, the multi-scale feature information is used for label propagation. Finally, classification results will be obtained by calculating the multi-scale label propagation results. Compared with TPN, in miniImageNet, the classification accuracy of 5-way 1-shot and 5-way 5-shot settings are increased by 2.77% and 4.02% respectively. While in tieredImageNet, the classification accuracy of 5-way 1-shot and 5-way 5-shot settings are increased by 1.16% and 1.27% respectively. The experimental results show that the proposed method in this paper can effectively improve the classification accuracy by using multi-scale feature information.
## Accuracy
### miniImageNet
方法 | 类型 | 5-way 1-shot | 5-way 5-shot
:-----:|:-----:|:-----:|:----------:|
Matching Network | 度量学习 | 43.56±0.84 | 55.31±0.73
Prototypical Network | 度量学习 | 49.42±0.78 | 68.20±0.66
Relation Network | 度量学习 | 50.44±0.82 | 65.32±0.70
DN4 | 度量学习 | 51.24±0.74 | 71.02±0.64
RCN | 度量学习 | 53.47±0.84 | 71.63±0.70
MSDN | 度量学习 | 52.59±0.81 | 68.51±0.69
MATANet | 度量学习 | 53.63±0.83 | 72.67±0.76
Looking-Back | 度量学习 | 55.91±0.86 | 70.99±0.68
TPN | 度量学习 | 53.75±0.86 | 69.43±0.67
MAML | 元学习 | 48.70±1.75 | 63.11±0.92
SNAL | 元学习 | 55.71±0.99 | 68.88±0.92
Meta-Learner LSTM | 元学习 | 43.44±0.77 | 60.60±0.71
MSLPN（本文） | 度量学习 | **56.52±0.92** | **73.45±0.86**
### tieredImageNet
方法 | 类型 | 5-way 1-shot | 5-way 5-shot
:-----:|:-----:|:-----:|:----------:|
Prototypical Network | 度量学习 | 53.31±0.89 | 72.69±0.74
Relation Network | 度量学习 | 54.48±0.93 | 71.32±0.78
Looking-Back | 度量学习 | **58.97±0.97** | 73.59±0.74
TPN | 度量学习 | 57.53±0.96 | 72.85±0.74
MAML | 元学习 | 51.67±1.81 | 70.30±1.75
MSLPN（本文） | 度量学习 | 58.69±0.96 | **74.12±0.73**
