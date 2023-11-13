# AGGDN
Adversarial Graph-Gated Differential Network

## Introduction: 
This is an implementation of the paper [
AGGDN: A Continuous Stochastic Predictive Model for Monitoring Sporadic Time Series on Graphs](https://link.springer.com/chapter/10.1007/978-981-99-8079-6_11#citeas) accepted by ICONIP 2023.

## Citation:
If you use this code, please cite:
```
@InProceedings{10.1007/978-981-99-8079-6_11,
    author="Xing, Yucheng
            and Wu, Jacqueline
            and Liu, Yingru
            and Yang, Xuewen
            and Wang, Xin",
            editor="Luo, Biao
            and Cheng, Long
            and Wu, Zheng-Guang
            and Li, Hongyi
            and Li, Chaojie",
    title="AGGDN: A Continuous Stochastic Predictive Model for Monitoring Sporadic Time Series on Graphs",
    booktitle="Neural Information Processing",
    year="2024",
    publisher="Springer Nature Singapore",
    address="Singapore",
    pages="130--146",
    abstract="Monitoring data of real-world networked systems could be sparse and irregular due to node failures or packet loss, which makes it a challenge to model the continuous dynamics of system states. Representing a network as graph, we propose a deep learning model, Adversarial Graph-Gated Differential Network (AGGDN). To accurately capture the spatial-temporal interactions and extract hidden features from data, AGGDN introduces a novel module, dynDC-ODE, which empowers Ordinary Differential Equation (ODE) with learning-based Diffusion Convolution (DC) to effectively infer relations among nodes and parameterize continuous-time system dynamics over graph. It further incorporates a Stochastic Differential Equation (SDE) module and applies it over graph to efficiently capture the underlying uncertainty of the networked systems. Different from any single differential equation model, the ODE part also works as a control signal to modulate the SDE propagation. With the recurrent running of the two modules, AGGDN can serve as an accurate online predictive model that is effective for either monitoring or analyzing the real-world networked objects. In addition, we introduce a soft masking scheme to capture the effects of partial observations caused by the random missing of data from nodes. As training a model with SDE component could be challenging, Wasserstein adversarial training is exploited to fit the complicated distribution. Extensive results demonstrate that AGGDN significantly outperforms existing methods for online prediction.",
    isbn="978-981-99-8079-6"
}
```
