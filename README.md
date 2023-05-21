# CVPR2023Summary
CVPR2023所有论文批量下载和利用ChatPaper批量总结，如果对大家有帮助，欢迎点个star。
![QQ图片20230402203458](https://github.com/kaixindelele/CVPR2023Summary/assets/28528386/2df91c3e-955c-476f-9e38-96b629982835)


## CVPR2023所有PDF打包下载地址：

官网地址：https://openaccess.thecvf.com/CVPR2023?day=all

如果是国外的同学，可以直接运行python脚本，速度也起飞。

```python
python Get_All_CVPR.py
```

国内同学可以等我的阿里云盘打包。


CVPR2023所有PDF打包：

https://www.aliyundrive.com/s/QCmJNbUQRPb

提取码: r96v

点击链接保存，或者复制本段内容，打开「阿里云盘」APP.

有两篇文章有破损。Rivas-Manzaneque_NeRFLight_Fast_and_Light_Neural_Radiance_Fields_Using_a_Shared_CVPR_2023_paper.pdf这篇文章官网给的链接就是坏的。

## ChatPaper的总结太慢了，我们持续更新：

目前更新前900篇的总结，欢迎大家根据关键词做整理。这900篇的总结大概需要50刀的额度。

后面我们将同步到我们的官网上chatpaper.org中，可以根据chat的对话形式来获得自己感兴趣的论文总结。

我先放几个demo：
## 解析示例

<details><summary><code>查看第一篇解析结果</code></summary>

# Paper:1     多重退出：加速统一视觉语言模型的动态早期退出



#### 1. Title: 
You Need Multiple Exiting: Dynamic Early Exiting for Accelerating Unified Vision Language Model

#### 2. Authors: 
Shengkun Tang, Yaqing Wang, Zhenglun Kong, Tianchi Zhang, Yao Li, Caiwen Ding, Yanzhi Wang, Yi Liang, Dongkuan Xu

#### 3. Affiliation: 
第一作者：Shengkun Tang，北卡罗来纳州立大学

#### 4. Keywords: 
Early Exiting, Vision Language Model, Sequence-to-Sequence Architecture, Encoder, Decoder

#### 5. Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Tang_You_Need_Multiple_Exiting_Dynamic_Early_Exiting_for_Accelerating_Unified_CVPR_2021_paper.html  Github: None

#### 6. Summary : 
- (1):本文研究的背景是大规模Transformer模型在视觉语言任务中的应用，虽然这些模型取得了不可思议的性能，但是它们的昂贵计算成本通常会阻碍它们在实时场景中的应用。

- (2):现有的早期退出策略通常采用中间层的输出置信度作为输入复杂度的代理，以决定是否跳过后续层。然而，这种策略无法应用于同时具有编码器和解码器的统一架构中的编码器，因为难以在编码器层中进行输出置信度估计。为了解决这个问题，本文提出了一种新的早期退出策略，允许根据输入层之间的相似性动态跳过编码器和解码器中的层，即MuE。通过对编码器中的图像和文本模态进行分解，MuE具有灵活性，可以根据模态跳过不同的层，提高推理效率，同时最小化性能下降。

- (3):本文提出了一种基于层间输入相似性的早期退出策略，该策略不同于现有的基于任务置信度的方法。具体来说，当层间相似性达到一定阈值时，模型被鼓励在编码器和解码器中跳过后续层。此方法受到饱和观察的启发，该观察表明，每个Transformer层的隐藏状态在进入深层时会达到饱和状态。为了在需要显著降低推理成本时帮助维持性能，我们设计了一种层间任务损失，将每个层与最终任务相关联。

- (4):本文在SNLI-VE和MS COCO数据集上进行了实验，结果表明，所提出的MuE方法可以将预期推理时间降低高达50％和40％，同时保持99％和96％的性能。
#### 7. 方法详细介绍：
本文提出了一种名为MuE的新型早期退出策略，用于统一的视觉语言模型。MuE允许根据多次早期退出的层内输入相似性动态跳过编码器和解码器组件中的层。该方法受到饱和观察的启发，该观察表明每个Transformer层的隐藏状态在进入深层时到达饱和状态。为了鼓励最小化性能损失的早期退出行为，设计了一种层内任务损失，该损失强制每个层输出最终任务的信息特征。具体步骤包括：
1. 将早期融合编码器分解为处理图像和文本的模态特定编码器。
2. 复制编码器以处理输入，其中图像标记和文本标记分别输入两个编码器。
3. 引入层内任务损失以在微调期间鼓励早期退出行为。
4. 基于余弦相似度作为估计饱和水平的代理来做出早期退出决策。

#### 8. 实验设置：
本文在SNLI-VE和MS COCO数据集上进行了实验，以评估所提出方法的性能。实验在一台服务器上进行，该服务器配备了Intel Xeon E5-2690 CPU和NVIDIA Tesla V100 GPU。实现基于PyTorch，使用Adam优化器进行训练，学习率为1e-4。

#### 9. 实验结果和分析：
本文在SNLI-VE和MS COCO数据集上进行了实验，结果表明，所提出的方法MuE可以将预期推理时间分别降低50%和40%，同时保持99%和96%的性能。MuE在预期时间减少率和任务性能方面优于几种最先进的早期退出方法，包括PABEE、DeeCap和DeeBERT。作者还进行了消融实验，结果表明，没有分解策略和训练目标的模型性能最差。在图像字幕生成中，缺少所提出的层内任务损失会导致性能和预期时间减少率的大幅下降。所提出的层内任务损失能够在解码的每个时间步骤上减少错误，这对最终结果有益。所有实验结果都是在视觉蕴含和图像字幕生成中得分和预期时间减少率之间的最佳平衡。
</details>

<details><summary><code>查看第二篇解析结果</code></summary>

# Paper:2     探测开放世界中的一切：通用目标检测



#### 1. Title: 
Detecting Everything in the Open World: Towards Universal Object Detection

#### 2. Authors: 
Zhenyu Wang, Yali Li, Xi Chen, Ser-Nam Lim, Antonio Torralba, Hengshuang Zhao, Shengjin Wang

#### 3. Affiliation: 
第一作者：清华大学电子工程系

#### 4. Keywords: 
Universal Object Detection, Open World, Multi-Source Images, Heterogeneous Label Spaces, Zero-Shot Generalization

#### 5. Paper: https://openaccess.thecvf.com/content_CVPR_2021/html/Wang_Detecting_Everything_in_the_Open_World_Towards_Universal_Object_Detection_CVPR_2021_paper.html  Github: https://github.com/zhenyuw16/UniDetector

#### 6. Summary : 
- (1):本文研究了通用目标检测，旨在检测每个场景并预测每个类别。传统检测器的通用性受到人类注释的依赖、有限的视觉信息和开放世界中的新类别的严重限制。本文提出了UniDetector，一种通用目标检测器，具有识别开放世界中巨大类别的能力。
 
- (2):传统目标检测只能检测训练时出现的类别。在通用目标检测中，需要检测的类别事先无法确定。本文提出的UniDetector通过对齐图像和文本空间，利用多源图像和异构标签空间进行训练，从而保证了通用表示的充分信息。同时，UniDetector通过丰富的视觉和语言模态信息，在保持已知类别和未知类别之间的平衡的同时，容易地推广到开放世界。此外，UniDetector通过提出的解耦训练方式和概率校准，进一步促进了对新类别的泛化能力。 

- (3):本文提出了UniDetector，一种通用目标检测框架，用于解决多源图像训练和开放世界推理的问题。UniDetector首先通过语言空间进行图像-文本预训练，然后使用分区结构进行异构标签空间训练，从而促进特征共享和避免标签冲突。为了利用区域提议阶段对新类别的泛化能力，本文提出了解耦合的提议生成和RoI分类阶段的训练方式。在解耦合的方式下，本文进一步提出了一个类别无关的定位网络（CLN）来产生广义的区域提议。最后，本文提出了概率校准来消除预测的偏差。 

- (4):UniDetector在大量实验中展现了其强大的通用性。它可以识别最大可测量的类别，并在不看到任何训练集中的图像的情况下，在现有大词汇数据集上比完全监督方法高出4%的AP。此外，UniDetector在13个公共检测数据集上也取得了最先进的性能，只使用了3%的训练数据。
#### 7. 方法详细介绍：
本文提出了UniDetector框架，用于解决通用目标检测任务。该框架利用多源图像和异构标签空间进行训练，通过图像和文本空间的对齐来实现。UniDetector采用分区结构来促进特征共享，并同时避免标签冲突。提议生成阶段和RoI分类阶段被解耦以充分探索类别敏感特征。本文提出了一个类不可知的本地化网络（CLN），用于生成广义区域提议。概率校准被提出用于后处理预测结果以减少基础类别的概率并增加新颖类别的概率，从而平衡最终的概率预测。具体步骤包括：
1. 对齐图像和文本空间，进行大规模的图像-文本对齐预训练。
2. 采用分区结构，同时避免标签冲突和促进特征共享。
3. 采用类不可知的本地化网络（CLN）生成广义区域提议。
4. 采用概率校准进行后处理，平衡最终的概率预测。

#### 8. 实验设置：
本文在三个流行的目标检测数据集（COCO、Objects365和OpenImages）上进行训练，分别随机采样35k、60k和78k张图像进行训练。主要在LVIS、ImageNetBoxes和VisualGenome数据集上进行推理，以评估检测器的开放世界性能。本文使用标准的box AP、top-1定位精度和平均召回率指标来评估性能。

#### 9. 实验结果和分析：
本文在多个数据集上评估了UniDetector的性能。在COCO数据集上，UniDetector的检测AP为49.3%，超过了现有的最佳封闭世界检测模型。在开放世界数据集上，UniDetector在13个ODinW数据集上的平均AP为47.3%，优于GLIP-T，具有更高的数据效率。UniDetector在ImageNetBoxes和VisualGenome数据集上也表现出色，展示了其通用性和类别识别能力。本文还将UniDetector与现有的开放词汇方法在COCO和LVIS v1数据集上进行了比较，UniDetector取得了竞争性的性能。
</details>

<details><summary><code>查看第三篇解析结果</code></summary>

# Paper:3     WIRE：小波隐式神经表示



#### 1. Title: 
WIRE: Wavelet Implicit Neural Representations

#### 2. Authors: 
Vishwanath Saragadam, Daniel LeJeune, Jasper Tan, Guha Balakrishnan, Ashok Veeraraghavan, Richard G. Baraniuk

#### 3. Affiliation: 
Rice University（莱斯大学）

#### 4. Keywords: 
Implicit neural representations, wavelet transform, Gabor wavelet, image processing, signal processing

#### 5. Paper: https://vishwa91.github.io/wire  Github: https://github.com/vishwa91/wire

#### 6. Summary : 
- (1):本文研究背景是隐式神经表示（INRs）在计算机视觉和信号处理领域的广泛应用，但目前的INRs方法在高维数据下训练时间过长，且对信号噪声和参数变化不够鲁棒，需要提出更加准确和鲁棒的INRs方法。

- (2):过去的方法包括使用ReLU非线性函数的INRs，但其在近似精度上表现不佳，需要进行改进。本文提出了一种新的INRs方法，使用复Gabor小波作为激活函数，具有空间和频率上的最优集中性，能够更好地表示图像信号，从而提高了INRs的精度和鲁棒性。

- (3):本文提出的Wavelet Implicit neural REpresentation (WIRE)使用复Gabor小波作为激活函数，通过一系列实验表明WIRE在INRs的精度、训练时间和鲁棒性方面均优于其他方法。WIRE的鲁棒性特别适用于解决图像去噪、图像修复和超分辨率等困难的视觉反问题。此外，WIRE还在信号表示任务中表现出色，如过度拟合图像和学习点云占用体积。最后，本文还展示了WIRE如何从极少的训练视图中实现更快、更鲁棒的神经辐射场（NeRF）的新视图合成。

- (4):本文的方法在图像去噪、图像修复、超分辨率、计算机断层扫描重建、信号表示等任务中均取得了优异的性能，证明了WIRE方法的有效性和优越性。
#### 7. 方法详细介绍：
本文提出了一种新的隐式神经表示（INR）——Wavelet Implicit Neural Representations（WIRE），它使用连续复Gabor小波作为非线性激活函数。WIRE的结构包括三个隐藏层，每个隐藏层的宽度为300个特征。WIRE的输入维度为Di，输出维度为Do，函数Fθ将输入映射到输出，其中θ表示MLP的可调参数。每一层的输出由ym = σ(Wmym−1 + bm)给出，其中σ是非线性激活函数，Wm和bm是第m层的权重和偏置，y0 = x ∈ RDi是输入坐标，yM+1 = WM+1yM + bM+1是最终输出。本文还讨论了WIRE的隐式偏差，并使用经验神经切向核（NTK）和NTK梯度流将其与其他INR进行了比较。

#### 8. 实验设置：
本文使用MLP对图像和占用体积进行了评估，其中每个非线性激活函数的参数和学习率都是根据最快逼近速率选择的。具体来说，WIRE的参数为ω0 = 20，s0 = 10，SIREN的参数为ω0 = 40，Gaussian的参数为s0 = 30。本文还将WIRE与乘法频率网络（MFN）进行了比较。评估指标为图像的PSNR和结构相似性（SSIM），占用体积的交并比（IOU）。

#### 9. 实验结果与分析：
本文的实验结果表明，WIRE在所有信号类别的表示学习中都比现有技术更快更准确。WIRE还适用于解决具有有限测量或测量受到噪声干扰的大类逆问题。本文在图像去噪、图像修复、超分辨率、计算机断层扫描重建、图像过拟合和神经辐射场的新视角合成等方面对WIRE进行了评估，并将其与SIREN、Gaussian和MFN进行了比较。实验结果表明，WIRE在准确性和收敛速度方面均优于其他非线性激活函数，且对于图像或噪声统计的精确信息要求较低。

</details>


