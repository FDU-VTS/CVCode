## Selective Search for Object Recognition

### 原文链接
 - http://www.icst.pku.edu.cn/F/course/ImageProcessing/2017/resource/IJCV2013_Selective%20Search%20for%20Object%20Recognition.pdf

### 概要
 - 本文主要介绍了`Selective Search`算法，该算法被广泛应用于物体检测算法中。

### 算法流程解读
 - 整体流程

        Algorithm 1: Hierarchical Grouping algorithm
          Input: (color)image
          Output: Set of object location hypotheses
          Obtain initial tegions R = {r1,...,rn} using[1]
          Initialise similarity set S = 0
          foreach Neighbouring region pair(ri, rj) do
            Calculate similarity s(ri,rj)
            S = S∪s(ri,rj)
          while S != ∅ do
            Get highest similarity s(ri,rj) = max(S)
            Merge corresponding regions rt = ri ∪ rj
            Remove similarities regarding ri:S = S\s(ri,r*)
            Remove similarities regarding rj:S = S\s(r*,rj)
            Calculate similarity set St between rt and its neighbours
            S = S ∪ St
            R = R ∪ rt
          Extract object location boxes L from all regions in R
 - 首先`input:`一张`W*H*3`的图片
 - `output:`一组由边界组成的集合
 - 获取初始化图像分割使用`Efficient Graph-Based Image Segmentation`
 - S表示的是所有区域之间的相似度，不断合并其中相似度最高的区域

### 相似度计算
 - colour(ri,rj)
   - 相似度计算公式: <a href="http://www.codecogs.com/eqnedit.php?latex=s_colour(r_i,r_j)&space;=&space;\sum^{n}_{k=1}\min(c_i^k,c_j^k)" target="_blank"><img src="http://latex.codecogs.com/svg.latex?s_colour(r_i,r_j)&space;=&space;\sum^{n}_{k=1}\min(c_i^k,c_j^k)" title="s_colour(r_i,r_j) = \sum^{n}_{k=1}\min(c_i^k,c_j^k)" /></a>
   - 解释: C1和C2之间的相似度即为，计算C1和C2的HOG，计算两个HOG的重叠面积即为相似度，计算之前先用L1_norm进行正则化
   - 合并公式: <a href="http://www.codecogs.com/eqnedit.php?latex=C_t&space;=&space;\frac{size(r_i)*C_i&plus;size(r_j)*C_j}{size(r_i)&plus;size(r_j)}" target="_blank"><img src="http://latex.codecogs.com/svg.latex?C_t&space;=&space;\frac{size(r_i)*C_i&plus;size(r_j)*C_j}{size(r_i)&plus;size(r_j)}" title="C_t = \frac{size(r_i)*C_i+size(r_j)*C_j}{size(r_i)+size(r_j)}" /></a>
   - 解释: 新生成的区域为对原有两个区域进行归一化
 - texture(ri,rj)
   - 使用`SIFT`算法进行梯度提取总和
 - size(ri,rj)
   - 公式: <a href="http://www.codecogs.com/eqnedit.php?latex=s_{size}(r_i,r_j)&space;=&space;1&space;-&space;\frac{size(r_i)&plus;size(r_j)}{size(im)}" target="_blank"><img src="http://latex.codecogs.com/svg.latex?s_{size}(r_i,r_j)&space;=&space;1&space;-&space;\frac{size(r_i)&plus;size(r_j)}{size(im)}" title="s_{size}(r_i,r_j) = 1 - \frac{size(r_i)+size(r_j)}{size(im)}" /></a>
   - 解释: ri和rj为选取的两个区域,im为整张图片,意思为优先合并小区域
 - fill(ri,rj)
   - 公式: <a href="http://www.codecogs.com/eqnedit.php?latex=fill(r_i,r_j)&space;=&space;1-\frac{size(BB_ij)-size(r_i)-size(r_i)}{size(im)}" target="_blank"><img src="http://latex.codecogs.com/svg.latex?fill(r_i,r_j)&space;=&space;1-\frac{size(BB_ij)-size(r_i)-size(r_i)}{size(im)}" title="fill(r_i,r_j) = 1-\frac{size(BB_ij)-size(r_i)-size(r_i)}{size(im)}" /></a>
   - 解释: 重叠面积占比越多越大

### 结果计算
 - 计算四个相似度之和，按照整体流程描述的合并思路进行合并
