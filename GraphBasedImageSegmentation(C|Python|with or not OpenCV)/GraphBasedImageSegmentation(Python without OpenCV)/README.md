# Efficient Graph-Based Image Segmentation

## 原文链接
 - http://www.cs.cornell.edu/~dph/papers/seg-ijcv.pdf

## 基础知识
 - 一张图是由不同的像素点构成的，本文的计算和构建都是基于像素点的运算，即`(RGB)`值
 - 高斯模糊/拉普拉斯变换:用于转换图像，减少图像噪声的平滑算法
 - 最小生成树`(Minimum Spanning Tree | MST)`指的是，在图中建立一个连通图并且没有回路是生成树，而最小生成树指的是构成结果权值最小
 - 不同像素点之间的差:即`RGB`值之间的欧氏距离
   - <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{80}&space;\sqrt{(R_1^2-R_2^2)&plus;(G_1^2-G_1^2)&plus;(B_1^2-B_2^2)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{80}&space;\sqrt{(R_1^2-R_2^2)&plus;(G_1^2-G_1^2)&plus;(B_1^2-B_2^2)}" title="\sqrt{(R_1^2-R_2^2)+(G_1^2-G_1^2)+(B_1^2-B_2^2)}" /></a>
 - 并查集算法(union find set)以及克鲁斯卡尔算法(Kruskal)，使用边建立并查集，并且使用kruskal进行搜索合并

## 早期的分割方法
 - Zahn提出了一种*基于图的最小生成树（MST）*的分割方法，用来进行点聚类以及图像分割，前者权值是点间距离，后者权值是像素差异。
  - 不足：根据阈值不同，会导致高可变性（大约是色彩对比强的一个区域）区域划分为多个区域；将ramp和constant region合并到一起。
 - Urquhart提出用边相连的点中边权值最小的进行归一化，找周围相似的。
 - 根据各个区域是否符合某种均匀性标准来分割，找均匀强度或梯度的区域，不适用于某个变化很大的区域。
 - 使用特征空间聚类：通过平滑数据——给定半径的超球面对各个点扩张其连通分量，找到簇，来保持该区域的边界，并对数据进行转换。

## 基于图的分割
### 定义
 - `G`:将图像由像素点转化为图
 - `V`:每一个像素点都是图中的点
 - `E`:任意两个相邻像素点之间边
 - `C`:被划分的`Segmentation`,一个`C`中有至少1个像素点
 - `Int(C)`:区域内最小生成树权值最大的边，表示的是，记为
   - <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{80}&space;Int(C)&space;=&space;\max_{e\in&space;MST(C,E)}w(e)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{80}&space;Int(C)&space;=&space;\max_{e\in&space;MST(C,E)}w(e)" title="Int(C) = \max_{e\in MST(C,E)}w(e)" /></a>
 - `Dif(C1,C2)`:表示C1和C2之间的距离，记为
   - <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{80}&space;Dif(C_1,C_2)&space;=&space;\min_{v_i\in&space;C_1,v_j\in&space;C_2,(v_i,v_j)\in&space;E}w(v_i,v_j)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{80}&space;Dif(C_1,C_2)&space;=&space;\min_{v_i\in&space;C_1,v_j\in&space;C_2,(v_i,v_j)\in&space;E}w(v_i,v_j)" title="Dif(C_1,C_2) = \min_{v_i\in C_1,v_j\in C_2,(v_i,v_j)\in E}w(v_i,v_j)" /></a>
 - 最后要形成的分组要求是(表示了所有区域之间的最小距离都比区域内的最大距离和权值的和要大)
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\LARGE&space;D(C_1,C_2)&space;=\lbrace_{false\quad&space;otherwise}^{true\quad&space;if&space;Dif(C_1,C_2)>MInt(C_1,C_2)}" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\LARGE&space;D(C_1,C_2)&space;=\lbrace_{false\quad&space;otherwise}^{true\quad&space;if&space;Dif(C_1,C_2)>MInt(C_1,C_2)}" title="\LARGE D(C_1,C_2) =\lbrace_{false\quad otherwise}^{true\quad if Dif(C_1,C_2)>MInt(C_1,C_2)}" /></a>
 - 其中`MInt(C1,C2)`的值为:
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;MInt(C_1,C_2)&space;=&space;min(Int(C_1)&plus;\tau(C_1),Int(C_2)&plus;\tau(C_2)))" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;MInt(C_1,C_2)&space;=&space;min(Int(C_1)&plus;\tau(C_1),Int(C_2)&plus;\tau(C_2)))" title="\large MInt(C_1,C_2) = min(Int(C_1)+\tau(C_1),Int(C_2)+\tau(C_2)))" /></a>
 - 阈值设定的原因是为了在开始时，因为只有单个像素点，那么点内的距离为0，而点之间的距离还存在，那么导致无法合并。所以加入阈值。
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;\tau(C)&space;=&space;k/\left|C\right|" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;\tau(C)&space;=&space;k/\left|C\right|" title="\large \tau(C) = k/\left|C\right|" /></a>

### 分割算法（与克鲁斯卡尔算法构建最小生成树有密切关系。）
 - 输入是一个有n个节点和m条边的图G，输出是一系列区域。步骤如下：
 - 0.将边按照权重值以非递减方式排序
 - 1.最初的分割记为S（0），即每一个节点属于一个区域。
 - 2.按照以下的方式由S(q-1)构造S(q)：记第q条边连接的两个节点为vi和vj，如果在S(q-1)中vi和vj是分别属于两个区域并且第q条边的权重小于两个区域的区域内间距，则合并两个区域。否则令S(q) = S(q-1)。
 - 3.从q=1到q=m，重复步骤2。
 - 4.返回S(m)即为所求分割区域集合。
---

### 关键点解释
 - 因为边的集合在最开始进行排序，所以不是判断两个区域之间的最短边，而是取当前未连通的最短边，看看边的两个节点所属的图能不能合并
 - 因为一个区域是一个MST，所以这个区域的信息都存储在根节点，这样就避免了连通图的生成，因为当都检索到一个节点时说明出现了连通图
 - 为什么每找到的一条符合条件可以连接的边时，该边就是当前图中最大的边。
 - 因为边的选取是从小到大的，所以之后选到的边都会比之前的大

## 补充
### 高斯滤波器
 - 高斯变换就是用高斯函数对图像进行卷积，高斯滤波器是一种线性滤波器，能够有效抑制噪声，并平滑图像。其实质是取滤波器窗口内像素的均值作为输出。
 - 高斯函数公式如下:
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;f(x)&space;=&space;\frac{1}{\sigma\sqrt{2\pi}}e^{&space;-\frac{(x-\mu)^2}{2\sigma^2}}" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;f(x)&space;=&space;\frac{1}{\sigma\sqrt{2\pi}}e^{&space;-\frac{(x-\mu)^2}{2\sigma^2}}" title="\large f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{ -\frac{(x-\mu)^2}{2\sigma^2}}" /></a>
其中，`u`是`x`的均值,`σ`是方差。
 - 由一维函数，我们可以推导出二维函数的公式如下：
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;f(x,y)&space;=&space;\frac{1}{2\pi&space;\sigma^2}&space;e^{-\frac{(x^2&plus;y^2)}{2\sigma^2}&space;}" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;f(x,y)&space;=&space;\frac{1}{2\pi&space;\sigma^2}&space;e^{-\frac{(x^2&plus;y^2)}{2\sigma^2}&space;}" title="\large f(x,y) = \frac{1}{2\pi \sigma^2} e^{-\frac{(x^2+y^2)}{2\sigma^2} }" /></a>
 - 高斯函数在图像处理中的使用，实际上就是对每个像素点的周边像素取平均值，从而达到平滑的效果，在取值(周边半径)时，周围像素点的半径越大，则图像的模糊度就越强。在实际计算时，利用高斯模糊按正态曲线分配周边像素的权重，从而求中心点的加权平均值。
 - 高斯模糊的具体计算方式如下：
   - 1.将中心点周围的八个点带入到高斯函数中，从而得到权重矩阵A1；
   - 2.为使归一化，将矩阵A1中的各个点除以所有点(9个点)的权重和，得到归一化后的权重矩阵A2;
   - 3.图片原始的像素矩阵分别乘以A2中各自的权重值，将得到的所有点的值加起来求平均，便得到中心点的高斯模糊值。图像中其余点相同求法。
   - 注：1.彩色图片，可对RGB三通道分别作高斯模糊。
 - 2.`σ`代表数据的离散程度，`σ`越大，中心系数越小，图像越平滑；反之，反之。

### 拉普拉斯变换：是为解决傅立叶变换等幅振荡的缺点。

 - 首先了解一下傅立叶变换：傅立叶变换是一种物理上探究频谱的方法，三角公式是：
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;f(t)&space;=&space;\sum_{n=1}^\infty&space;A_ncos(nw_0t&plus;\varphi_n)&plus;B" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;f(t)&space;=&space;\sum_{n=1}^\infty&space;A_ncos(nw_0t&plus;\varphi_n)&plus;B" title="\large f(t) = \sum_{n=1}^\infty A_ncos(nw_0t+\varphi_n)+B" /></a>
   - 其中,`w0`表示基波。
 - 由欧拉公式：
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;\left\{&space;\begin{aligned}&space;e^{ix}&space;=cosx&plus;isinx,&space;\\&space;e^{-ix}&space;=cosx&space;-isinx,&space;\end{aligned}&space;\right." target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;\left\{&space;\begin{aligned}&space;e^{ix}&space;=cosx&plus;isinx,&space;\\&space;e^{-ix}&space;=cosx&space;-isinx,&space;\end{aligned}&space;\right." title="\large \left\{ \begin{aligned} e^{ix} =cosx+isinx, \\ e^{-ix} =cosx -isinx, \end{aligned} \right." /></a>
 - 将傅立叶三角形式公式中的正余弦函数用指数函数表示，改写为用复指数表示的公式，如下：
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;f(t)&space;=&space;\sum_{-\infty}^\infty&space;F(nw_0)e^{jw_0t}" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;f(t)&space;=&space;\sum_{-\infty}^\infty&space;F(nw_0)e^{jw_0t}" title="\large f(t) = \sum_{-\infty}^\infty F(nw_0)e^{jw_0t}" /></a>
 - 将上述公式改为积分形式，即得到复指数形式公式为：
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;F(w)&space;=\int_{-\infty}^\infty&space;f(t)e^{-jwt}dt" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;F(w)&space;=\int_{-\infty}^\infty&space;f(t)e^{-jwt}dt" title="\large F(w) =\int_{-\infty}^\infty f(t)e^{-jwt}dt" /></a>
 - 但由于傅立叶变换是等幅振荡的正弦波，故当f(t)不断趋向无穷时，此时函数将不再收敛，这时候便不再适合使用傅立叶变换。于是，我们引入一个衰减因子，对其作变换。对函数y=f(t)乘上一个<a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;e^{\sigma&space;t}" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;e^{\sigma&space;t}" title="\large e^{\sigma t}" /></a>,其中，`σ`>0。
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;F(w)&space;=\int_{-\infty}^\infty&space;f(t)e^{-\sigma&space;t}e^{-jwt}dt" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;F(w)&space;=\int_{-\infty}^\infty&space;f(t)e^{-\sigma&space;t}e^{-jwt}dt" title="\large F(w) =\int_{-\infty}^\infty f(t)e^{-\sigma t}e^{-jwt}dt" /></a>
 - 对上式进行合并同类项，可得到<a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;F(w)&space;=\int_{-\infty}^\infty&space;f(t)e^{-t(\sigma&plus;jw)}dt" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;F(w)&space;=\int_{-\infty}^\infty&space;f(t)e^{-t(\sigma&plus;jw)}dt" title="\large F(w) =\int_{-\infty}^\infty f(t)e^{-t(\sigma+jw)}dt" /></a>
 - 我们将指数中的`σ+jw`最初的分割记为S，于是得到拉普拉斯公式：
   - <a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;F(w)&space;=\int_{-\infty}^\infty&space;f(t)e^{-st}dt" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;F(w)&space;=\int_{-\infty}^\infty&space;f(t)e^{-st}dt" title="\large F(w) =\int_{-\infty}^\infty f(t)e^{-st}dt" /></a>
 - 由上式推导，很清楚的知道，当s=jw时，拉普拉斯函数就变成了傅立叶函数，也就相当于拉氏不再具有衰减功能。
 - 又由上述公式可以很直观地看到当取值<a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;\sigma_0" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;\sigma_0" title="\large \sigma_0" /></a>刚好收敛时，则<a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;\sigma&space;>&space;\sigma_0" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\large&space;\sigma&space;>&space;\sigma_0" title="\large \sigma > \sigma_0" /></a>的区域全都收敛。
