# 用户评论标签抽取

## 原项目

[Github](https://github.com/shijing888/CommentsMining) (感觉代码不全)

[Blog](https://blog.csdn.net/shijing_0214/article/details/71036808) 

本项目是在原项目上进行的改进与修正

数据来源于原项目Github

## 对原project的改进与修正

- 用的Python
- Stanford的coreNLP下载巨慢，换了[pyltp](https://pyltp.readthedocs.io/zh_CN/latest/api.html#id4) and [jieba](https://github.com/fxsjy/jieba) 各写了一套
- 淘宝评论语言口语化程度高，进行语料分句预处理，每一个，分一句
- 淘宝评论语法过于不标准，脏数据太多，抽取规则除了基于依存句法分析，还加入词性标注结果（基于jieba的一套没有使用依存句法方面的信息，因为jieba没提供这个功能）
- 该project数据很少，且聚类本身就起到了消重作用，使用simhash预消重不会显著提升速度，同时造成标签多样性和丰富性降低，故弃之不用
- dbscan算法本身不能得到簇的几何中心，这里改用简单好用的kmeans

## pyltp requirements

- python 3.6 or lower
- pip install scikit-learn, gensim, pyltp
- [百度云](https://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569#list/path=%2F)进去，进入 ltp-models/3.4.0/ 根据你的系统下载其中一个压缩包(windows下载.zip, linux下载.tar) 下载好后将压缩包在该project根目录解压

## jieba requirement

- pip install jieba



## 参数简介

所有参数都在config.json里调

#### segmentMethod 

用jieba还是pyltp分词



#### selectedMethod

用jieba还是pyltp抽取规则



#### vectorSize

词向量长度



#### nCluster

聚类个数



#### ltpPath

放ltp模型的文件夹



## 文件夹简介

#### data/ 

下面包括训练集，测试集，最终输出，停用词表和情感词典



#### train/ 

下面包括训练集处理过程中的中间数据，路径改为“”后则不输出



#### test/ 

同上



#### models/ 

是生成的词向量模型所在文件夹





