# StatMS
软件简介

本软件是一款面向生物信息学分析的工具，旨在为科研人员提供简单高效的数据分析功能。软件主要包括判别分析、聚类分析、单因素和多因素分析等模块，帮助用户快速挖掘数据中的潜在规律，生成可视化结果，用于学术研究或行业应用。
软件设计灵感来源于 SIMCA 和 MetaboAnalyst，结合了经典的分析算法与直观的用户界面，用户无需编程经验即可操作。

2.安装与启动

系统要求： 

操作系统：Windows 10 及以上 / macOS

内存：8GB 及以上

Python 环境：3.8 版本及以上，3.12版本及以下

安装步骤：

1.下载软件安装包，解压到本地目录。

使用下列命令直接下载至当前路径下或直接下载zip文件至本地

wget https://github.com/hjzeng111/StatMS/releases/download/StatMS_V1.0.1/StatMS.zip -OutFile StatMS.zip

2.打开终端，进入软件目录，运行以下命令：

pip install -r requirements.txt

如有网络连接超时等情况可使用

(pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple)清华源镜像

(pip install -r requirements.txt --index-url https://mirrors.aliyun.com/pypi/simple/)阿里云镜像

(pip install -r requirements.txt --index-url https://pypi.mirrors.ustc.edu.cn/simple/)中科大镜像

(pip install -r requirements.txt--index-url https://mirrors.huaweicloud.com/repository/pypi/simple/)华为云镜像

3.安装完成后，运行以下命令启动软件：

python work.py
