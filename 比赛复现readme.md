# README (中文版)

## 🌟 北大金融科技社团 2024 内培项目 - Optiver 赛题研究

[简体中文](#) | [English](./README_EN.md)

---

### 项目简介

欢迎来到北大金融科技社团 2024 年内培活动的项目！🎉  
本项目以 **Kaggle 2023 Optiver: Trading at Close** 比赛题目为核心，通过从 **探索性数据分析（EDA）** 到 **特征工程（FE）**，再到 **模型训练与优化** 的完整流程实现，帮助社团成员学习与掌握金融科技领域的建模与研究能力。

我们为每一个模块设计了独立的文件结构，代码分为可复现的 **src** 模块和实验记录的 **ipynb** 文件。每一步都记录了我们对于特征选择、模型搭建以及优化的努力 💪！

---

### 文件结构

```plaintext
Project Structure
├── AFT-EDA                     # 数据探索模块
│   └── 各成员的个人 EDA 文件
├── AFT-FE                      # 特征工程模块
│   ├── data                    # 数据及特征相关文件
│   ├── pkl_files               # 特征预处理模型保存
│   └── 相关代码与配置文件
├── AFT-GAT                     # 图注意力网络模块
│   └── src                     # 核心代码
├── AFT-Master                  # 综合模型模块
│   └── src                     # 核心代码
├── AFT-RNN                     # 循环神经网络模块
│   └── src                     # 核心代码
├── AFT-TCN                     # 时间卷积网络模块
│   └── src                     # 核心代码
└── 2024AFT内培-比赛复现组.pptx # 项目总结与展示
```

---

### 亮点功能 ✨

1. **模块化代码设计**：每种模型均以模块化形式实现，方便研究与扩展。
2. **清晰的实验记录**：包含 Jupyter Notebook 的原始实验文件，记录完整的研究过程。
3. **TODO 清单**：持续优化与开发，期待你的贡献！

---

### 📌 TODO 清单

以下工作亟待完成：

- [ ] 上传数据
- [ ] 更新树模型代码
- [ ] 统一 `requirements.txt` 文件
- [ ] 规范化文件路径
- [ ] 一键运行全流程梳理
- [ ] 实现结果的统一可视化
- [ ] 抽取通用 `dataloader` 架构，支持不同数据维度
- [ ] 抽取通用模型架构，实现神经网络的快速切换

---

### 贡献指南 ❤️

我们欢迎所有对金融科技和机器学习感兴趣的朋友加入我们！  
如果你有改进代码、完善文档或实现新功能的想法，请随时提交你的 PR 或联系项目负责人！🌟

---

### 开始使用 🚀

1. 克隆本项目到本地：
   ```bash
   git clone https://github.com/your_repo_name.git
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行对应模块的代码。

---

### 联系我们 📧

如有问题或建议，请通过社团官方邮箱联系我们：**fintech@pku.edu.cn**

期待你的加入，一起探索金融科技的更多可能！🚀

---

# README (English Version)

## 🌟 Peking University FinTech Society 2024 Training Program - Optiver Competition Research

[简体中文](./README.md) | [English](#)

---

### Project Overview

Welcome to the **Peking University FinTech Society** 2024 training program! 🎉  
This project centers on the **Kaggle 2023 Optiver: Trading at Close** competition, with a complete workflow from **Exploratory Data Analysis (EDA)** to **Feature Engineering (FE)** and **Model Training and Optimization**. It aims to equip our members with robust modeling and research skills in the FinTech domain.

Each module is designed with a well-structured directory, including reproducible **src** code and experiment-logging **ipynb** files. Every step reflects our dedication to feature selection, model development, and optimization 💪!

---

### Directory Structure

```plaintext
Project Structure
├── AFT-EDA                     # Data exploration module
│   └── Members' personal EDA files
├── AFT-FE                      # Feature engineering module
│   ├── data                    # Data and feature-related files
│   ├── pkl_files               # Preprocessing model files
│   └── Codes and configuration files
├── AFT-GAT                     # Graph Attention Network module
│   └── src                     # Core code
├── AFT-Master                  # Integrated model module
│   └── src                     # Core code
├── AFT-RNN                     # Recurrent Neural Network module
│   └── src                     # Core code
├── AFT-TCN                     # Temporal Convolutional Network module
│   └── src                     # Core code
└── 2024AFT_Training_Summary.pptx # Project summary and presentation
```

---

### Key Features ✨

1. **Modular Design**: Each model is implemented in a modular structure, enabling easy research and extension.
2. **Comprehensive Experiment Logs**: Jupyter Notebook files provide a detailed record of the research process.
3. **Ongoing Enhancements**: A clear TODO list for further development—your contribution matters!

---

### 📌 TODO List

We are working on the following goals:

- [ ] Unify the `requirements.txt` file
- [ ] Standardize file paths
- [ ] Enable one-click execution
- [ ] Implement unified result visualization
- [ ] Extract a general `dataloader` architecture for datasets with varying dimensions
- [ ] Extract a general model architecture for flexible neural network switching

---

### Contribution Guide ❤️

We welcome anyone interested in FinTech and machine learning to join us!  
Feel free to submit your PRs or reach out to the project maintainers if you have ideas to improve the code, enhance the documentation, or implement new features! 🌟

---

### Getting Started 🚀

1. Clone the repository:
   ```bash
   git clone https://github.com/your_repo_name.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the code for the corresponding module.

---

### Contact Us 📧

For inquiries or suggestions, feel free to reach out via our official email: **fintech@pku.edu.cn**

Let’s explore the endless possibilities of FinTech together! 🚀