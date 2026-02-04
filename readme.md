DeepQCFN
复现的论文是：Deep_Q-Network_Based_Task_Scheduling_Strategy_in_Computing_First_Network

这是一个基于深度强化学习的 算力优先网络 (Compute First Networking, CFN) 仿真与优化库。

核心目标是解决 任务卸载（Task Offloading）与资源调度 问题，即决定将不同优先级的计算任务分配给哪一个边缘服务器（Edge Server），以最小化总延迟（加权延迟）。

库的核心功能分解：
仿真环境 (
deepqcfn/environment/cfn_env.py
):
场景: 模拟了一个由 TEs (终端设备) 和 ESs (边缘服务器) 组成的网络环境。
任务生成: 每个时间片（Time Slot）终端设备会生成带有不同 数据量 和 优先级 的计算任务。
物理模型: 考虑了详细的通信与计算模型，包括：
传输延迟: 取决于信道带宽、噪声和传输功率。
排队延迟: 服务器当前的负载情况。
计算延迟: 取决于任务大小和服务器算力。
迁移延迟: 如果任务需要在不同节点间迁移产生的开销。
核心算法: PST-DQN:
这个库似乎是 PST-DQN (Priority Sort Task - Deep Q Network) 算法的实现。
第一步 (PST): 在每个时间片开始时，使用 Priority Sort Algorithm 对所有生成的任务按优先级进行排序（
cfn_env.py
 line 68）。这确保了高优先级任务优先获得调度机会。
第二步 (DQN): 使用 Deep Q-Network (
deepqcfn/agent/agent.py
) 为每个排序后的任务选择最佳的边缘服务器。
状态 (State): 包含当前任务特征（大小、优先级、源节点）以及所有服务器的实时状态（负载延迟、距离）。
动作 (Action): 选择其它的边缘服务器之一。
奖励 (Reward): 负的加权延迟 (-1 * Total_Delay * Priority)，即延迟越低、高优先级任务处理越快，奖励越高。
代码结构:
main.py
: 主程序入口，负责初始化环境和智能体，运行训练循环，并绘制结果。
deepqcfn/models/: 包含网络组件的物理模型（任务 task.py，网络拓扑 network.py，服务器 server.py）。
deepqcfn/agent/: 包含强化学习智能体（PSTDQNAgent）和神经网络定义。

## 如何运行

### 1. 环境准备
确保已安装 Python 3.8+。推荐创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate
```

安装依赖：
```bash
pip install numpy matplotlib torch gymnasium
```

### 2. 运行仿真
本项目包含两个主要的运行脚本：

**A. 运行完整的 PSTDQN 算法 (推荐)**
这是论文的核心复现，包含生成器（Generator）和基于进化策略的修正过程。
```bash
python run_pstdqn.py
```
运行结束后，会生成 `pstdqn_training_results.png` 展示训练曲线。

**B. 运行基础 DQN 基准**
这是一个标准的 DQN 实现，不包含 PSTDQN 的进化增强部分，可作为对比基准。
```bash
python main.py
```
运行结束后，会生成 `training_results.png`。