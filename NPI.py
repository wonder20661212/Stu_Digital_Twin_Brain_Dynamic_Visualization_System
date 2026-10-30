import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#让代码自动选择用显卡（GPU），还是CPU运行。


class ANN_MLP(nn.Module):
#class：定义一个新模型, ANN_MLP：模型名字叫ANN_MLP（人工神经网络 + 多层感知机）;nn.Module继承PyTorch的神经网络模板，让它能训练、能更新参数、能做预测。
    "Use MLP as a surrogate brain"#N代表着N个脑区
    #input_dim  = 每个样本的特征数(3N规模的输入层) hidden_dim = 并行神经元的数量(2N规模的隐藏层) latent_dim = 更抽象特征的数量(0.8N规模的隐藏层) output_dim = 你要预测的变量数(N规模的输出层)
    #多个历史时间点拼接作为输入，使mlp能够隐式建模时间依赖关系 隐藏层：足够你和脑区之间的非线性交互关系，同时又防止参数爆炸，过拟合
    #latent高维输入压缩到较低的维度，最后再预测下一个全脑的状态。
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):#用前三个时间点预测下一个时间点的输出
        super().__init__()#继承PyTorch模型功能
        #func定义的可复用的计算流水线
        self.func = nn.Sequential(#MLP前向传播流水线：前3个时间点拼接 → 升到2N → 压缩到0.8N → 预测下一个时间点
            nn.Linear(input_dim, hidden_dim),#nn.linear(a,b) b个神经元 每个都有a个权重（第一层线性层：输入信号 → 隐藏层神经元。(将3个时间段拼接的数据（3*N）转换为隐藏层2N的数据)）
            nn.ReLU(),#加入非线性，让模型能学习大脑复杂关系。（将负数变为0）
            nn.Linear(hidden_dim, latent_dim),#第二层线性层：把2N维特征压缩到0.8N
            nn.ReLU(),#再一次加入非线性。
            nn.Linear(latent_dim, output_dim),#输出层：预测下一个时间点的全脑信号。0.8N到1N
        ).to(device)
    #定义的x如何变成y
    def forward(self, x):
        return self.func(x)#输入信号走一遍网络流水线，输出预测结果。



class ANN_CNN(nn.Module):#用 CNN 做替代脑模型，专门处理时序型脑信号（fMRI）

    "Use CNN as a surrogate brain"
#in_channels是脑区数量N；data_length是输入的时间长度（3 个时间点）
    def __init__(self, in_channels, hidden_channels, out_channels, data_length):
        super().__init__()
        self.in_channels = in_channels
        self.data_length = data_length
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size = 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size = 2), 
        ).to(device)#这是 1D 卷积（一维卷积），它在时间维度上滑动，提取相邻时间点的规律，比如：t 和 t+1 之间有什么变化模式，适合时序信号（脑电、fMRI、语音）。两层卷积作用：第一层提取简单时序模式，第二层提取更复杂时序模式
        self.Linear = nn.Linear(out_channels * (data_length - 2), in_channels).to(device)
#把卷积提取的特征压平，最后映射成 N 个脑区的预测信号，和 MLP 输出一样预测下一个时间点
    def forward(self, x):#整理输入形状，用卷积提取时间规律，输出下一个时间点的预测
        x = x.view(-1, self.data_length, self.in_channels)
        x = x.permute(0, 2, 1)
        pred = self.CNN(x)
        pred = torch.squeeze(pred)
        pred = self.Linear(pred)
        return pred



class ANN_RNN(nn.Module):#用 RNN 做替代脑模型

    "Use RNN as a surrogate brain"
    
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, data_length):
        super().__init__()#初始化网络
        self.input_dim = input_dim
        self.data_length = data_length
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        ).to(device)#把输入信号做特征压缩
        self.rnn = nn.RNN(input_size = 1, hidden_size = latent_dim, batch_first = True).to(device)#接收上一时刻的状态和当前时刻的输入，输出新一时刻的状态
        self.output = nn.Linear(latent_dim, output_dim).to(device)#输出预测的脑信号

    def forward(self, x):#把输入整理好，丢进编码器
        x = x.view(-1, self.data_length, self.input_dim)
        encodes = self.enc(x)
        ht, _ = self.rnn(torch.zeros((x.shape[0], 1, 1)).to(device), torch.permute(encodes[:, self.data_length-1:, :], (1, 0, 2)).contiguous())
        return self.output(ht)[:,0,:]#给 RNN 一个初始状态，RNN 自己一步步往前走，输出一个最终隐藏状态 ht



class ANN_VAR(nn.Module):# 线性自回归模型，最简单的“大脑动态模型”，

    "Use VAR as a surrogate brain"

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.func = nn.Linear(input_dim, output_dim).to(device)
    
    def forward(self, x):#输入 × 权重矩阵 → 直接输出预测
        return self.func(x)


#对应公式(1)
def multi2one(time_series, steps):#time_series：原始脑信号（时间，脑区）；steps：用前几步预测。论文里是3

    "Split the data into several input-output pairs"

    n_area = time_series.shape[1]#n_area = 脑区数量N
    n_step = time_series.shape[0]#n_step = 总时间点数
    # print("n_area = ", n_area)
    # print("n_step = ", n_step)
    #构建空的输入、输出矩阵
    #n_step - steps= 总时间点 - 用来预测的时间步数= 最终能生成多少条训练数据
    input_X = np.zeros((n_step - steps, n_area * steps))#输入长度 = 脑区数 × 步数（3）
    target_Y = np.zeros((n_step - steps, n_area))#输出长度 = 脑区数
    for i in range(n_step - steps):#遍历每一个可以用来预测的训练数据
        #把连续steps个时间点全部压扁成一行|(3个时间点, N个脑区) → 压扁 → (1行, 3×N特征)
        input_X[i] = time_series[i:steps+i].flatten()#时间维度塞入特征维度，模型自学哪些ROI重要，哪些时间点重要，时间顺序如何影响结果。
        #压扁下一个时间点(预测)
        target_Y[i] = time_series[steps+i].flatten()#数值顺序没有丢，MLP模型感知时间顺序只能通过模型去悟，RNN/transfomer结构上时间敏感
        #input_X形状：(样本数，脑区数×3)
        #target_Y形状：(样本数，脑区数)
    return np.array(input_X), np.array(target_Y)



def train_NN(model, input_X, target_Y, batch_size = 50, train_set_proportion = 0.8, num_epochs = 100, lr = 1e-3, l2 = 0):

    "Use empirical data to tune the model"#用真实脑数据调整模型参数θ
#train_inputs和test_inputs的区别看':'的位置
    #torch.tensor把numpy数组转为PyTorch张量,.to(device)把数据送到 GPU/CPU
    train_inputs = torch.tensor(input_X[:int(train_set_proportion * input_X.shape[0])], dtype = torch.float).to(device)#前80%数据做训练集
    train_targets = torch.tensor(target_Y[:int(train_set_proportion * target_Y.shape[0])], dtype = torch.float).to(device)
    test_inputs = torch.tensor(input_X[int(train_set_proportion * input_X.shape[0]):], dtype = torch.float).to(device)#后20%的数据做测试集
    test_targets = torch.tensor(target_Y[int(train_set_proportion * target_Y.shape[0]):], dtype = torch.float).to(device)
    #TensorDataset把(输入，输出)打包成对；DataLoader自动分成小批量（batch=50）；shuffle=True训练时打乱数据，防止模型记住顺序；shuffle=False测试不打乱，保证结果可复现。
    train_dataset = data.TensorDataset(train_inputs, train_targets)
    test_dataset = data.TensorDataset(test_inputs, test_targets)
    train_iter = data.DataLoader(train_dataset, batch_size, shuffle = True)#训练要shuffle 防止模型记住顺序
    test_iter = data.DataLoader(test_dataset, batch_size, shuffle = False)#测试不需要shuffle 方便复现。

    #MSELoss为均方误差，预测值与真实值差的平方和（论文公式 (2)），衡量模型预测值与真实脑信号的误差，用于反向传播更新参数
    loss = nn.MSELoss()#MSE（Mean Squared Error）：预测错一点 → 惩罚一点 错得越多 → 惩罚成平方地变大
    #Adam优化器做了升级,记住过去梯度的方向，记住梯度的变化幅度，对每个参数自适应调节步长。
    # weight_decay:它等价于在 loss 里偷偷加了一项惩罚L2 正则如果没有它，模型可能会：用非常大的权重 把训练数据“硬记住”
    #model.parameters()：待优化的神经网络所有可学习参数
    #lr=1e-3：学习率，论文固定为1e-3
    #weight_decay=l2：L2 正则化，抑制参数过大，防止过拟合
    trainer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = l2)
    #两个空列表，分别记录每一轮的训练损失和测试损失，用于观察模型是否收敛、是否过拟合。
    train_epoch_loss = []; test_epoch_loss = []
    for _ in range(num_epochs):#遍历完整数据集num_epochs次，论文使用60 轮，此处设为 100 轮
        model.train()#切换模型为训练模式，启用dropout、batchnorm等训练特有的结构
        for X, y in train_iter:#按小批量遍历训练集，从训练集train_iter中取出输入变量X(前面打包的一个数据集，将三个输入的时间点的数据集压缩成一个)和输出变量y(同上)
            y_hat = model(X)#模型前向传播，输入X得到预测值
            l = loss(y_hat, y)#计算当前batch的预测误差
            trainer.zero_grad()#清空上一轮梯度，避免梯度累积
            l.backward()#反向传播：自动计算所有参数的梯度（对每一个参数θ，计算导数 dl/dθ），算出每个参数（θ）要怎么改才能减少误差，(如果导数正：θ变大 → l变大 → 要减小θ；如果导数负：θ变大 → l变小 → 要增大θ)
            trainer.step()#根据梯度更新所有可学习参数，最小化损失函数（更新参数θ）θ_new = θ_old - lr * 导数
        model.eval()#切换模型为评估模式，关闭训练层，保证输出稳定
        with torch.no_grad():#禁用梯度计算
            total_loss = 0; total_num = 0
            for X, y in train_iter:#遍历训练集，累加总误差与总样本数,计算平均损失，存入列表.记录模型现在学得有多准，方便后面画曲线看模型是不是越练越准。
                y_hat = model(X)
                l = loss(y_hat, y)
                total_loss += l * y.shape[0]
                total_num += y.shape[0]
            train_epoch_loss.append(float(total_loss / total_num))#训练误差=看模型是否拟合真实脑动态
            total_loss = 0; total_num = 0
            for X, y in test_iter:#遍历测试集，计算泛化误差，存入列表.记录泛化能力：模型能不能预测新数据
                y_hat = model(X)
                l = loss(y_hat, y)
                total_loss += l * y.shape[0]
                total_num += y.shape[0]
            test_epoch_loss.append(float(total_loss / total_num))#泛化误差=评估模型是否能可靠预测未知脑信号
    return model, train_epoch_loss, test_epoch_loss#返回三项核心结果：训练完成的 surrogate brain 模型，训练损失曲线，测试损失曲线

#对应论文中的公式(3)
#计算全脑的功能连接矩阵（FC）,输入 signals：形状 = (时间点 T, 脑区数 N)，就是 fMRI 脑信号矩阵
def corrcoef(signals):#相关系数（correlation coefficient）矩阵 计算任意两条时间序列的“同步程度” 在功能连接分析中，皮尔逊相关是最常用、也最直观的方法
#计算N个脑区之间的两两相关
    "Calculate FC of the empirical data"
    #为什么要signals.T signals.shape = (T, ROI) 因为corrcoef要的是(ROI, T)
    return torch.corrcoef(torch.tensor(signals.T, dtype = torch.float).to(device)).detach().cpu().numpy()
    #signals.T是signals的转置矩阵
#torch.corrcoef(...)：计算 皮尔逊相关系数矩阵，输出形状：(N, N)。F矩阵 [i,j] = 脑区i和 脑区j的信号相似度越接近1→ 越同步；越接近0 →越无关


#用训练好的神经网络自己生成一段“虚拟fMRI时间序列”，再对这段时间序列 计算FC（相关系数矩阵）。
def model_FC(model, node_num, steps):
#输入：训练好的模型、脑区数量、用前几步预测（steps=3）
#输出：模型自己生成的脑信号算出来的 FC 矩阵
    "Simulate data with random noise and calculate model-FC"

    NN_sim = []#建一个空列表，用来存模型生成的每一个时间点的脑信号
    for _ in range(steps): NN_sim.append(np.zeros(node_num)) #_ 只是一个“占位变量名”，表示“我不打算用这个循环变量”(先放 steps 个全0信号作为初始状态)
    for _ in range(1200):#让模型连续滚动预测1200个时间点，模拟一段长时程 fMRI 信号
        #rand:random n:normal 会生成 n 个服从“标准正态分布”的随机数 生成“均值 0、方差 1”的高斯噪声
        #0.1用于控制噪声的强度 对每个节点、每个时间步加入小的高斯噪声 模拟神经活动的不确定性 避免系统陷入完全确定性的轨道 更贴近真实 EEG / BOLD 信号
        #在随机噪声驱动下，对训练好的动力学模型进行长时间滚动模拟（rollout），从而生成自发脑活动时间序列。
        noise = 0.1 * np.random.randn(steps * node_num)#输入加小幅度高斯噪声，论文明确要求必须加噪声模拟真实脑活动的随机性，若不加噪声，模型会进入固定轨道，不真实。
        model_input = np.array(NN_sim[-steps:]).flatten() + noise#FC 是在“自发活动 + 噪声”下统计出来的 而不是 deterministic rollout.取最近 3 个时间点（NN_sim[-steps:]）压平成一行（flatten）加上噪声就是模型的输入
        #isinstance(对象, 类/子类)判断 model 这个对象，是不是用 ANN_RNN 这个类（或它的子类）创建出来的 类/子类 true else false
        if isinstance(model, ANN_RNN): NN_sim.append(model(torch.tensor(model_input, dtype = torch.float).to(device)).detach().cpu().numpy()[0])
        else: NN_sim.append(model(torch.tensor(model_input, dtype = torch.float).to(device)).detach().cpu().numpy())
    #pred = model(input) 这段代码的作用是：用模型预测下一个时间点的全脑活动；如果是RNN，就只取输出，不要隐藏状态；最后把结果转成numpy，接到模拟时间序列后面。
    # ANN: pred = output
    # RNN: pred = (output, hidden)
    #if RNN:
    #    pred = pred[0]
    #NN_sim.append(pred)
    #把model_input送给训练好的模型，模型预测下一个时间点的脑信号，把预测结果存进NN_sim，就是模型自己生成的脑活动。
    NN_sim = np.array(NN_sim)# 把列表转成矩阵，形状：(1200 + steps, node_num)= 时间点 × 脑区
    return corrcoef(NN_sim)#对模型生成的脑信号计算皮尔逊相关

#对应论文中的公式(3)
#对训练好的替代脑模型做 “虚拟扰动”，计算全脑有效连接 EC
def model_EC(model, input_X, target_Y, pert_strength):#model是训练好的替代脑模型，input_X是所有输入（前 3 个时间点），target_Y是真实信号（只用来取脑区数），pert_strength是扰动强度（论文设为 BOLD 标准差的一半）

    "Infer EC by perturbing the surrogate brain"

    node_num = target_Y.shape[1]#脑区数量 N
    steps = int(input_X.shape[1] / node_num)#用前几步预测（固定为 3）
    NPI_EC = np.zeros((node_num, node_num))# 新建一个 N×N 的空矩阵，用来存放最终的有效连接矩阵，EC [i,j] = 脑区 i 对脑区 j 的因果影响强度
    for node in range(node_num):#对每一个脑区，都做一次虚拟扰动
        unperturbed_output = model(torch.tensor(input_X, dtype = torch.float).to(device)).detach().cpu().numpy()#计算无扰动时，模型对所有输入窗口的预测
        #这个代码生成的是一个 (steps × node_num) 的二维矩阵，表示输入窗口内各时间点、各脑区的扰动；然后在最近一个时间点（最后一行），只对当前 node 对应的脑区施加一个扰动。
        #对应图1g
        perturbation = np.zeros((steps, node_num)); perturbation[-1, node] = pert_strength
        perturbed_output = model(torch.tensor(input_X + perturbation.flatten(), dtype = torch.float).to(device)).detach().cpu().numpy()
        NPI_EC[node] = np.mean(perturbed_output - unperturbed_output, axis = 0)#axis=0 指的是 第 0 维 第 0 维在这个矩阵里 = 样本维 实际就是列。公式（3）
    return NPI_EC                                                              #用扰动后 − 扰动前得到响应，对所有时间窗口取平均，正是兴奋型连接，负是抑制型连接，大小是连接强度
# 扰动预测perturbed_output =
# [[1.2, 2.1, 3.0],
# [1.6, 2.7, 3.6]]
# 未扰动预测：unperturbed_output =
# [[1.0, 2.0, 3.0],
# [1.5, 2.5, 3.5]]
# Δ =
# [[0.2, 0.1, 0.0],
#  [0.1, 0.2, 0.1]] mean(axis=0) [0.15, 0.15, 0.05]
#  扰动 node 0，
# 平均来说会让
# node 0 ↑ 0.15 node 1 ↑ 0.15 node 2 ↑ 0.05
# 在 NPI-EC 的计算中，我们通过对某一脑区施加微小扰动，并比较扰动前后模型预测输出的差异，再在所有时间窗口样本上取平均，从而估计该脑区对其他脑区的平均因果影响。

#计算模型的雅可比矩阵
def model_Jacobian(model, input_X, steps):#

    "Calculate the Jacobian matrix of the model"

    node_num = int(input_X.shape[1] / steps)#脑区数量N
    jacobian = np.zeros((node_num, node_num))#为雅可比矩阵创建一个空列表
    model.train()#切换模式为训练模式，打开梯度计算，因为要求导
    for i in range(input_X.shape[0]): #遍历每一个输入窗口
        #自动求模型的导数：输出 对 输入 的偏导数，雅可比矩阵 = 理论上的真实有效连接
        if isinstance(model, ANN_RNN): jacobian += torch.autograd.functional.jacobian(model, torch.tensor(input_X[i], dtype = torch.float).to(device)).cpu().detach().numpy()[0, :, -node_num:]
        #MLP / CNN / VAR 版本的雅可比计算
        else: jacobian += torch.autograd.functional.jacobian(model, torch.tensor(input_X[i], dtype = torch.float).to(device)).cpu().detach().numpy()[:, -node_num:]
    model.eval()#切换为评估模式
    jacobian_EC = jacobian.T / input_X.shape[0]#取平均并转置，让它和 EC 矩阵格式一致：行 = 源区，列 = 靶区
    return jacobian_EC