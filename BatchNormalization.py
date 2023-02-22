import numpy as np

class BatchNormalization:
    def __init__(self, gamma, beta, eps=1e-5):
        self.gamma = gamma     # 缩放因子
        self.beta = beta       # 偏移量
        self.eps = eps         # 用于避免除以0的小量
        self.moving_mean = None  # 移动平均值
        self.moving_var = None   # 移动方差
        self.batch_size = None   # 批量大小

    def forward(self, x, train_flg=True):
        """
        前向传递
        """
        if self.moving_mean is None:
            # 在第一次运行前向传递时初始化移动平均值和方差
            N, D = x.shape
            self.moving_mean = np.zeros(D)
            self.moving_var = np.zeros(D)

        if train_flg:
            # 在训练模式下，计算批量均值和方差
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            self.batch_size = x.shape[0]
            # ======== 1. ========
            self.xc = x - mu
            self.std = np.sqrt(var + self.eps)
            self.xn = self.xc / self.std  
            # ======== 2. ========  
            self.gamma_norm = self.gamma * self.xn
            out = self.gamma_norm + self.beta

            # 计算移动平均值和方差
            '''
            移动平均值和方差是由所有批次的均值和方差计算得出的，
            因此它们可以更好地表示整个数据集的分布
            '''
            self.moving_mean = 0.9 * self.moving_mean + 0.1 * mu
            self.moving_var = 0.9 * self.moving_var + 0.1 * var

        else:
            # 在测试模式下，使用移动平均值和方差来标准化输入数据
            xc = x - self.moving_mean
            xn = xc / np.sqrt(self.moving_var + self.eps)
            out = self.gamma * xn + self.beta

        return out

    def backward(self, dout):
        """
        反向传递
        """
        dx = (1.0 / self.batch_size) * self.gamma * self.std * (self.batch_size * dout - np.sum(dout, axis=0)
                - self.xc * (self.std ** -2) * np.sum(dout * self.xc, axis=0))

        dgamma = np.sum(dout * self.xn, axis=0)
        dbeta = np.sum(dout, axis=0)

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx



if __name__ == "__main__":
    # 生成一个随机输入张量
    np.random.seed(0)
    x = np.random.randn(100, 20)
    gamma = np.ones(20)
    beta = np.zeros(20)

    # 创建 BatchNormalization 实例并运行前向传递
    bn = BatchNormalization(gamma, beta)
    out = bn.forward(x)

    # 生成一个随机的输出梯度张量
    dout = np.random.randn(*out.shape)

    # 运行反向传递，并检查梯度是否正确
    dx = bn.backward(dout)
    numeric_dx = np.zeros_like(dx)
    h = 1e-5
    for i in range(x.size):
        x_flat = x.flat[i]

        # 计算数值梯度
        x.flat[i] = x_flat + h
        fxph = bn.forward(x)
        x.flat[i] = x_flat - h
        fxmh = bn.forward(x)
        x.flat[i] = x_flat
        numeric_dx.flat[i] = np.sum(dout * (fxph - fxmh)) / (2 * h)

    # 检查数值梯度是否与解析梯度一致
    print("dx 数值梯度误差：", np.abs(numeric_dx - dx).max())

    # 检查前向传递输出是否与预期一致
    expected_out = gamma * (x - x.mean(axis=0)) / np.sqrt(x.var(axis=0) + bn.eps) + beta
    print("前向传递输出误差：", np.abs(expected_out - out).max())
