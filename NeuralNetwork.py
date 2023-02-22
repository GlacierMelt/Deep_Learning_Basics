import numpy as np

class DeepNeuralNetwork:
    """
    DNN 神经网络类
    """

    def __init__(self, layers):
        """
        初始化函数
        :param layers: 各层神经元数量
        """
        np.random.seed(42)

        self.num_layers = len(layers)
        self.layers = layers
        self.parameters = {}

        # 初始化权重矩阵和偏差向量
        for i in range(1, self.num_layers):
            self.parameters[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i - 1]) * 0.01
            self.parameters[f"b{i}"] = np.zeros((self.layers[i], 1))

    def sigmoid(self, Z):
        """
        Sigmoid 激活函数
        """
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        """
        Sigmoid 激活函数的导数
        """
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def forward(self, X):
        """
        正向传播函数
        :param X: 输入数据
        """
        # 输入层
        A_prev = X.T

        # 隐藏层
        for i in range(1, self.num_layers - 1):
            W = self.parameters[f"W{i}"]
            b = self.parameters[f"b{i}"]

            Z = np.dot(W, A_prev) + b
            A = self.sigmoid(Z)

            A_prev = A

        # 输出层
        W = self.parameters[f"W{self.num_layers - 1}"]
        b = self.parameters[f"b{self.num_layers - 1}"]

        Z = np.dot(W, A_prev) + b
        A = self.sigmoid(Z)

        return A, A_prev

    def backward(self, X, y, A1, A2, learning_rate):
        """
        反向传播函数
        :param X: 输入数据
        :param y: 真实标签
        :param A1: 第一层隐藏层的输出
        :param A2: 输出层的输出
        :param learning_rate: 学习率
        """
        m = y.shape[1]

        # 输出层的误差
        dZ2 = A2 - y.T
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # 隐藏层的误差
        dZ1 = np.dot(self.parameters[f"W{self.num_layers - 1}"].T, dZ2) * self.sigmoid_derivative(A1)
        dW1 = np.dot(dZ1, X) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # 更新权重矩阵和偏差向量
        self.parameters[f"W{self.num_layers - 1}"] -= learning_rate * dW2
        self.parameters[f"b{self.num_layers - 1}"] -= learning_rate * db2

        self.parameters[f"W1"] -= learning_rate * dW1
        self.parameters[f"b1"] -= learning_rate * db1

    def train(self, X, y, num_iterations, learning_rate):
        """
        训练函数
        :param X: 输入数据
        :param y: 真实标签
        :param num_iterations: 迭代次数
        :param learning_rate: 学习率
        """
        for i in range(num_iterations):
            # 正向传播
            A2, A1 = self.forward(X)

            # 反向传播
            self.backward(X, y, A1, A2, learning_rate)

            # 计算损失函数
            loss = -np.mean(y * np.log(A2.T) + (1 - y) * np.log(1 - A2.T))

            # 每 100 次迭代输出一次损失函数值
            if i % 100 == 0:
                print(f"迭代次数：{i}，损失函数值：{loss:.6f}")

    def predict(self, X):
        """
        预测函数
        :param X: 输入数据
        """
        A2, _ = self.forward(X)
        predictions = (A2 > 0.5).astype(int)
        return predictions.ravel()

