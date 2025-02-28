import torch
import numpy as np

data = [[1, 2], [3, 4]]


def func1():
    """
    数据表示：
        PyTorch张量（Tensor）：是多维数组的数据结构，支持CPU和GPU上的计算。它是专门为深度学习设计的，具有自动微分的能力，能够高效地执行大规模数值运算。
        普通数组（如Python列表）：是一种动态大小的数组，可以存储任意类型的对象。虽然灵活性高，但在数值计算方面效率较低，不适合进行大规模或高性能的数学运算。
        NumPy中的ndarray：是一个同构的多维数组对象，所有元素必须是相同类型。它被设计用于高效的数值计算，并提供了大量的函数库来进行复杂的数学运算。
    功能特性
        PyTorch Tensor
        支持自动微分，这对于训练神经网络模型至关重要。
        可以轻松地在CPU和GPU之间切换，利用硬件加速提高性能。
        提供了丰富的API，便于构建和训练深度学习模型。
    NumPy ndarray
        不支持自动微分，主要用于科学计算和数据分析。
        高效的内存管理和优化过的计算能力使其成为处理大量数值数据的理想选择。
        广泛应用于各种需要数值计算的领域，比如机器学习、图像处理等。
    普通数组（如Python列表）
        主要用于通用编程任务，不特别针对数值计算优化。
        因其灵活性，适用于需要动态大小和多样化的数据类型的场景。
    互操作性
        PyTorch与NumPy之间的转换非常方便：
        可以通过.numpy()方法将一个PyTorch张量转换为NumPy的ndarray。
        使用torch.from_numpy()可以从NumPy的ndarray创建一个PyTorch张量。
        这种紧密的集成使得可以在两个库之间无缝地转移数据，同时利用各自的优势。
        PyTorch张量与普通数组的关系：
        PyTorch张量不能直接由普通数组创建，但可以通过先将普通数组转换为NumPy的ndarray，再从ndarray创建张量的方式间接实现。
    总结
        PyTorch Tensor 和 NumPy ndarray 都是为了高效处理数值数据而设计的，但前者更侧重于深度学习应用，特别是其对自动微分的支持，使得它非常适合构建和训练神经网络模型。
        普通数组（如Python列表） 则更加灵活，适合一般用途的编程任务，但对于大规模数值计算来说，效率不如前两者。
        PyTorch Tensor 和 NumPy ndarray 在很多情况下可以直接互相转换，这使得开发者可以根据具体需求选择最合适的工具，或者结合两者的优势来解决问题。
    """

    x_data = torch.tensor(data)
    print(x_data)
    return x_data


def fun2():
    """
    [[1, 2], [3, 4]]
    [[1 2]
     [3 4]]
    tensor([[1, 2],
            [3, 4]], dtype=torch.int32)
    """
    print(data)
    np_array = np.array(data)
    print(np_array)
    x_np = torch.from_numpy(np_array)
    print(x_np)
    return x_np


def fun3(x_data):
    x_ones = torch.ones_like(x_data)
    print(f"Ones Tensor:\n {x_ones} \n")


def fun4(x_data):
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f"Random Tensor:\n {x_rand} \n")


def fun5():
    shape = (3, 4,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print(f"Random Tensor:\n {rand_tensor} \n")
    print(f"Ones Tensor:\n {ones_tensor} \n")
    print(f"Zeros Tensor:\n {zeros_tensor} \n")


def fun6():
    m = torch.ones(5, 3, dtype=torch.double)
    n = torch.rand_like(m, dtype=torch.float)
    print(m)
    print(n)
    print(m.size())


def fun7():
    # 均匀分布
    # 只会输出0,1之间的小数
    print(torch.rand(5, 3))
    # 如果你需要不同范围的随机数 [a,b)
    a = 3
    b = 5
    print(torch.rand(3, 4) * (b - a) + a)


def fun8():
    # 标准正态分布
    print(torch.randn(5, 3))
    # 离散正态分布
    # mean 均值  std 标准差
    print(torch.normal(mean=2.0, std=3.0, size=(5, 3)))
    # 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点)
    print(torch.linspace(start=5, end=7, steps=7))

def fun9():
    tensor = torch.rand(3,4)
    # print(f"Shape of tensor: {tensor.shape}")
    # print(f"Datatype of tensor: {tensor.dtype}")
    # print(f"Device tensor is stored on: {tensor.device}")
    return tensor

def fun10(tensor):
    print(tensor)
    if torch.cuda.is_available():
        print("1")
        device = torch.device("cuda")
        tensor = tensor.to(device)
        print(tensor)
        print(tensor.device)

if __name__ == '__main__':
    # fun4(func1())
    fun10(fun9())
    print(torch.version.cuda)

