import torch
x_data = torch.tensor(data)
print(x_data)

import numpy as np
np_array = np.array(data)   #中间变量np_array是numpy数组，data是python列表，需要先将data转换为numpy数组，才能进行后续的处理
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) # 保留of x_data的属性
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # 覆盖 x_data的数据类型
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)  #shape是张量维度的元组。在下⾯的函数中，它决定了输出张量的维度。
rand_tensor = torch.rand(shape) #生成随机张量
ones_tensor = torch.ones(shape) #生成全1张量
zeros_tensor = torch.zeros(shape) #生成全0张量
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 基于现有tensor构建，但使⽤新值填充
m = torch.ones(5,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)

# 获取tensor的⼤⼩
print(m.size()) 
torch.Size([5,3])

# 均匀分布 
torch.rand(5,3)

# 标准正态分布 
torch.randn(5,3)

# 离散正态分布 
torch.normal(mean=.0,std=1.0,size=(5,3))

# 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点) 
torch.linspace(start=1,end=10,steps=20)

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}") #输出形状
print(f"Datatype of tensor: {tensor.dtype}")    #输出数据类型
print(f"Device tensor is stored on: {tensor.device}")    #输出设备，如果是CPU则输出cpu，如果是GPU则输出cuda:0

# 设置张量在GPU上运算
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1]) 
tensor[:,1] = 0
print(tensor)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 计算两个张量之间矩阵乘法的⼏种⽅式。 y1, y2, y3 最后的值是⼀样的 dot
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
# 计算张量逐元素相乘的⼏种⽅法。 z1, z2, z3 最后的值是⼀样的。
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum()  #数组求和，返回一个标量
agg_item = agg.item()  #将agg转换为标量
print(agg_item, type(agg_item))  #打印agg_item的值和类型

print(tensor, "\n") #输出张量，tensor([[1, 2, 3], [4, 5, 6]])，shape为(2,3)，数据类型为int64
tensor.add_(5)  #tensor.add(5)也可，tensor加5，结果为tensor([[6, 7, 8], [9, 10, 11]])
print(tensor)   #输出tensor([[6, 7, 8], [9, 10, 11]])，shape为(2,3)，数据类型为int64

t = torch.ones(5)   #数组初始化，  5表示数组的长度
print(f"t: {t}")    #输出数组，  [1. 1. 1. 1. 1.]
n = t.numpy()   #将torch数组转化为numpy数组， 并赋值给n
print(f"n: {n}")    #输出numpy数组，  [1. 1. 1. 1. 1.]

#张量的变更也影响了原变量的值   
t.add_(1)   
print(f"t: {t}")
print(f"n: {n}")

#Numpy数组的变化也会影响到Torch张量的值
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

import torch
from torchviz import make_dot
# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(10, 10,requires_grad=True)  # 10x10 矩阵, requires_grad=True 表示对该矩阵求导
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)
# 计算 x^T * A + b * x + c
result = torch.matmul(A, x.T) + torch.matmul(b, x) + c  # matmul()函数代表矩阵乘法, x.T 代表矩阵转置,"b,x"表示向量乘法
# ⽣成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)
