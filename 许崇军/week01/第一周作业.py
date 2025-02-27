Numpy练习code：
import numpy as np
   #创建数组，严格的说是一维数组
a = [1,2,3]
b = np.array(a)
b

a =np.array([1,3,5,7,9],dtype=np.float_)  #这个为了看下类型
a

#创建多维数组
dwsz =np.array([0,1],[23,45],[3,56,7887,6],[4545,33])  这个写法是错误的，报错array() takes from 1 to 2 positional arguments but 4 were given
print(dwsz)
#正确写法
dwsz1 =np.array([(1,23,3,5),(23,456,75454,545),(434,0,4,32)])  
dwsz1  #这里输出的是矩阵
dwsz2 =np.array([(1,23,3),(23,456,75454,545),(434,0,4,32)])  
dwsz2  #这里输出的是一维，因为每个元素的个数不一样

a1 =np.zeros((3,4))  #创建一个3行4列的数组，元素都是0
a1
a2 =np.ones((4,2),dtype=int)  #创建一个4行2列的数组，元素都是1
a2  
a3 =np.arange(12,108,12)  #创建一个从12开始，到108结束，步长为12的数组
a3
a4 =np.eye(4)  #创建一个4行4列的单位矩阵
a4
a6 =np.array([(1,2,3,4),(5,6,7,8),(9,10,11,12)])
print(a6)
print(a6[0])  #输出第一行
print(a6[1:])  #输出第二行以后的所有行

遍历
a6 =np.array([(1,2,3,4),(5,6,7,8),(9,10,11,12)])
for i,j,z,y in a6:
    print(i*j*z*y)

a6 =np.array([(1,2,3,4),(5,6,7,8),(9,10,11,12)])
print("ndim:",a6.ndim)  #维度  #因为这里是二维数组，
#所以是2
print("shape:",a6.shape)  #形状
print("size:",a6.size)  #元素个数
print("dtype:",a6.dtype)  #元素类型

pytorch练习code

import torch
import numpy as np
#张量直接从数据中创建
data = [[1,4],[2,8]]
x_data = torch.tensor(data)
print(x_data)

#  张量从numpy数组中创建 
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

#张量从另外一个张量创建,保留of x_data的属性
x_data = torch.ones_like(x_data)
print(f"ones tensor: \n {x_data}\n")

#张量从另外一个张量创建,覆盖 x_data的属性
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"random tensor: \n {x_rand}\n")

#shape是张量的维度.使用随机值创建张量
shape = (2,3,)  #2行3列
rand_tensor = torch.rand(shape) #创建一个随机张量   
ones_tensor = torch.ones(shape)     #创建一个全是1的张量   
zeros_tensor = torch.zeros(shape)   #创建一个全是0的张量    

m = torch.ones(5,3,dtype=torch.double)
n = torch.ones(5,3,dtype=torch.float)
print(f"m: {m}\n")
print(f"n: {n}\n")
print(m.size())  #获取张量的大小
print(m.shape)  #获取张量的形状

tensor = torch.rand(5,3)
print(f"shape of tensor: {tensor.shape}")
print(f"datatype of tensor: {tensor.dtype}")
print(f"device of tensor: {tensor.device}")


#设置张量在GPU上运算
 if torch.cuda.is_available():
     tensor = tensor.to('cuda')

#张量的索引与切片
tensor = torch.ones(4,4)
tensor[:,1] = 0   #将第二列的所有元素设置为0
print(tensor)
print("first row: ",tensor[0])  #获取第一行
print("first column: ",tensor[:,0])  #获取第一列
print("last column: ",tensor[...,-1])  #获取最后一列
print("last column: ",tensor[:,-1])  #获取最后一列


#张量的拼接
t1 = torch.cat([tensor,tensor,tensor],dim=1) #在第二维度上拼接
print(t1)


tensor1 = torch.tensor([[1,13],[2,45]])
# print(tensor1)

# y1 = tensor1 @ tensor1.T  #矩阵乘法
# y2 = tensor1.matmul(tensor1.T)  #矩阵乘法
# y3 = torch.rand_like(tensor1)  #创建一个与tensor相同形状的张量
#这里要在后面加上 dtype=torch.float
# torch.matmul(tensor1,tensor1.T,out=y3)  #矩阵乘法
# print(y1)   
# print(y2)  
# print(y3)

y4 = tensor1 * tensor1  #对应元素相乘
print(y4)
y5 = tensor1.mul(tensor1)  #对应元素相乘
print(y5)
y6 = torch.rand_like(tensor1,dtype=torch.float)  #创建一个与tensor相同形状的张量
torch.mul(tensor1,tensor1,out=y6)  #对应元素相乘        
print(y6)





agg = tensor1.sum()  #求和
print(agg)
agg_item = agg.item()  #将张量的值转换为python数值
print(agg_item)

print(tensor1)
print(tensor1.numpy())  #将张量转换为numpy数组 

tensor2 =tensor1.add(5)  #加法
print(tensor2)

n = np.ones(5)
t =torch.from_numpy(n)  #将numpy数组转换为张量
print(t)
n = t.numpy()  #将张量转换为numpy数组
print(n)
















