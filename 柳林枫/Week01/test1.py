import numpy as np

#创建narray数组
arr = [1,2,3,4,5]
arr
b=np.array([1,2,3,4,5],float)
b

a = np.array([(1,2,3),(4,5,6),(7,8,9)])
a

a = np.zeros((3,3),dtype=float)
a

a1 = np.ones((3,3))
a1 

a2 = np.arange(1,20,0.5)
a2

a3 = np.eye(3)
a3 #one-hot编码

a4 = np.random.random(5) # 生成5个随机数，作为模型运算参数初始值
a4

a5 = np.random.normal(0,0.3,5) #模型运算参数初始化       
a5  #均值为0，标准差为0.3，生成5个随机数

a6 = np.array([(1, 2),(3, 4),(5, 6)])
print(a6)
print(a6[:,1])
a6[1:]

a7 = np.array([(1,2),(3,4),(5,6)])
# i,j = a7[0]
for i,j in a7:
    print(i,j)

a8 = np.array([(1,2,3),(4,5,6),(7,8,9)])
print("ndim:",a.ndim)#数组的维度
print("shape:",a.shape)#数组的大小
print("size:",a.size)#数组的元素个数
print("dtype:",a.dtype)#数组元素的数据类型

a9 = np.array([(1,2), (3,4)])
#检测元素是否在数组中
print(3 in a)#True
print(5 in a)#False

a10 = np.arange(1,10)
print(a10)
print(a10.shape)  #数组的形状

a10 = a10.reshape(3,3,1)  #维度大小乘积 == 元素个数
print(a10)
print(a10.shape)  #高维矩阵，每个维度都有含义

#加载图像信息
# img.shape [1,3,120,120] 1个样本，3个颜色特征通道，120x120像素

a11 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a11)
a11.transpose() #数组的转置，行变列，列变行

a12 = np.array([(1,2), (3,4), (5,6)])
a12.flatten()   #数组展平，返回一维数组，元素为原数组的元素

a13 = np.array([1,2,3])
print(a13.shape) #输出(3,)
print(a13)
a13 = a13[:,np.newaxis] #增加维度,变成二维数组
print(a13.shape) #输出(3,1)
print(a13)

a = np.ones((3, 3))
b = np.array([(-1,1,1),(-1,1,1),(-1,1,1)])
print(a)
print(b)
c = a+b
d = a-b
e = a*b
f = a/b
print(c)
print(d)
print(e)
print(f)

a = np.array([1,2,1])
b = a.sum() #求和，即所有元素相加
c = a.prod() #求积，即所有元素相乘
print(b)
print(c)

a = np.array([5,3,1])
print("mean:",a.mean()) # 数组的平均值
print("var:", a.var())  # 数组的方差
print("std:", a.std())  # 数组的标准差

a = np.array([1.2, 3.8, 4.9])
print("argmax:", a.argmax())    #数组中最大值的索引，返回的是整数
print("argmin:", a.argmin())    #数组中最小值的索引，返回的是整数
print("ceil:", np.ceil(a))  #向上取整，返回浮点数
print("floor:", np.floor(a))  #向下取整，返回浮点数
print("rint:", np.rint(a))  #四舍五入，返回浮点数

a = np.array([16,31,12,28,22,31,48])
a.sort()    #数组排序，默认升序，返回None，降序排序，a[::-1]
a

# 定义两个简单的矩阵 
m1 = np.array([[1, 2], [3, 4]] , dtype=np.float32)
m2 = np.array([[5, 6], [7, 8]] , dtype=np.float32)
# 使⽤ np.dot 进⾏矩阵乘法 
result_dot = np.dot(m1, m2)
# 使⽤ @ 运算符进⾏矩阵乘法 
result_at = m1 @ m2

print("矩阵 1:")
print(m1)
print("矩阵 2:")
print(m2)
print("使⽤ np.dot 得到的矩阵乘法结果:")
print(result_dot)
print("使⽤ @ 运算符得到的矩阵乘法结果:")
print(result_at)


# 创建⼀个全零矩阵，⽤于存储⼿动推演的结果
# 结果矩阵的⾏数等于 matrix1 的⾏数，列数等于 matrix2 的列数
manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)

# 外层循环：遍历 matrix1 的每⼀⾏
# i 表⽰结果矩阵的⾏索引 
for i in range(m1.shape[0]):
    # 中层循环：遍历 matrix2 的每⼀列
    # j 表⽰结果矩阵的列索引
    for j in range(m2.shape[1]):
        # 初始化当前位置的结果为 0
        manual_result[i, j] = 0
        # 内层循环：计算 matrix1 的第 i ⾏与 matrix2 的第 j 列对应元素的乘积之和
        # k 表⽰参与乘法运算的元素索引
        for k in range(m1.shape[1]):
            # 打印当前正在计算的元素
            print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
            # 将 matrix1 的第 i ⾏第 k 列元素与 matrix2 的第 k ⾏第 j 列元素相乘，并累
            manual_result[i, j] += m1[i, k] * m2[k, j]
        # 打印当前位置计算完成后的结果
        print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")
print("⼿动推演结果:")
print(manual_result)

import numpy as np   #导入numpy库

a = np.array([1,2,3])
b = np.array([4,5,6])
c= a + b
print(c)

a = np.array([(1,2), (2,2), (3,3), (4,4)]) # shape(4,2)
b = np.array([-1,1]) # shape(2)-> shape(1,2) -> shape(4,2) [ -1,1],[-1,1],[-1,1],[-1
d = a + b
print(d)    #必须最后一维的维度相同才可使用
