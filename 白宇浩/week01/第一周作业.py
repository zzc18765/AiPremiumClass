import numpy as np
#创建ndarray数组
arr = np.array([1,2,3,4,5],int)
arr
print(arr)
#创建ndarray数组 练习
test1 = np.array([111,222,333],float)
test1
print(f'{test1}')
#创建多维数组 
a = np.array([(1,2,3),(4,5,6),(7,8,9)])
a
print(f'{a}')
#创建多维数组 练习
test1 = np.array([(1,2,3,4),(2,3,4,5),(3,4,5,6),(4,5,6,7)])
test1
print(f'{test1}')
#特殊数据 空数组
a = np.zeros((3,3),dtype = int)
#a = np.zeros((3,3),dtype = np.float32)
a
#上面例子中 zero  表示值域全部为0  
#          (3,3) 表示维度 即3行3列
#          dtype 表示数值类型  可以用np 中具体的长度
#继续上面 讲zeros 替换为 ones 可得全部为1的矩阵   但没有twos之类
a = np.ones((3,3),dtype = np.int64)
a
#创建等差数列
a = np.arange(1,5,0.5)
a 
#创建等差数列 练习
#np.arange 函数第一位表示起始，第二位表示终止，第三位表示差额 函数是左闭右开结构的区间
test2 = np.arange(10,20,2.5)
test2 
#特殊矩阵 单位矩阵 矩阵乘法中的 1  
# one-hot 编码 每行第一个都是1
a = np.eye(3)
a
#特殊矩阵 单位矩阵 矩阵乘法中的 1  练习
# one-hot 编码 每行第一个都是1
test2 = np.eye(5)
test2
#[0-1)之间的随机数 5表示生成数量 作为模型运算的初始参数
np.random.random(5)
np.random.random(10)
#[0-1)之间的随机数 5表示生成数量 作为模型运算的初始参数 练习改为10
np.random.random(10)
#正态分布的随机数
#在正态分布中，参数 μ 和 σ 分别代表分布的均值和标准差）。
mu,sigma =0,0.1
np.random.normal(mu,sigma,5)
#正态分布的随机数 练习修改了mu,sigma 的值
#在正态分布中，参数 μ 和 σ 分别代表分布的均值和标准差）。
mu,sigma =10,8
np.random.normal(mu,sigma,6)
#np中数组的的访问
a6 = np.array([(1,2),(3,4),(5,6)])
a6
print(a6)
print('-------------------------')
print(a6[:,1])
print('-------------------------')
print(a6[1:])
#np中数组的的访问  练习
test3 = np.array([(1,2,3),(4,5,6),(7,8,9)])
test3
# ：指 行没有范围 1 指 列的范围为1
print(test3)
print('-------------------------')
print(test3[:,1])
print('-------------------------')
print(test3[1,1])
print('-------------------------')
#输出135
#输出147
print(test3[:,0])
print('-------------------------')
print(test3[0,:])
print(test3[:,2])
print(test3[0,0],test3[1,1],test3[0,2])
#数组中的遍历
a6 = np.array([(1,2),(3,4),(5,6)])
a6
for i in a6:
    print(i)
  #多为数组中的遍历 + 作业
a6 = np.array([(1,2),(3,4),(5,6)])
a6
for i,j in a6:
    print(i*j)
    print('-------------------------')
    print(i+j)
#多为数组中的遍历 + 作业
a7 = np.array([(1,2),(3,4),(5,6)])
a7
i,j = a7[0]
print(i,j)
print('-------------------------')
for i,j in a7:
    print(i,j,i*j)
print('-------------------------')
#数组的属性
aa = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(aa)
print('-------------------------')
#ndim 查看维度
print("ndim:",aa.ndim)
print('-------------------------')
#ndim 查看形状
print("shape:",aa.shape)
print('-------------------------')
#ndim 查看总数
print("size:",aa.size)
print('-------------------------')
#ndim 查看类型
print("dtype:",aa.dtype)
print('-------------------------')
#对数组进行计算时，进行提前了解数组内部
#数组的基本操作
#1.是否存在
aa = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(aa)
print('-------------------------')
print(3 in aa)
print('-------------------------')
print(77 in aa)
print('-------------------------')
print(3 not in aa)
print('-------------------------')
print(77 not in aa)
print('-------------------------')
#用if进行判断
if 3 in aa:
    print("3 in aa")
else: 
    print("3 not in aa")  
#用if进行判断
if 77 in aa:
    print("3 in aa")
else: 
    print("3 not in aa")     
#直接输出
print(99 in aa)
print('-------------------------')
print(1 in aa)
print('-------------------------')
#print((2,3) in aa) 不符合维度 会报错
print((1,2,3) in aa)  #这样可以
#reshape 把高维数组重新编排
a = np.zeros([2,3,4])
print(a)
print('-------------------------')
a.reshape(24)
print('-------------------------')
a.reshape(2,12)
#reshape 把高维数组重新编排
a = np.arange(1,10)
print(a)
print('-------------------------')
a.reshape(9)
print('-------------------------')
a = a.reshape(3,3)
print('-------------------------')
print(a)
print('-------------------------')
print(a.shape)
#reshape 维度大小的乘积 == 元素个数  
#实际使用时 每个维度都有含义
#加载图像数据
#img.shape[1，3，120，120] 1 图像数量 3 颜色特征通道 120高 *120宽 
#transpose 把数组转置 
a = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(a)
print('-------------------------')
a.transpose()
#transpose 把数组转置  a.T
a = np.array([(1,2),(3,4),(5,6)])
print(a)
print('-------------------------')
#a.transpose()
a.T
#flatten 把数组转为一维的向量
a = np.array([(1,2),(3,4),(5,6)])
print(a)
print('-------------------------')
a.flatten()
#flatten 把数组转为一维的向量  练习
a = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(a)
print('-------------------------')
#a = a.transpose()
a = a.flatten()
print(a)
#newaxis 增加维度
a = np.array([(1,2),(3,4),(5,6)])
print(a)
print(a.shape)
print('-------------------------')
#a = a[:,np.newaxis,:]
print('-------------------------')
print(a)
print(a.shape)
a = a[:,:,np.newaxis]
print('-------------------------')
print(a)
print(a.shape)
#数学计算 数组 对 数组 
a = np.ones((2,2))
b = np.array([(-1,1),(-1,1)])
print(a)
print('---------------------------')
print(b)
print('---------------------------')
print(a+b)
print('---------------------------')
print(a*b)
#数学计算 数组 对 内
#sum 求和  prod 求积  mean 平均 var 方差 std 标准差 max 最大 min 最小
a = np.array([(1,2,3,4)])
print(a.sum())
print('---------------------------')
print(a.prod())
print('---------------------------')
print(a.mean())
print('---------------------------')
print(a.var())
print('---------------------------')
print(a.std())
print('---------------------------')
print(a.max())
print('---------------------------')
print(a.min())
#数组中的索引
#argmax 最大值的索引  argmin 最小值的索引  ceil 向上取整 floor 向下取整 rint 四舍五入 
a = np.array([(1.2,3.8,4.9)])
print(f'argmax',a.argmax())
print('---------------------------')
print(f'argmin',a.argmin())
print('---------------------------')
print(f'ceil',np.ceil(a))
print('---------------------------')
print(f'floor',np.floor(a))
print('---------------------------')
print(f'rint',np.rint(a))
#排序
#sort
a = np.array([16,31,12,28,22,31,48])
print(a)
print('---------------------------')
a.sort()
print(a)
#数组，矩阵，维度 
#数组，矩阵: 数据组织的结构形式
#维度：地址 ，把现实中的实际落进去
#numpy线性代数 在numpy.linalg中
# 矩阵matrix / array  都可以用array更灵活
# 矩阵乘法  两个矩阵相同的 直接乘
a = np.array([[1,2],[4,5]]) 
b = np.array([[4,5],[7,8]]) 
#一维计算的是这两个数组对应下标元素的乘积和
#计算的是两个数组的矩阵乘积
#对于多维数组，它的通⽤计算公式如下，即结果数组中的每个元素都是：数组 a 的最
#后⼀维上的所有元素与数组 b 的倒数第⼆位上的所有元素的乘积和：以下为公式
#dot(a, b)[i,j,k,m]= sum(a[i,j,:] * b[k,:,m])
print(a)
print()
print(b)
print()
a2=np.dot(a,b)
print(a2)
#即将第一行的数组中的元素与第二行的数组中对应的元素相乘做和
#1：[1,2] 同 [4，7]进行第一次计算 1*4 + 2*7 == 4 + 14 = 18
#2：[1,2] 同 [5，8]进行第二次计算 1*5 + 2*8 == 5 + 16 = 21
#3：[4,5] 同 [4，7]进行第三次计算 4*4 + 5*7 == 16 + 35 = 51
#4：[4,5] 同 [5，8]进行第三次计算 4*5 + 5*8 == 20 + 40 = 60
#其他计算方式同样支持 eig 特征值，特征向量 inv 逆 qr QR分解 svd 奇异值分解 solve 解线性方程Ax = b 
#可以对np数据进行文件保存
#二进制 np.save np.load  /npy后缀格式
#文本 np.loadtxt np.savetxt 
np.save('a2.npy',a2)
np.load('a2.npy')
import numpy as np
# 定义两个简单的矩阵
m1 = np.array([[1,2], [3,4]] , dtype=np.float32)
m2 = np.array([[5,6], [7,8]] , dtype=np.float32)
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
    manual_result[i,j] = 0
 # 内层循环：计算 matrix1 的第 i ⾏与 matrix2 的第 j 列对应元素的乘积之和
 # k 表⽰参与乘法运算的元素索引
    for k in range(m1.shape[1]):
 # 打印当前正在计算的元素
      print(f"{m1[i,k]} * {m2[k,j]} = {m1[i,k] * m2[k,j]}")
 # 将 matrix1 的第 i ⾏第 k 列元素与 matrix2 的第 k ⾏第 j 列元素相乘，并累
      manual_result[i,j] += m1[i,k] * m2[k,j]
 # 打印当前位置计算完成后的结果
      print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")
print("⼿动推演结果:")
print(manual_result)
#练习
a = np.array([[1,2,3],[4,5,6],[7,8,9]]) 
b = np.array([[1,2,3],[4,5,6],[7,8,9]]) 
print(a)
print()
print(b)
print()
print(a @ b)
print()
#这里的81 是456 同258 进行计算 8 + 25 + 48 = 81 得到的
#np的广播机制 重要
#计算两个数组，维度不同的时候，将低维进行高维的自动扩充
a = np.array([1,2,3])
b = np.array([1,2,3])
a+b
#不同的时候
#计算要求左边矩阵的列= 右边矩阵的行 
#这里的a为4行两列，b只有一个维度 因为右边的行必须= 左边的列 所以这里b会被按列进行定义计算
a = np.array([(1,2),(2,3),(3,4),(4,5)])
b = np.array([1,2])
print(a.shape)
print()
print(b.shape)
print()
print(a)
print()
print(b)
print()
print(a+b)
print()
print(a @ b)
print()
#  1*1 + 2*2 = 5
#  2*1 + 3*2 = 8
#  3*1 + 4*2 = 11
#  4*1 + 5*2 = 14
a = np.array([(1,2,3),(3,4,5),(4,5,6)])
b = np.array([1,2,3])
print(a)
print(a.shape)
print()
print(b)
print(b.shape)
print()
c = a @ b 
print(c)
print(c.shape)









