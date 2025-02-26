# 基本操作

# 数组的形状和维度
np_array = np.array([[[2, 4],[3, 6],[5, 7]], [[1, 3],[5, 7],[6,8]]])
print("数组的形状：", np_array.shape)
print("数组的维度：", np_array.ndim)
print("数组的元素个数：", np_array.size)
print("数组的元素类型：", np_array.dtype)

# 数组的索引和切片
print("数组的第一个元素：", np_array[0, 0, 0])
print("数组的第一行：", np_array[0, 0, :])
print("数组的最后一维元素：", np_array[..., -1])

# 判定元素是否在数组中
print("元素 5 是否在数组中：", 5 in np_array)
print("元素 [2,4] 是否在数组中：", (2,4) in np_array)
# print("元素 [2,4,8] 是否在数组中：", (2,4,8) in np_array)

# 遍历
for i,j,k in np_array:
    print(i,j,k)
    
# 基础运算
np_array1 = np.array([[1, 2], [3, 4]])
np_array2 = np.array([[5, 6], [7, 8]])
print("数组相加：", np_array1 + np_array2)
print("数组相减：", np_array1 - np_array2)
print("数组相乘-元素级乘法-对应位置相乘：", np_array1 * np_array2)
print("数组相乘-矩阵乘法-内积运算1：", np_array1 @ np_array2)
print("数组相乘-矩阵乘法-内积运算2：", np.dot(np_array1, np_array2))
print("数组相除：", np_array1 / np_array2)
print("数组的转置1：", np_array1.T)
print("数组的转置2：", np_array1.transpose())

np_array3 = np.array([1.5, 2.6, 3.4])
print("求和：", np.sum(np_array3))
print("乘积：", np.prod(np_array3))
print("均值：", np.mean(np_array3))
print("标准差：", np.std(np_array3))
print("方差：", np.var(np_array3))
print("最大值：", np.max(np_array3))
print("最小值：", np.min(np_array3))
print("求和：", np.sum(np_array3))
print("最大值下标索引：", np.argmax(np_array3))
print("最小值下标索引：", np.argmin(np_array3))
print("向上取整：", np.ceil(np_array3))
print("向下取整：", np.floor(np_array3))
print("四舍五入1：", np.round(np_array3))
print("四舍五入2：", np.rint(np_array3))
print("取绝对值：", np.abs(np_array3))
print("排序:", np.sort(np_array3))

