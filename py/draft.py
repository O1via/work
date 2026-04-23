import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("原始数组:")
print("a:", a, "形状:", a.shape)  # (3,)
print("b:", b, "形状:", b.shape)  # (3,)

# 默认沿轴0堆叠（创建新维度在最前面）
stacked_0 = np.stack((a, b), axis=0)
print("\n沿axis=0堆叠:")
print(stacked_0)
print("形状:", stacked_0.shape)  # (2, 3)

# 沿轴1堆叠（创建新维度在最后面）
stacked_1 = np.stack((a, b), axis=1)
print("\n沿axis=1堆叠:")
print(stacked_1)
print("形状:", stacked_1.shape)  # (3, 2)