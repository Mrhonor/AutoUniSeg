import torch
import numpy as np

# Numpy 实现（作为基准）
def calculate_loss_numpy(M_A, M_B, Omega):
    N, a = M_A.shape
    _, b = M_B.shape
    L_r = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            if Omega[i, j] == 1:
                L_r[i, j] = 1 - np.max(M_A[:, i] * M_B[:, j])
            else:
                L_r[i, j] = np.max(M_A[:, i] * M_B[:, j])
    return L_r

# PyTorch 实现（待测试）
def calculate_loss_torch(M_A, M_B, Omega):
    M_A = M_A.unsqueeze(2)  # [N, a, 1]
    M_B = M_B.unsqueeze(1)  # [N, 1, b]
    max_prod = torch.max(M_A * M_B, dim=0)[0]  # 得到 [a, b]
    loss = torch.where(Omega == 1, 1 - max_prod, max_prod)
    return loss

# 测试程序
def test_calculate_loss():
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 随机生成测试数据
    N = 5   # 特征维度
    a = 3   # M_A 的列数
    b = 4   # M_B 的列数

    # 随机生成 Numpy 和 PyTorch 测试输入
    M_A_np = np.random.rand(N, a)
    M_B_np = np.random.rand(N, b)
    Omega_np = np.random.randint(0, 2, (a, b))  # 随机 0/1 矩阵

    # 将 Numpy 输入转换为 PyTorch 输入
    M_A_torch = torch.tensor(M_A_np, dtype=torch.float32)
    M_B_torch = torch.tensor(M_B_np, dtype=torch.float32)
    Omega_torch = torch.tensor(Omega_np, dtype=torch.float32)

    # 使用 Numpy 实现计算结果
    loss_numpy = calculate_loss_numpy(M_A_np, M_B_np, Omega_np)

    # 使用 PyTorch 实现计算结果
    loss_torch = calculate_loss_torch(M_A_torch, M_B_torch, Omega_torch).numpy()

    # 检查 PyTorch 实现的结果和 Numpy 实现的结果是否相同
    assert np.allclose(loss_numpy, loss_torch, atol=1e-6), "Test failed!"

    print("Test passed! PyTorch implementation matches Numpy implementation.")

# 运行测试
test_calculate_loss()
