import numpy as np

def generate_random_points_on_sphere(n):
    """
    随机生成四维单位球面上的 n 个点。
    
    Args:
    n (int): 要生成的点的数量。
    
    Returns:
    numpy.ndarray: 形状为 (n, 4) 的四维单位球面上的点的数组。
    """
    print("即将随机生成",n,"个点。。。。。。")
    points = np.random.randn(n, 4)  # 从标准正态分布中生成随机数
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit_sphere_points = points / norms  # 将点归一化到单位球面上
    return unit_sphere_points

def format_points(points):
    """
    将点格式化为包含方括号和逗号的字符串，并控制小数位数不超过3位。
    
    Args:
    points (numpy.ndarray): 四维单位球面上的点的数组。
    
    Returns:
    str: 格式化后的点的字符串。
    """
    formatted_points = ["[" + ", ".join([f"{coord:.3f}" for coord in point]) + "]" for point in points]
    return "[" + ", ".join(formatted_points) + "]"

def main():
    # 指定要生成的点的数量
    n = 23 # 你可以调整这个值
    random_points = generate_random_points_on_sphere(n)
    
    np.save("/Users/rhy/Desktop/GNN/newtry/results/generate_spheres.npy", random_points)
    print("点已保存到 generate_spheres.npy 文件中")

if __name__ == "__main__":
    main()
