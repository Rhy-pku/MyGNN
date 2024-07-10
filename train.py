import torch
import torch.optim as optim
from model import GNN  # 请确保model.py文件中定义了GNN模型
from dataset import SphereDataset  # 请确保dataset.py文件中定义了SphereDataset
from utils import  save_spheres, compute_edges, cos_losses, normalize_vectors  # 请确保utils.py文件中定义了这些函数
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NumpyDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.load(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)
    
def update_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def add_perturbation(data, indices, magnitude=0.01):
    perturbation = np.random.randn(len(indices), data.shape[1]) * magnitude
    data[indices] += perturbation
    return data

def add_all_perturbation(data, magnitude=0.01):
    perturbation = np.random.randn(*data.shape) * magnitude
    return data + perturbation

def check_cos_similarity(points, threshold=0.50):
    n = len(points)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    normalized_points = points / norms
    cos_similarity_matrix = np.dot(normalized_points, normalized_points.T)
    mask = np.triu(np.ones((n, n)), k=1)  # 上三角矩阵掩码，排除对角线和下三角部分
    indices_to_perturb = np.argwhere(cos_similarity_matrix > threshold)
    mask_indices = mask[indices_to_perturb[:, 0], indices_to_perturb[:, 1]].astype(bool)
    indices_to_perturb = indices_to_perturb[mask_indices]
    unique_indices_to_perturb = np.unique(indices_to_perturb[:, 0])  # 只选择第一个索引，避免重复
    return len(unique_indices_to_perturb), list(unique_indices_to_perturb)

        # tail -f output.log
        # caffeinate -i nohup python -u "/Users/rhy/Desktop/GNN/newtry/src/train.py" > output.log 2>&1 &
        # ps aux | grep python
def check_for_nan(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"Warning: NaN detected in {name}!")
        return True
    return False


def train_model(file_path, best_loss=1e20, epochs=3600000, lr=0.005, lambda_cos=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用自定义的数据集类从 .npy 文件中加载数据
    dataset = NumpyDataset("/Users/rhy/Desktop/GNN/newtry/results/generate_spheres.npy")
    # dataset = SphereDataset()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)  # 保证点的顺序不变
    
    model = GNN(input_dim=4, hidden_dim=64, output_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_saved = False
    cnt = 0
    cntcnt0 = 0
    losschanged = False
    best_loss_zero = False
    boomcnt = 0
    yeahhh = False
    for epoch in range(epochs):
        if cnt == 0:
            cntcnt0 += 1
        else:
            cntcnt0 = 0
        model.train()
        total_loss = 0.0


        for batch in dataloader:
            optimizer.zero_grad()
            batch_points = batch.to(device)
            threshold = 0.15
            edge_index = compute_edges(batch_points.detach().cpu().numpy(), threshold)
            output = model(batch_points, torch.tensor(edge_index, dtype=torch.long).to(device))
            
            # 检查模型输出
            if check_for_nan(output, "model output"):
                return best_loss, final_spheres

            output = normalize_vectors(output)
            
            # 检查归一化后的输出
            if check_for_nan(output, "normalized output"):
                return best_loss, final_spheres

            cos_loss_val = cos_losses(output)
            
            # 检查损失值
            if check_for_nan(cos_loss_val, "cosine loss"):
                return best_loss, final_spheres

            loss = cos_loss_val
            loss.backward()
            
            # 检查梯度
            for name, param in model.named_parameters():
                if check_for_nan(param.grad, f"{name} gradient"):
                    return best_loss, final_spheres

            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / len(dataloader)
        final_spheres = normalize_vectors(output).detach().cpu().numpy()

        if check_cos_similarity(final_spheres,0.5)[0] == 0 and not yeahhh:
            yeahhh = True
            save_spheres(final_spheres, "/Users/rhy/Desktop/GNN/newtry/results/amazing_spheres.npy")
            print("###########################################################")
            print("###########################################################")
            print("YEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEAH")
            print("###########################################################")
            print("###########################################################")
        if best_loss < 10e-6:
            save_spheres(final_spheres, "/Users/rhy/Desktop/GNN/newtry/results/alright_spheres.npy")
        if average_loss < best_loss:
            cnt = 0
            best_loss = average_loss
            save_spheres(final_spheres, "/Users/rhy/Desktop/GNN/newtry/results/tempbestbest_spheres.npy")
            if not batch_saved:
                save_spheres(batch_points.detach().cpu().numpy(), "/Users/rhy/Desktop/GNN/newtry/data/tempbest_data.npy")
                batch_saved = True
            print("已保存目前最优解,best_loss = ", best_loss)
        else:
            cnt += 1

        if cntcnt0 > 10:
            cntcnt0 = 0
            lr *= 1.05       
            losschanged = False  
            update_learning_rate(optimizer, lr)
            print(f"lr changes to {lr}")
        if cnt > 80:
            if not losschanged:
                lr *= 0.5   
                losschanged = True       
            update_learning_rate(optimizer, lr)
            print(f"lr changes to {lr}")
        
        cnt_cos_sim, indices_to_perturb = check_cos_similarity(final_spheres, 0.9)
        if cnt > 50 and cnt_cos_sim > 0 or cnt > 160:
            print("###########################################################")
            print("###########################################################")
            print("BOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOM")
            print("###########################################################")
            print("###########################################################")
            boomcnt += 0.5
            cnt = 0
            if cnt_cos_sim > 0:
                print("发现重合")
                perturbed_data = add_perturbation(dataset.data, indices_to_perturb, magnitude=100000)
            else:
                perturbed_data = add_all_perturbation(dataset.data, magnitude=best_loss*(10**(-7+boomcnt)))
            dataset.data = perturbed_data
            lr = 1e-3
            best_loss = 1e20

        save_spheres(final_spheres, "/Users/rhy/Desktop/GNN/newtry/results/current_spheres.npy")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss}, best_loss = {best_loss}, cnt = {cnt}, lr = {lr},连续{cntcnt0}次找到最优解")
    
    final_spheres = normalize_vectors(output).detach().cpu().numpy()
    save_spheres(final_spheres, "/Users/rhy/Desktop/GNN/newtry/results/final_spheres.npy")
    print("Final spheres saved at : temp_spheres.npy")
    return best_loss, final_spheres

if __name__ == "__main__":
    file_path = ""
    best_loss, final_spheres = train_model(file_path)
    print("Best loss:", best_loss)
