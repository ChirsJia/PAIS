import torch
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random

def set_seed(seed=1998):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

current_dir = "/nfs/home/svu/e1516749/safe_detector/"

# 切换到该路径
os.chdir(current_dir)
# 配置路径
# 场景信息列表
scenarios = [
    {
        "model_dir": "reparry/EnvironmentSAFE",
        "data_file": "reparry/EnvironmentSAFE/data.pt",
        "output_dir": "reparry/EnvironmentSAFE/results"
    },
    {
        "model_dir": "reparry/ROOMSAFE",
        "data_file": "reparry/ROOMSAFE/data.pt",
        "output_dir": "reparry/ROOMSAFE/results"
    },
    {
        "model_dir": "reparry/Scenario1",
        "data_file": "reparry/Scenario1/data.pt",
        "output_dir": "reparry/Scenario1/results"
    },
    {
        "model_dir": "reparry/MultiID",
        "data_file": "reparry/MultiID/safe_multiID_IDdata.pt",
        "output_dir": "reparry/MultiID/ID_results"
    },
    {
        "model_dir": "reparry/Multiperson",
        "data_file": "reparry/Multiperson/safe_multiperson_multipledata.pt",
        "output_dir": "reparry/Multiperson/ID_results"
    }
]

for scenario in scenarios:
    model_dir = scenario["model_dir"]
    data_file = scenario["data_file"]
    output_dir = scenario["output_dir"]

    print(f"\nProcessing scenario: {model_dir}")
    # 创建结果文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("Loading dataset...")
    data = torch.load(data_file)
    X = data['X']
    y = data['y']
    labels=y
    N = len(X)

    dataset= TensorDataset(X,y)
    total = len(dataset)
    tr = int(0.7*total);
    va=int(0.15*total);
    te=total-tr-va;
    train_set,val_set,test_set=random_split(dataset,[tr,va,te],generator=torch.Generator().manual_seed(1998))
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    # 查找模型文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.jit')]

    # 评估每个 baseline
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        print(f"\nEvaluating model: {model_file}")

        # 加载模型
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()

        all_preds = []
        


        with torch.no_grad():
            for x_batch, y_batch in test_loader:  # 用 DataLoader 避免单样本循环
                x_batch = x_batch.float()  # 确保类型正确
                start_time = time.time()
                output = model(x_batch)
                end_time = time.time()

                batch_preds = torch.argmax(output, dim=1).tolist()
                all_preds.extend(batch_preds)


        elapsed_time = end_time - start_time
        # 对应的真实标签
        labels = []
        for _, y_batch in test_loader:
            labels.extend(y_batch.tolist())
        # 计算指标
        acc = accuracy_score(labels, all_preds)
        precision = precision_score(labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(labels, all_preds, average='macro', zero_division=0)

        # 打印结果
        result_txt = (
            f"Model: {model_file}\n"
            f"Accuracy : {acc:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall   : {recall:.4f}\n"
            f"F1 Score : {f1:.4f}\n"
            f"Time     : {elapsed_time:.2f} sec\n"
        )
        print(result_txt)

        # 写入文件
        output_file = os.path.join(output_dir, f"{os.path.splitext(model_file)[0]}.txt")
        with open(output_file, 'w') as f:
            f.write(result_txt)

    print("✅ 所有模型评估完成，结果已保存")

    # 自动生成 requirements.txt
    with open("requirements.txt", "w") as f:
        f.write(f"torch=={torch.__version__}\n")
        f.write(f"numpy=={np.__version__}\n")
        import sklearn
        f.write(f"scikit-learn=={sklearn.__version__}\n")

    print("requirements.txt 已自动生成，内容如下：")
    with open("requirements.txt") as f:
        print(f.read())
