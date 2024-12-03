import os
import glob
from PIL import Image, ImageSequence
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np
import cv2

# 处理GIF图像，提取每一帧并保存为JPG格式
def process_gif(gif_path, output_dir):
    try:
        gif = Image.open(gif_path)
        frame_num = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for frame in ImageSequence.Iterator(gif):
            rgb_frame = frame.convert('RGB')
            frame_filename = f"{os.path.splitext(os.path.basename(gif_path))[0]}_frame_{frame_num:03d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            rgb_frame.save(frame_path, 'JPEG')
            print(f"Saved frame {frame_num} to {frame_path}")
            frame_num += 1
        gif.close()
        os.remove(gif_path)
        print(f"Deleted original GIF file: {gif_path}")
    except PermissionError as e:
        print(f"Failed to delete {gif_path}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {gif_path}: {e}")

# 将目录中的所有图像文件转换为JPG格式
def convert_images_to_jpg(directory):
    image_files = glob.glob(os.path.join(directory, '*.*'))
    
    for image_file in image_files:
        if image_file.lower().endswith('.gif'):
            if directory == './input':  # 如果目录是'./input'，则不调用process_gif
                continue  # 跳过.gif文件
            else:
                process_gif(image_file, directory)
        elif not image_file.lower().endswith(('.jpg', '.jpeg', '.gif', '.mp4', '.avi', '.mov')):
            img = Image.open(image_file).convert('RGB')
            jpg_path = os.path.join(directory, f"{os.path.splitext(os.path.basename(image_file))[0]}.jpg")
            img.save(jpg_path, 'JPEG')
            print(f"Converted {image_file} to {jpg_path}")
            os.remove(image_file)
            print(f"Deleted original file: {image_file}")

class NailongDataset(Dataset):
    def __init__(self, positive_root, negative_root, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for filename in glob.glob(os.path.join(positive_root, '*.jpg')):
            self.image_paths.append(filename)
            self.labels.append(1)  # 正样本标签为1

        for filename in glob.glob(os.path.join(negative_root, '*.jpg')):
            self.image_paths.append(filename)
            self.labels.append(0)  # 负样本标签为0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # 转换为RGB图像
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# 数据增强
data_augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 减少翻转概率
    transforms.RandomRotation(10),  # 保持旋转角度不变
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 减少颜色抖动程度
    transforms.RandomCrop(224),
    transforms.ToTensor()
])

# 创建数据集和数据加载器
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 增大尺寸
    data_augmentation_transform
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

convert_images_to_jpg('./train_positive')
convert_images_to_jpg('./train_negative')
convert_images_to_jpg('./test')
convert_images_to_jpg('./negative_test')
convert_images_to_jpg('./input')

# 创建数据集
train_dataset = NailongDataset(positive_root='./train_positive', negative_root='./train_negative', transform=train_transform)
test_dataset = NailongDataset(positive_root='./test', negative_root='./negative_test', transform=test_transform)

# 平衡数据集
def balance_dataset(dataset):
    X = [i for i in range(len(dataset))]
    y = [dataset[i][1] for i in range(len(dataset))]
    
    # 打印标签分布
    unique, counts = np.unique(y, return_counts=True)
    print(f"Label distribution before balancing: {dict(zip(unique, counts))}")
    
    # 将X转换为二维数组
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    
    # 使用SMOTE和随机欠采样
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    pipeline = Pipeline(steps=[('o', smote), ('u', rus)])
    
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    
    # 打印标签分布
    unique, counts = np.unique(y_resampled, return_counts=True)
    print(f"Label distribution after balancing: {dict(zip(unique, counts))}")
    
    # 创建平衡的数据集
    balanced_dataset = torch.utils.data.Subset(dataset, X_resampled.flatten())
    return balanced_dataset

# 平衡训练集
balanced_train_dataset = balance_dataset(train_dataset)

# 创建数据加载器
train_loader = DataLoader(balanced_train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 使用预训练的ResNet50模型
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 修改最后一层为分类层
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # 降低学习率并添加L2正则化
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to('cuda' if torch.cuda.is_available() else 'cpu'))  # 加权损失函数

# 多阶段训练
def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    model.train()
    best_val_loss = float('inf')
    patience = 5
    no_improvement_count = 0

    for stage in range(2):
        if stage == 1:
            # 二阶段降低学习率
            optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # 验证集损失
            val_loss = evaluate_model(model, val_loader, device)
            print(f'Stage [{stage+1}/2], Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}')

            # 早停法
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f'Early stopping at Stage [{stage+1}/2], Epoch [{epoch+1}/{epochs}]')
                    break

# 测试模型
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    return running_loss / len(test_loader)  # 验证集损失

# 训练
train_model(model, train_loader, test_loader, epochs=10, optimizer=optimizer, criterion=criterion, device='cuda' if torch.cuda.is_available() else 'cpu')

# 评估
evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')

# 保存
model_path = './nailong.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

def predict_frame(frame, model, transform, device):
    """ 对单帧图像进行预测 """
    model.eval()
    frame = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(frame)
        _, pred = torch.max(output, 1)
    return pred.item() == 1  # 返回是否为奶龙元素

def predict_image_or_gif(file_path, model, transform, device):
    """ 对图像或GIF文件进行预测 """
    model.eval()
    if file_path.lower().endswith('.gif'):
        gif = Image.open(file_path)
        for frame in ImageSequence.Iterator(gif):
            frame = frame.convert('RGB')
            if predict_frame(frame, model, transform, device):
                return True  # 发现奶龙元素
        return False  # 没有发现奶龙元素
    else:
        image = Image.open(file_path).convert('RGB')
        return predict_frame(image, model, transform, device)  # 返回是否为奶龙元素

def predict_video(video_path, model, transform, device):
    """ 对视频文件的每一帧进行预测 """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    found = False
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        if predict_frame(pil_image, model, transform, device):
            found = True
            print(f"Video: {video_path}, Frame {frame_count}: True")  # 发现奶龙元素
        frame_count += 1
    cap.release()
    if not found:
        print(f"Video: {video_path}, Prediction: False")  # 没有发现奶龙元素
    return found

def test_input_directory(input_dir, model, transform, device):
    convert_images_to_jpg(input_dir)
    all_files = glob.glob(os.path.join(input_dir, '*.*'))
    for file_path in all_files:
        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            predict_video(file_path, model, transform, device)
        else:
            result = predict_image_or_gif(file_path, model, transform, device)
            print(f"File: {file_path}, Prediction: {'True' if result else 'False'}")

# 输入目录
input_dir = './input'

# 设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 调用函数测试目录中的文件
test_input_directory(input_dir, model, test_transform, device)