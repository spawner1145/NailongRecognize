import os
import glob
from PIL import Image, ImageSequence
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# 处理GIF图像，提取第一帧并保存为JPG格式
def process_gif(gif_path, output_dir):
    try:
        # 打开GIF文件
        gif = Image.open(gif_path)
        
        # 提取第一帧
        first_frame = next(ImageSequence.Iterator(gif))
        first_frame = first_frame.convert('RGB')
        
        # 保存第一帧为JPG格式
        frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(gif_path))[0]}.jpg")
        first_frame.save(frame_path, 'JPEG')
        print(f"Saved first frame to {frame_path}")
        
        # 删除原始GIF文件
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
            process_gif(image_file, directory)
        elif not image_file.lower().endswith('.jpg'):
            img = Image.open(image_file).convert('RGB')
            jpg_path = os.path.join(directory, f"{os.path.splitext(os.path.basename(image_file))[0]}.jpg")
            img.save(jpg_path, 'JPEG')
            print(f"Converted {image_file} to {jpg_path}")
            os.remove(image_file)
            print(f"Deleted original file: {image_file}")

# 定义数据预处理
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 定义数据集类
class NailongDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root, '*.jpg'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # 转换为RGB图像
        if self.transform:
            image = self.transform(image)
        return image, image_path

# 加载
def load_model(model_path, device):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 修改最后一层为分类层
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

# 预测
def predict_image(image, model, device):
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return pred.item()

# 运行预测
def run_predictions(input_dir, model, transform, device):
    convert_images_to_jpg(input_dir)
    
    dataset = NailongDataset(root=input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for image, image_path in dataloader:
        prediction = predict_image(image.squeeze(0), model, device)
        print(f"Image: {image_path[0]}, Prediction: {'True' if prediction == 1 else 'False'}")

def main():
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # 模型路径
    model_path = './nailong.pth'
    # 输入目录
    input_dir = './input'
    model = load_model(model_path, device)
    run_predictions(input_dir, model, test_transform, device)

if __name__ == '__main__':
    main()