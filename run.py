import os
import glob
from PIL import Image, ImageSequence
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2

# 将目录中的所有非JPG且非GIF图像文件转换为JPG格式
def convert_images_to_jpg(directory):
    image_files = glob.glob(os.path.join(directory, '*.*'))
    
    for image_file in image_files:
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.gif', '.mp4', '.avi', '.mov')):
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

# 加载模型
def load_model(model_path, device):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 修改最后一层为分类层
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

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

def run_predictions(input_dir, model, transform, device):
    convert_images_to_jpg(input_dir)
    all_files = glob.glob(os.path.join(input_dir, '*.*'))
    for file_path in all_files:
        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            predict_video(file_path, model, transform, device)
        else:
            result = predict_image_or_gif(file_path, model, transform, device)
            print(f"File: {file_path}, Prediction: {'True' if result else 'False'}")


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