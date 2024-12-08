# 唐死我了，这是一个奶龙鉴别模型及其训练代码和数据集，byd一边做一边笑
### 基于resnet50训练
### release里获取最新模型和数据集
### 理论上这个库可以进行所有二分类模型训练，只要自己改掉train和test以及相应negative文件夹内的图片再运行train.py就可以做自己想要的图片分类（检测图片中是否有某元素，输出是或否）
### 支持gif和视频，原理遍历所有帧，有一张有就输出有，否则输出无
### 我参与开发的一个机器人已加入奶龙检测功能https://github.com/avilliai/Manyana.git
### 上面的机器人如果要更换模型打开plugins/nailong11文件夹把里面的nailong.pth替换掉
### 奶龙识别nonebot机器人插件链接，作者445，https://github.com/Refound-445/nonebot-plugin-nailongremove?tab=readme-ov-file
