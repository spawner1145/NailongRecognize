from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import argparse
import os
import gradio as gr
import base64
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from easy_inference import detect_nailong_from_base64, test_transform, load_model

def get_model_list():
    # 只列举当前目录下的.pth文件
    cwd = os.path.dirname(os.path.abspath(__file__))
    return [f for f in os.listdir(cwd) if f.endswith('.pth')]

def predict(file, model_name):
    if file is None:
        return "请上传图片、GIF或视频文件"
    # 兼容file为文件对象或字符串
    if hasattr(file, "read"):
        file_bytes = file.read()
    elif isinstance(file, str):
        try:
            import base64
            file_bytes = base64.b64decode(file)
        except Exception:
            with open(file, "rb") as f:
                file_bytes = f.read()
    else:
        return "文件格式不支持"
    base64_str = base64.b64encode(file_bytes).decode()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_name)
    model = load_model(model_path, device)
    result = detect_nailong_from_base64(base64_str, model, test_transform, device)
    return "检测到元素(True)" if result else "未检测到元素(False)"

def create_gradio_interface():
    model_list = get_model_list()
    return gr.Interface(
        fn=predict,
        inputs=[
            gr.File(label="上传图片/GIF/视频"),
            gr.Dropdown(choices=model_list, value="nailong.pth", label="选择模型")
        ],
        outputs=gr.Textbox(label="检测结果"),
        title="元素检测",
        description="上传图片、GIF或视频，检测是否包含元素",
        allow_flagging="never",
    )



class DetectRequest(BaseModel):
    base64: str
    model: str = "nailong.pth"

def add_api(app: FastAPI, prefix: str = "/api/v1"):
    @app.post(f"{prefix}/detect_nailong")
    async def detect_nailong_api(req: DetectRequest):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, req.model or "nailong.pth")
        model = load_model(model_path, device)
        result = detect_nailong_from_base64(req.base64, model, test_transform, device)
        return {"result": result}

def parse_args():
    parser = argparse.ArgumentParser(description="Detector")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务host')
    parser.add_argument('--port', type=int, default=7860, help='服务port')
    return parser.parse_args()

def run_server():
    args = parse_args()
    app = FastAPI(title="Detector API", description="元素检测API和WebUI", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    add_api(app, prefix="/api/v1")
    gradio_app = create_gradio_interface()
    app = gr.mount_gradio_app(app, gradio_app, path="")
    print(f"WebUI: http://{args.host}:{args.port}/")
    print(f"API: http://{args.host}:{args.port}/api/v1/detect_nailong")
    print(f"Docs: http://{args.host}:{args.port}/docs  或  /redoc")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    run_server()
