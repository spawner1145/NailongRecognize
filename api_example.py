import httpx
import asyncio
import base64

async def main():
    with open("你的图片.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    data = {
        "base64": img_b64,
        "model": "nailong.pth"
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post("http://127.0.0.1:7860/api/v1/detect_nailong", json=data)
        result = resp.json().get("result")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
