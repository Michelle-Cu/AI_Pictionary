import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr

# 1. 初始化模型
print("正在載入模型...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

def compare_and_score(ref_img, team_img):
    """
    這是 Gradio 的核心函數：
    輸入：兩張 PIL 圖片物件
    輸出：相似度分數與視覺化結果
    """
    if ref_img is None or team_img is None:
        return "請上傳兩張圖片", 0
    
    try:
        # 轉換為 RGB
        img1 = ref_img.convert("RGB")
        img2 = team_img.convert("RGB")
        
        # 提取特徵
        inputs = processor(images=[img1, img2], return_tensors="pt")
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            image_features = model.visual_projection(vision_outputs.pooler_output)
        
        # 正規化與計算相似度
        image_features = F.normalize(image_features, p=2, dim=1)
        similarity = torch.dot(image_features[0], image_features[1]).item()
        
        # 換算為 0-100 分 (根據之前建議的 score_mapper)
        min_sim, max_sim = 0.5, 0.85
        game_score = int(max(0, min(100, (similarity - min_sim) / (max_sim - min_sim) * 100)))
        
        # 回傳評價與分數
        if game_score > 85: rank = "🏆 神還原！"
        elif game_score > 60: rank = "👌 很接近了"
        else: rank = "🤔 好像哪裡怪怪的"
        
        return f"等級：{rank}\nAI 原始相似度：{similarity:.4f}", game_score
    
    except Exception as e:
        return f"發生錯誤：{str(e)}", 0

# 2. 建立 Gradio 介面
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 資訊營：AI 你畫我猜計分系統")
    gr.Markdown("請左邊上傳「標準原圖」，右邊上傳「小隊生成的圖片」")
    
    with gr.Row():
        with gr.Column():
            img_input1 = gr.Image(type="pil", label="標準原圖")
        with gr.Column():
            img_input2 = gr.Image(type="pil", label="小隊生成圖")
            
    btn = gr.Button("🚀 開始比對分數", variant="primary")
    
    with gr.Row():
        output_text = gr.Textbox(label="比對分析")
        output_score = gr.Number(label="最終得分")

    btn.click(fn=compare_and_score, inputs=[img_input1, img_input2], outputs=[output_text, output_score])

# 3. 啟動 (設定 share=True 會產生公用網址)
demo.launch(share=True)