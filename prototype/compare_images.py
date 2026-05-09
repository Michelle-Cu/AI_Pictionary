import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time

def load_clip_model():
    """
    載入預訓練的 CLIP 模型與處理器。
    第一次執行時會從網路下載模型檔案（約 600MB），之後就會從本地快取讀取。
    """
    print("正在載入 CLIP 模型 (openai/clip-vit-base-patch32)...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    print("模型載入完成！\n" + "-"*30)
    return model, processor

def calculate_similarity(model, processor, image_path_1, image_path_2):
    """
    計算兩張圖片的餘弦相似度。
    """
    try:
        # 1. 載入並轉換圖片格式 (確保是 RGB，避免 PNG 透明通道造成報錯)
        img1 = Image.open(image_path_1).convert("RGB")
        img2 = Image.open(image_path_2).convert("RGB")
        
        # 2. 將圖片送入處理器進行預處理
        inputs = processor(images=[img1, img2], return_tensors="pt")
        
        # 3. 提取圖片的特徵向量 (Embedding)
        with torch.no_grad(): 
            # 【修改這裡】我們直接抓取底層的 vision_model，確保拿出來的絕對是 Tensor
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            # 將提取出的特徵 (pooler_output) 進行視覺投影，轉換成最終的比對向量
            image_features = model.visual_projection(vision_outputs.pooler_output)
            
        # 4. 將向量進行 L2 正規化 (Normalize)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # 5. 計算兩張圖的相似度
        similarity = torch.dot(image_features[0], image_features[1]).item()
        
        return similarity
        
    except Exception as e:
        print(f"處理圖片時發生錯誤: {e}")
        return None
    
def score_mapper(similarity_score):
    """
    【營隊專用】將原始相似度轉換成 0-100 的遊戲分數。
    (註: CLIP 模型的相似度通常落在 0.5 ~ 0.9 之間，即使是完全不同的圖也很少低於 0.4)
    你可以根據測試結果自行調整這個映射函數的門檻！
    """
    # 假設基準線：相似度 0.5 算 0 分，0.85 以上就算 100 分
    min_sim = 0.50
    max_sim = 0.85
    
    if similarity_score <= min_sim:
        return 0
    elif similarity_score >= max_sim:
        return 100
    else:
        # 線性映射到 0~100 分
        mapped_score = (similarity_score - min_sim) / (max_sim - min_sim) * 100
        return int(mapped_score)

# ==========================================
# 主程式區塊
# ==========================================
if __name__ == "__main__":
    # 填入你要比對的圖片路徑
    # 建議先準備兩張測試圖放在同一個資料夾下
    REFERENCE_IMAGE = "reference_image.png" # 你的標準原圖
    # reference_image.png prompt: A cartoon image with a brown fox in the snow, there are mountains with snow caps in the backdrop, the fox is pouncing on the snow trying to catch a small grey mouse, there are fox footprints in the snow.

    TEAM_IMAGE = "image3.png"           # 小隊生成的圖
    
    # image1.png prompt寫的比較簡略: 74% 相似
    # image2.png prompt寫的比較仔細: 96% 相似
    # image3.png 完全無關的照片: 48% 相似

    # 初始化模型
    model, processor = load_clip_model()
    
    # 計算分數
    print(f"正在比對: '{REFERENCE_IMAGE}' 與 '{TEAM_IMAGE}'")
    start_time = time.time()
    
    sim_score = calculate_similarity(model, processor, REFERENCE_IMAGE, TEAM_IMAGE)
    
    end_time = time.time()
    
    if sim_score is not None:
        game_score = score_mapper(sim_score)
        
        print("\n【比對結果】")
        print(f"AI 原始相似度 (Cosine Similarity): {sim_score:.4f} (滿分 1.0)")
        print(f"換算營隊分數: {game_score} / 100 分")
        print(f"比對耗時: {end_time - start_time:.2f} 秒")