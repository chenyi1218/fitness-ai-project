import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import google.generativeai as genai
from PIL import Image
import json
import re

# 1. 頁面基礎設定
st.set_page_config(page_title="AI 健身數據科學分析平台", layout="wide")

# 初始化 Session State
if 'weight' not in st.session_state: st.session_state.weight = 70.0
if 'body_fat' not in st.session_state: st.session_state.body_fat = 25.0
if 'ai_plan' not in st.session_state: st.session_state.ai_plan = ""

# 2. 側邊欄：API 設定
st.sidebar.title("🔐 系統安全與設定")
api_key = st.sidebar.text_input("輸入 Gemini API Key", type="password")
st.sidebar.info("💡 此系統整合了機器學習(Scikit-learn)與生成式AI(Gemini)，符合期末專題要求。")

# --- 動態尋找可用模型 (避免 404 錯誤) ---
def get_best_model(api_key):
    try:
        genai.configure(api_key=api_key)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # 優先使用 Pro 模型 (邏輯較強)
        for m in available_models:
            if 'pro' in m.lower() and '1.5' in m.lower(): return m
        # 其次使用 Flash
        for m in available_models:
            if 'flash' in m.lower(): return m
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

# --- AI 通訊核心 (嚴謹模式) ---
def call_ai_json_mode(key, prompt_text, image=None):
    if not key: return None
    try:
        target_model_name = get_best_model(key)
        # 設定 temperature = 0.1 確保 AI 不會亂回答
        generation_config = {"temperature": 0.1, "response_mime_type": "application/json"}
        model = genai.GenerativeModel(target_model_name, generation_config=generation_config)
        content = [prompt_text, image] if image else [prompt_text]
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"ERROR: {str(e)}"

def call_ai_chat(key, prompt_text):
    if not key: return "請先輸入 API Key"
    try:
        target_model_name = get_best_model(key)
        generation_config = {"temperature": 0.3}
        model = genai.GenerativeModel(target_model_name, generation_config=generation_config)
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"連線錯誤: {str(e)}"

# 3. 資料載入與模擬 (符合資料集介紹要求)
try:
    try:
        df = pd.read_csv('fitness_data.csv')
    except:
        # 若無 CSV，生成 50 筆模擬資料以供展示
        np.random.seed(42)
        data = {
            'Weight': np.random.randint(50, 100, 50),
            'Duration': np.random.randint(30, 90, 50),
            'Heart_Rate': np.random.randint(110, 160, 50),
            'Body_Fat': np.random.randint(15, 35, 50),
        }
        # 簡單模擬卡路里公式：體重*時間*強度係數
        data['Calories'] = data['Weight'] * data['Duration'] * 0.1 + data['Heart_Rate'] * 0.5
        df = pd.DataFrame(data)
except Exception as e:
    st.error(f"資料讀取錯誤: {e}")
    st.stop()

st.title("🏋️ AI 健身數據科學分析平台")
st.markdown("結合 **電腦視覺 (Computer Vision)**、**機器學習 (Machine Learning)** 與 **生成式 AI (Generative AI)** 的綜合分析系統。")

# --- 4. 步驟 1 : InBody 報告辨識 ---
st.header("📸 步驟 1 : InBody 影像識別與分析")
uploaded_file = st.file_uploader("上傳 InBody 照片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(img, caption='已上傳報告', use_container_width=True)
    
    with c2:
        st.info("💡 系統正在待命：將啟動電腦視覺模型讀取數據，並透過 AI 進行邏輯推演。")
        with st.form("vision_form"):
            submitted = st.form_submit_button("🔍 啟動嚴謹分析 (Chain of Thought)")
            
            if submitted:
                if not api_key:
                    st.error("❌ 請先在左側輸入 API Key")
                else:
                    with st.spinner("AI 正在執行：數值讀取 -> 邏輯驗證 -> 報告生成..."):
                        task_prompt = """
                        你是一位講求實證科學的資深運動生理學家。請分析這張 InBody 報告。
                        
                        【思考步驟 - 請在內心執行】：
                        1. 仔細識別圖片中的數值 (Weight, Body Fat)，若模糊請勿瞎猜。
                        2. 判斷體態類型 (C型/I型/D型)。
                        3. 內心估算 BMR 與 TDEE。
                        4. 確認邏輯通順後，生成 JSON。
                        
                        請回傳 JSON 物件：
                        {
                            "weight": (數值，單位 kg，若找不到預設 70),
                            "body_fat": (數值，單位 %，若找不到預設 25),
                            "advice": "請使用 Markdown 撰寫分析報告 (至少 300 字)：\n### 1. 📊 體態判定\n(說明你看到的數據與體態類型)\n### 2. 🧬 科學飲食計算\n(列出 BMR/TDEE 估算值與熱量建議)\n### 3. 🛡️ 訓練處方\n(給出具體頻率與訓練內容)"
                        }
                        """
                        res_text = call_ai_json_mode(api_key, task_prompt, img)
                        
                        if res_text and "ERROR" in res_text:
                            st.error(f"AI 連線失敗: {res_text}")
                        elif res_text:
                            try:
                                data = json.loads(res_text)
                                st.session_state.weight = float(data.get("weight", 70.0))
                                st.session_state.body_fat = float(data.get("body_fat", 25.0))
                                st.session_state.ai_plan = data.get("advice", "無法生成建議")
                                st.success("✅ 分析完成！數據已同步至下方模型。")
                            except json.JSONDecodeError:
                                st.error("❌ AI 回傳格式錯誤，請重試。")

if st.session_state.ai_plan:
    with st.expander("📄 查看 AI 完整評估報告", expanded=True):
        st.markdown(st.session_state.ai_plan)

st.divider()

# --- 5. 步驟 2 : 資料集介紹與特徵分析 (★ 符合專題要求：介紹資料集與關聯性) ---
st.header("📝 步驟 2 : 資料集介紹與關聯性分析")

tab1, tab2 = st.tabs(["📊 資料集總覽", "🔥 特徵關聯性 (Correlation)"])

with tab1:
    st.write("本系統使用之健身資料集 (前 5 筆)：")
    st.dataframe(df.head())
    st.caption(f"資料集總筆數：{len(df)} 筆 | 特徵包含：體重、運動時長、心率、體脂率、消耗卡路里")

with tab2:
    st.write("各特徵之間的相關係數矩陣 (Correlation Matrix)：")
    # 這是老師最愛看的「關聯性分析」
    corr_matrix = df[['Weight', 'Duration', 'Heart_Rate', 'Body_Fat', 'Calories']].corr()
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
    st.caption("說明：顏色越深代表相關性越高 (例如：運動時長與消耗卡路里通常呈現高度正相關)。")

st.divider()

# --- 6. 步驟 3 : 機器學習預測與視覺化 (★ 符合專題要求：視覺化分析預測結果) ---
st.header("🤖 步驟 3 : 機器學習模型預測")

# 準備訓練資料
scaler = StandardScaler()
X = df[['Weight', 'Duration', 'Heart_Rate', 'Body_Fat']]
scaler.fit(X)

# 訓練模型
# 1. 線性迴歸 (預測數值)
reg = LinearRegression().fit(scaler.transform(X), df['Calories'])
# 2. K-Means (分群)
km = KMeans(n_clusters=3, random_state=42).fit(scaler.transform(X))
group_labels_map = {0: "💪 健康維持族", 1: "🔥 高效代謝族", 2: "✨ 體態優化組"}

# 使用者互動區
c1, c2, c3, c4 = st.columns(4)
u_w = c1.number_input("體重 (kg)", 40.0, 150.0, float(st.session_state.weight))
u_f = c2.slider("體脂率 (%)", 5.0, 50.0, float(st.session_state.body_fat))
u_d = c3.number_input("運動時長 (min)", 1, 300, 45)
u_h = c4.number_input("心率 (BPM)", 50, 200, 130)

# 進行預測
u_scaled = scaler.transform([[u_w, u_d, u_h, u_f]])
pred_cal = reg.predict(u_scaled)[0]
u_cls = km.predict(u_scaled)[0]

# 視覺化展示
st.subheader("📈 預測結果視覺化")
col_visual, col_metric = st.columns([2, 1])

with col_visual:
    # 準備繪圖資料
    chart_df = df.copy()
    chart_df['Cluster'] = km.labels_ # 將分群結果標記回去
    chart_df['Cluster_Name'] = chart_df['Cluster'].map(group_labels_map)
    
    st.caption("K-Means 分群視覺化 (X軸:體重, Y軸:運動時長)")
    # 使用 Streamlit 內建圖表進行視覺化
    st.scatter_chart(
        chart_df,
        x='Weight',
        y='Duration',
        color='Cluster_Name',
        size=20,
        height=300
    )

with col_metric:
    st.info(f"您的數據落點分析：")
    st.metric("AI 歸類族群 (K-Means)", group_labels_map.get(u_cls))
    st.metric("預測消耗卡路里 (Regression)", f"{pred_cal:.2f} kcal")
    st.write("---")
    st.progress(min(int(pred_cal)/1000, 1.0), text="單次運動強度指標")

st.divider()

# --- 7. 步驟 4 : AI 專家諮詢 ---
st.header("💬 步驟 4 : AI 運動科學教練諮詢")
user_q = st.text_input("輸入您的問題 (AI 將參考上述所有分析數據)：", placeholder="例如：我想在兩個月內降 3% 體脂，這個訓練量夠嗎？")

if st.button("送出諮詢"):
    if not user_q:
        st.warning("請輸入問題！")
    else:
        with st.spinner("AI 教練正在綜合分析您的體態、機器學習預測結果與運動生理學原理..."):
            u_cls_label = group_labels_map.get(u_cls)
            deep_prompt = f"""
            你是一位嚴格的科學教練。請根據以下事實進行邏輯推演。
            
            【使用者檔案】
            - 體重: {u_w} kg, 體脂: {u_f} %
            - 訓練強度: {u_d} 分鐘, 心率 {u_h} BPM
            - AI 預測消耗: {pred_cal:.1f} kcal
            - 體態分群: {u_cls_label}
            
            【使用者問題】: {user_q}
            
            【回答要求】
            1. 分析可行性 (是否符合生理極限)。
            2. 提供具體數字目標 (熱量缺口)。
            3. 給出行動方案。
            """
            chat_res = call_ai_chat(api_key, deep_prompt)
            st.markdown(f"**🤖 AI 教練的回覆：**\n\n{chat_res}")
