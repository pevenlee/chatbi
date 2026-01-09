import streamlit as st
import pandas as pd
import json
import warnings
import os
import re
import numpy as np
import base64
import time
from google import genai
from google.genai import types

# 忽略无关警告
warnings.filterwarnings('ignore')

# ================= 1. 配置区域 =================

st.set_page_config(
    page_title="ChatBI (Lite)", 
    layout="wide", 
    page_icon="🧬", 
    initial_sidebar_state="expanded"
)

# 修改为从 secrets 读取
try:
    FIXED_API_KEY = st.secrets["GENAI_API_KEY"]
except:
    FIXED_API_KEY = "" # 防止本地运行时报错

# 【这里填你的 Excel 文件名】
FIXED_FILE_NAME = "hcmdata.xlsx" 

# 【Logo 文件名】
LOGO_FILE = "logo.png"

# 【代理设置】
# PROXY_URL = "http://127.0.0.1:10809"

# 【限制设置】
PREVIEW_ROW_LIMIT = 500   # 纯表模式下可以适当增加预览行数
EXPORT_ROW_LIMIT = 5000   

# ================= 2. 核心逻辑函数 =================

@st.cache_resource
def get_client():
    if not FIXED_API_KEY: return None
    # 注意：这里不需要 os.environ 设置代理了
    try:
        return genai.Client(api_key=FIXED_API_KEY, http_options={'api_version': 'v1beta'})
    except Exception as e:
        st.error(f"SDK 初始化失败: {e}")
        return None

def safe_generate_content(client, model_name, contents, config=None, retries=6):
    base_delay = 10 
    for i in range(retries):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if i < retries - 1:
                    wait_time = base_delay * (2 ** i)
                    st.toast(f"⏳ API 配额限制，正在深度等待 ({i+1}/{retries})...")
                    time.sleep(wait_time)
                    continue
            raise e

@st.cache_data
def load_data():
    if not os.path.exists(FIXED_FILE_NAME):
        st.error(f"❌ 找不到文件: {FIXED_FILE_NAME}")
        return None
    try:
        if FIXED_FILE_NAME.endswith('.csv'):
            df = pd.read_csv(FIXED_FILE_NAME)
        else:
            df = pd.read_excel(FIXED_FILE_NAME)
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if any(k in str(col) for k in ['额', '量', 'Sales', 'Qty', '金额']):
                try:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '', regex=False),
                        errors='coerce'
                    ).fillna(0)
                except: pass
        return df
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None

def get_img_as_base64(file_path):
    if not os.path.exists(file_path): return ""
    with open(file_path, "rb") as f: data = f.read()
    return base64.b64encode(data).decode()

def get_history_context(messages, turn_limit=3):
    if len(messages) <= 1: return "无历史对话。"
    recent_msgs = messages[:-1]
    valid_msgs = [m for m in recent_msgs if m['type'] in ['text', 'report_block']]
    slice_start = max(0, len(valid_msgs) - (turn_limit * 2))
    target_msgs = valid_msgs[slice_start:]
    context_list = []
    for msg in target_msgs:
        role = "User" if msg['role'] == 'user' else "AI"
        content_str = ""
        if msg['type'] == 'text':
            content_str = msg['content']
        elif msg['type'] == 'report_block':
            data = msg['content']
            mode = data.get('mode', 'analysis')
            if mode == 'simple':
                s = data.get('summary', {})
                content_str = f"[历史取数] 意图: {s.get('intent')}, 逻辑: {s.get('logic')}"
            else:
                intent = data.get('intent', '无意图')
                insight = data.get('insight', '无洞察')
                angles_summary = [f"<{a['title']}: {a['explanation']}>" for a in data.get('angles_data', [])]
                content_str = f"[历史分析] 意图: {intent} | 发现: {'; '.join(angles_summary)} | 洞察: {insight}"
        context_list.append(f"{role}: {content_str}")
    return "\n".join(context_list)

def analyze_time_structure(df):
    time_col = None
    for col in df.columns:
        if '年季' in col or 'Quarter' in col or 'Date' in col:
            sample = str(df[col].iloc[0])
            if 'Q' in sample and len(sample) <= 6:
                time_col = col; break
    if time_col:
        sorted_periods = sorted(df[time_col].unique().astype(str))
        max_q = sorted_periods[-1]
        min_q = sorted_periods[0]
        mat_list = sorted_periods[-4:] if len(sorted_periods) >= 4 else sorted_periods
        is_mat_complete = True
        mat_list_prior = []
        if len(sorted_periods) >= 8:
            mat_list_prior = sorted_periods[-8:-4]
        elif len(sorted_periods) >= 4:
            mat_list_prior = sorted_periods[:-4]
            is_mat_complete = False
        else:
            is_mat_complete = False
        ytd_list, ytd_list_prior = [], []
        import re
        year_match = re.search(r'(\d{4})', str(max_q))
        if year_match:
            curr_year = year_match.group(1)
            try:
                prev_year = str(int(curr_year) - 1)
                ytd_list = [p for p in sorted_periods if curr_year in str(p)]
                expected_priors = [str(p).replace(curr_year, prev_year) for p in ytd_list]
                ytd_list_prior = [p for p in sorted_periods if p in expected_priors]
            except: pass
        return {
            "col_name": time_col, "all_periods": sorted_periods, "max_q": max_q, "min_q": min_q, 
            "mat_list": mat_list, "mat_list_prior": mat_list_prior, "is_mat_complete": is_mat_complete,
            "ytd_list": ytd_list, "ytd_list_prior": ytd_list_prior
        }
    return {"error": "未找到标准年季列"}

def build_metadata(df, time_context):
    info = []
    info.append(f"【时间列名】: {time_context.get('col_name')}")
    info.append(f"【当前MAT】: {time_context.get('mat_list')}")
    info.append(f"【同期MAT完整性】: {time_context.get('is_mat_complete')}")
    info.append(f"【当前YTD】: {time_context.get('ytd_list')}")
    for col in df.columns:
        dtype = str(df[col].dtype)
        uniques = df[col].dropna().unique()
        desc = f"- `{col}` ({dtype})"
        if dtype == 'object' or len(uniques) < 2000:
            vals = list(uniques[:5]) if len(uniques) > 100 else list(uniques)
            desc += f" | 示例: {vals}"
        info.append(desc)
    return "\n".join(info)

def normalize_result(res):
    if isinstance(res, pd.DataFrame): return res
    if isinstance(res, pd.Series): return res.to_frame()
    if isinstance(res, dict):
        try: return pd.DataFrame(list(res.items()), columns=['指标', '数值'])
        except: pass
    try: return pd.DataFrame([res])
    except: return pd.DataFrame({"Result": [str(res)]})

def format_df_for_display(df_raw):
    if not isinstance(df_raw, pd.DataFrame): return df_raw
    df_fmt = df_raw.copy()
    percent_keywords = ['Rate', 'Ratio', 'Share', 'Percent', 'Pct', 'YoY', 'CAGR', '率', '比', '占比', '份额']
    exclude_keywords = ['Value', 'Amount', 'Qty', 'Volume', 'Contribution', 'Abs', '额', '量']
    for col in df_fmt.columns:
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            col_str = str(col)
            is_percent = any(k in col_str for k in percent_keywords)
            has_exclude = any(k in col_str for k in exclude_keywords)
            if is_percent and not has_exclude:
                df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "-")
            else:
                is_integer = False
                try:
                    if (df_fmt[col].dropna() % 1 == 0).all(): is_integer = True
                except: pass
                fmt = "{:,.0f}" if is_integer else "{:,.2f}"
                df_fmt[col] = df_fmt[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else "-")
    return df_fmt

def parse_response(text):
    reasoning = text
    json_data = None
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            potential_json = text[start_idx : end_idx + 1]
            try:
                json_data = json.loads(potential_json)
                reasoning = text[:start_idx].strip()
            except json.JSONDecodeError: pass
    except Exception: pass
    return reasoning, json_data

# ================= 3. UI 样式 =================

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        .stApp { background-color: #F8F9FC; font-family: 'Inter', -apple-system, "Microsoft YaHei", sans-serif; }
        
        /* =================================================================
           🔥 强力隐藏 Streamlit 原生 UI 元素 (Manage App / Deploy / Menu)
           ================================================================= */
        
        /* 1. 隐藏右上角汉堡菜单 */
        #MainMenu {visibility: hidden; display: none;}
        
        /* 2. 隐藏底部 "Made with Streamlit" */
        footer {visibility: hidden; display: none;}
        
        /* 3. 隐藏顶部彩色装饰条 */
        header {visibility: hidden; display: none;}
        
        /* 4. 核心：隐藏 "Manage app" 按钮和工具栏 */
        [data-testid="stToolbar"] {
            visibility: hidden !important; 
            display: none !important;
            height: 0px !important;
        }
        
        /* 5. 隐藏可能出现的浮动部署按钮 */
        .stDeployButton {
            visibility: hidden !important; 
            display: none !important;
        }
        
        /* 6. 隐藏右上角的运行状态动画 (Running Man) */
        [data-testid="stStatusWidget"] {
            visibility: hidden !important;
        }
        
        /* ================================================================= */

        .header-container {
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(12px);
            padding: 12px 24px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.03); 
            margin-bottom: 30px; display: flex; align-items: center; border: 1px solid rgba(255,255,255,0.6);
        }
        .header-logo-img { height: 32px; margin-right: 12px; width: auto; }
        .header-title {
            color: #0F172A; font-size: 22px; font-weight: 800; margin: 0; letter-spacing: -0.5px;
            background: linear-gradient(90deg, #0F172A 0%, #334155 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .header-meta {
            color: #64748B; font-size: 12px; font-family: 'JetBrains Mono', monospace;
            background: #F1F5F9; padding: 4px 10px; border-radius: 6px; margin-left: 10px;
        }
        
        div.stButton > button {
            border: 1px solid #E2E8F0; background-color: #FFFFFF; color: #1E293B;
            border-radius: 8px; padding: 15px 20px; font-size: 14px; font-weight: 500;
            transition: all 0.2s; text-align: left; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        div.stButton > button:hover {
            border-color: #3B82F6; color: #3B82F6; background-color: #F0F9FF;
            transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        }
        
        .summary-box {
            background-color: #F8FAFC; padding: 20px; border-radius: 8px;
            border: 1px solid #E2E8F0; border-left: 4px solid #10B981; margin-bottom: 20px;
        }
        .summary-title {
            font-family: 'Microsoft YaHei', sans-serif; font-weight: 600; color: #059669; 
            font-size: 14px; margin-bottom: 12px; letter-spacing: 0.5px;
        }
        .summary-list li { margin-bottom: 8px; font-size: 14px; color: #334155; display: flex; }
        .summary-label { min-width: 60px; color: #64748B; font-size: 12px; font-weight: 500; margin-top: 2px; }
        
        .tech-card {
            background-color: white; padding: 24px; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;
            border: 1px solid #E2E8F0; transition: all 0.2s ease-in-out;
        }
        .tech-card:hover {
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05); transform: translateY(-2px); border-color: #CBD5E1;
        }
        .angle-title { font-size: 16px; font-weight: 700; color: #1E293B; margin-bottom: 6px; }
        .angle-desc { color: #64748B; font-size: 13px; margin-bottom: 15px; line-height: 1.5; }
        .mini-insight {
            background-color: #F1F5F9; padding: 12px 16px; border-radius: 6px;
            font-size: 13px; color: #475569; margin-top: 15px; border-left: 3px solid #94A3B8; line-height: 1.6;
        }
        .insight-box {
            background: white; padding: 24px; border-radius: 12px; position: relative;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0;
        }
        .insight-box::before {
            content: ''; position: absolute; left: 0; top: 12px; bottom: 12px;
            width: 4px; background: linear-gradient(180deg, #3B82F6 0%, #06B6D4 100%);
            border-radius: 0 4px 4px 0;
        }
        .step-header {
            font-weight: 700; color: #1E293B; font-size: 16px; margin-top: 35px; 
            margin-bottom: 20px; display: flex; align-items: center;
        }
        .step-header::before {
            content: ''; display: inline-block; width: 4px; height: 18px;
            background: #3B82F6; margin-right: 12px; border-radius: 2px;
        }
        div[data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 8px; overflow: hidden; }
        
        .stop-btn-container button { border: 1px solid #EF4444 !important; color: #EF4444 !important; }
        .stop-btn-container button:hover { background-color: #FEF2F2 !important; }
        </style>
    """, unsafe_allow_html=True)

# ================= 4. 主界面逻辑 =================

inject_custom_css()

logo_html = ""
if os.path.exists(LOGO_FILE):
    b64_img = get_img_as_base64(LOGO_FILE)
    logo_html = f'<img src="data:image/png;base64,{b64_img}" class="header-logo-img">'

st.markdown(f"""
    <div class="header-container">
        {logo_html}
        <div class="header-title">ChatBI (Lite)</div>
        <div style="flex-grow: 1;"></div>
        <div class="header-meta">数据源: {FIXED_FILE_NAME}</div>
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_draft" not in st.session_state:
    st.session_state.last_query_draft = ""
if "is_interrupted" not in st.session_state:
    st.session_state.is_interrupted = False

client = get_client()
df = load_data()

if df is not None:
    time_context = analyze_time_structure(df)
    meta_data = build_metadata(df, time_context)
    
    with st.sidebar:
        if os.path.exists(LOGO_FILE):
            st.image(LOGO_FILE, width=150)
            st.markdown("---")
        else:
            st.markdown("### 🧬 控制台")
        st.caption("状态: 在线 (Active)")
        st.info(f"总行数: {len(df):,}")
        st.info(f"时间跨度: {time_context.get('min_q')} ~ {time_context.get('max_q')}")
        st.divider()
        if st.button("🗑️ 清空会话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_query_draft = ""
            st.session_state.is_interrupted = False
            st.rerun()

    # 1. 渲染历史记录
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "report_block":
                content = msg["content"]
                mode = content.get('mode', 'analysis') 
                
                if mode == 'simple':
                    if 'summary' in content:
                        s = content['summary']
                        st.markdown(f"""
                        <div class="summary-box">
                            <div class="summary-title">⚡ 取数执行协议 (Protocol)</div>
                            <ul class="summary-list">
                                <li><span class="summary-label">意图</span> <span class="summary-val">{s.get('intent', '-')}</span></li>
                                <li><span class="summary-label">范围</span> <span class="summary-val">{s.get('scope', '-')}</span></li>
                                <li><span class="summary-label">指标</span> <span class="summary-val">{s.get('metrics', '-')}</span></li>
                                <li><span class="summary-label">逻辑</span> <span class="summary-val">{s.get('logic', '-')}</span></li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ 数据提取完成")
                    
                    if 'data' in content:
                        data_payload = content['data']
                        if isinstance(data_payload, pd.DataFrame):
                            data_payload = {"查询结果": data_payload}
                        
                        for table_name, table_df in data_payload.items():
                            if len(data_payload) > 1:
                                st.markdown(f"**📄 {table_name}**")
                            
                            st.dataframe(format_df_for_display(table_df).head(PREVIEW_ROW_LIMIT), use_container_width=True)

                            csv = table_df.head(EXPORT_ROW_LIMIT).to_csv(index=False).encode('utf-8-sig')
                            st.download_button(f"📥 导出 ({table_name})", csv, f"{table_name}.csv", "text/csv", key=f"dl_simple_{msg_idx}_{table_name}")
                            if len(data_payload) > 1: st.markdown("---")

                else:
                    st.markdown('<div class="step-header">1. 意图深度解析</div>', unsafe_allow_html=True)
                    st.markdown(content.get('intent', ''))
                    if 'angles_data' in content:
                        st.markdown('<div class="step-header">2. 多维分析报告</div>', unsafe_allow_html=True)
                        for i, angle in enumerate(content['angles_data']):
                            with st.container():
                                st.markdown(f"""
                                <div class="tech-card">
                                    <div class="angle-title">📐 {angle['title']}</div>
                                    <div class="angle-desc">{angle['desc']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.dataframe(format_df_for_display(angle['data']).head(PREVIEW_ROW_LIMIT), use_container_width=True)
                                
                                csv = angle['data'].head(EXPORT_ROW_LIMIT).to_csv(index=False).encode('utf-8-sig')
                                st.download_button(f"📥 下载数据", csv, f"angle_{i}_hist.csv", "text/csv", key=f"dl_hist_{msg_idx}_{i}")
                                st.markdown(f'<div class="mini-insight">💡 <b>深度解读:</b> {angle["explanation"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="step-header">3. 综合业务洞察</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="insight-box">{content.get("insight", "")}</div>', unsafe_allow_html=True)

    # 2. 引导卡片
    if len(st.session_state.messages) == 0 and not st.session_state.is_interrupted:
        st.markdown("### 💡 猜你想问")
        col1, col2, col3 = st.columns(3)
        q1, q2, q3 = "康缘在各个省份的市场份额多少？", "康缘的哪些产品同比增长较高？", "康缘不同区域的市场表现怎么样？"
        if col1.button(f"🗺️ **份额分析**\n\n{q1}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "type": "text", "content": q1}); st.rerun()
        if col2.button(f"📈 **增长分析**\n\n{q2}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "type": "text", "content": q2}); st.rerun()
        if col3.button(f"📊 **区域表现**\n\n{q3}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "type": "text", "content": q3}); st.rerun()

    # 3. 中止回显 & 输入框
    if st.session_state.is_interrupted:
        st.warning("⚠️ 已中止生成。您可以修改刚才的问题并重新发送：")
        def submit_edit():
            new_val = st.session_state["edit_input_widget"]
            if new_val:
                st.session_state.messages.append({"role": "user", "type": "text", "content": new_val})
                st.session_state.is_interrupted = False
                st.session_state.last_query_draft = ""
        st.text_area("编辑问题", value=st.session_state.last_query_draft, key="edit_input_widget", height=100)
        st.button("🚀 重新发送", on_click=submit_edit, type="primary")

    if not st.session_state.is_interrupted:
        if query_input := st.chat_input("🔎 请输入问题 (例如：“查询康缘销量” 或 “分析增长趋势”)"):
            st.session_state.last_query_draft = query_input
            st.session_state.messages.append({"role": "user", "type": "text", "content": query_input})
            st.rerun()

    # 4. 核心处理逻辑
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and not st.session_state.is_interrupted:
        current_query = st.session_state.messages[-1]["content"]
        history_context_str = get_history_context(st.session_state.messages, turn_limit=3)
        stop_btn_placeholder = st.empty()
        
        if stop_btn_placeholder.button("⏹️ 中止生成 (Stop)", type="primary", use_container_width=True):
            st.session_state.is_interrupted = True; st.rerun()

        with st.chat_message("assistant"):
            try:
                # 意图路由
                intent_type = "analysis" 
                with st.spinner("🔄 正在识别需求场景..."):
                    router_prompt = f"""
                    基于用户当前问题："{current_query}" 以及历史上下文判断用户意图。
                    【历史上下文】:{history_context_str}
                    请将其分类为以下三类之一：
                    1. "simple": 简单取数、排序、排名、计算基础指标（如增长率、同比）。
                    2. "analysis": 开放式问题，寻求洞察、原因分析、市场格局等深度内容。
                    3. "irrelevant": 与数据完全无关的闲聊。
                    仅输出 JSON: {{"type": "simple" 或 "analysis" 或 "irrelevant"}}
                    """
                    router_resp = safe_generate_content(
                        client, "gemini-2.0-flash", router_prompt, config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    try: intent_type = json.loads(router_resp.text).get('type', 'analysis')
                    except: intent_type = 'analysis'

                mat_list = time_context.get('mat_list')
                mat_list_prior = time_context.get('mat_list_prior')
                is_mat_complete = time_context.get('is_mat_complete')
                ytd_list = time_context.get('ytd_list')
                ytd_list_prior = time_context.get('ytd_list_prior')

                if intent_type == 'irrelevant':
                    st.warning("⚠️ 当前提问不在 ChatBI 覆盖范围内")
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": "抱歉，当前提问与数据内容无关。"})

                # ================= [Simple Mode] =================
                elif intent_type == 'simple':
                    with st.spinner("⚡ 正在解析意图并生成代码..."):
                        simple_prompt = f"""
                        你是一位 Pandas 数据处理专家。用户需求："{current_query}"
                        【元数据】{meta_data}
                        【历史记录】{history_context_str}
                        【时间上下文】MAT: {mat_list} (完整性: {is_mat_complete}), YTD: {ytd_list}
                        
                        【任务】
                        1. 生成 `results` 字典：Key=表标题, Value=DataFrame。
                        2. **严禁绘图**：不要生成任何 fig, plt, sns 相关代码。只处理数据。
                        
                        【严格约束】
                        - 在代码内部定义所有列表/变量。
                        - 结果必须是 DataFrame。
                        
                        输出 JSON: {{ 
                            "summary": {{ "intent": "意图描述", "scope": "数据范围", "metrics": "指标", "logic": "计算逻辑" }}, 
                            "code": "results = {{...}}" 
                        }}
                        """
                        simple_resp = safe_generate_content(
                            client, "gemini-3-pro-preview", simple_prompt, config=types.GenerateContentConfig(response_mime_type="application/json")
                        )
                        simple_json = json.loads(simple_resp.text)
                        
                        execution_context = {
                            'df': df, 'data': df, 'df_market': df, 'df_mat': df, 'df_ytd': df,
                            'pd': pd, 'np': np, 
                            'results': {}, 'result': None,
                            'current_mat': mat_list, 'mat_list': mat_list, 'prior_mat': mat_list_prior,
                            'mat_list_prior': mat_list_prior, 'ytd_list': ytd_list, 'ytd_list_prior': ytd_list_prior
                        }
                        exec(simple_json['code'], execution_context)
                        
                        final_results = execution_context.get('results')
                        
                        if not final_results and execution_context.get('result') is not None:
                            final_results = {"查询结果": execution_context.get('result')}
                        
                        if final_results:
                            formatted_results = {k: normalize_result(v) for k, v in final_results.items()}
                            s = simple_json.get('summary', {})
                            
                            st.markdown(f"""
                            <div class="summary-box">
                                <div class="summary-title">⚡ 取数执行协议</div>
                                <ul class="summary-list">
                                    <li><span class="summary-label">意图</span> {s.get('intent','-')}</li>
                                    <li><span class="summary-label">逻辑</span> {s.get('logic','-')}</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for table_name, table_df in formatted_results.items():
                                if len(formatted_results) > 1: st.markdown(f"**📄 {table_name}**")
                                
                                st.dataframe(format_df_for_display(table_df).head(PREVIEW_ROW_LIMIT), use_container_width=True)

                                csv = table_df.head(EXPORT_ROW_LIMIT).to_csv(index=False).encode('utf-8-sig')
                                st.download_button(f"📥 导出 ({table_name})", csv, f"{table_name}.csv", "text/csv", key=f"dl_simple_{msg_idx}_{table_name}")
                                if len(formatted_results) > 1: st.markdown("---")
                            
                            st.session_state.messages.append({
                                "role": "assistant", "type": "report_block",
                                "content": { "mode": "simple", "summary": s, "data": formatted_results }
                            })
                        else:
                            st.error("未提取到数据")
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": "抱歉，未提取到有效数据。"})

                # ================= [Analysis Mode] =================
                else:
                    with st.spinner("🧠 正在拆解问题..."):
                        prompt_plan = f"""
                        你是一位医药行业 BI 专家。请将问题："{current_query}" 拆解为 2-5 个分析角度。
                        请结合竞争关系数据库，带入竞争对手视角进行分析。
                        当被询问评价表现时，请尽可能结合时间动态地思考问题
                        【元数据】{meta_data}
                        【历史记录】{history_context_str}
                        【时间上下文】MAT: {mat_list} (完整性: {is_mat_complete}), YTD: {ytd_list}
                        
                        【严格约束】
                        0. **数据源变量名**：DataFrame 变量名为 `df`。
                        1. **严禁绘图**：不要生成任何 fig, plt, sns 相关代码。只处理数据。
                        2. **结果赋值**：最终的 DataFrame 必须赋值给变量 `result`。
                        3. **语言**：所有分析思路、标题、描述必须使用**中文**。
                        
                        输出 JSON: {{ "intent_analysis": "意图深度解析(Markdown)", "angles": [ {{"title": "分析角度标题", "code": "result=..."}} ] }}
                        """
                        response_plan = safe_generate_content(client, "gemini-3-pro-preview", prompt_plan)
                        reasoning_text, plan_json = parse_response(response_plan.text)

                    if plan_json and 'angles' in plan_json:
                        st.markdown('<div class="step-header">1. 意图深度解析</div>', unsafe_allow_html=True)
                        st.markdown(plan_json.get('intent_analysis', '自动分析'))
                        
                        angles_data = [] 
                        st.markdown('<div class="step-header">2. 多维分析报告</div>', unsafe_allow_html=True)
                        
                        for i, angle in enumerate(plan_json['angles']):
                            with st.container():
                                st.markdown(f"""
                                <div class="tech-card">
                                    <div class="angle-title">📐 {angle['title']}</div>
                                    <div class="angle-desc">{angle.get('description','')}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                time.sleep(5) 
                                
                                try:
                                    execution_context = {
                                        'df': df, 'data': df, 'df_market': df, 'df_mat': df, 'df_ytd': df,
                                        'pd': pd, 'np': np,
                                        'result': None,
                                        'current_mat': mat_list, 'mat_list': mat_list, 'prior_mat': mat_list_prior,
                                        'mat_list_prior': mat_list_prior, 'ytd_list': ytd_list, 'ytd_list_prior': ytd_list_prior
                                    }
                                    exec(angle['code'], execution_context)
                                    
                                    if execution_context.get('result') is None:
                                        for k, v in list(execution_context.items()):
                                            if isinstance(v, pd.DataFrame) and k != 'df':
                                                execution_context['result'] = v; break
                                    
                                    if execution_context.get('result') is not None:
                                        res_df = normalize_result(execution_context['result'])
                                        
                                        st.dataframe(format_df_for_display(res_df).head(PREVIEW_ROW_LIMIT), use_container_width=True)
                                        
                                        csv = res_df.head(EXPORT_ROW_LIMIT).to_csv(index=False).encode('utf-8-sig')
                                        st.download_button(f"📥 下载", csv, f"angle_{i}.csv", "text/csv", key=f"dl_{i}")
                                        
                                        with st.spinner(f"⚡ 深度解读..."):
                                            mini_prompt = f"""
                                            对数据进行深度解读（200-300字）。
                                            预览：{res_df.head(20).to_markdown()}
                                            要求：提炼趋势/异常，结合业务含义，语言专业，不给建议。
                                            """
                                            mini_resp = safe_generate_content(client, "gemini-2.0-flash", mini_prompt)
                                            explanation = mini_resp.text
                                            st.markdown(f'<div class="mini-insight">💡 <b>深度解读:</b> {explanation}</div>', unsafe_allow_html=True)
                                        
                                        angles_data.append({
                                            "title": angle['title'], "desc": angle.get('description',''),
                                            "data": res_df, "explanation": explanation
                                        })
                                    else:
                                        st.error("该角度未返回数据")
                                except Exception as e:
                                    st.error(f"执行报错: {e}")

                        if angles_data:
                            st.markdown('<div class="step-header">3. 综合业务洞察</div>', unsafe_allow_html=True)
                            with st.spinner("🤖 生成综述..."):
                                time.sleep(5)
                                all_findings = "\n".join([f"[{ad['title']}]: {ad['explanation']}" for ad in angles_data])
                                final_prompt = f"""
                                问题: "{current_query}"
                                发现: {all_findings}
                                生成最终洞察 (Markdown)。严禁提供建议，仅陈述事实。
                                """
                                resp_final = safe_generate_content(client, "gemini-3-pro-preview", final_prompt)
                                insight_text = resp_final.text
                                st.markdown(f'<div class="insight-box">{insight_text}</div>', unsafe_allow_html=True)
                                
                                st.session_state.messages.append({
                                    "role": "assistant", "type": "report_block",
                                    "content": {
                                        "mode": "analysis", "intent": plan_json.get('intent_analysis', ''),
                                        "angles_data": angles_data, "insight": insight_text
                                    }
                                })
                    else:
                        st.error("无法生成分析方案")
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": "分析生成失败"})
            except Exception as e:
                st.error(f"系统错误: {e}")
            finally:
                stop_btn_placeholder.empty()




