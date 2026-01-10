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

# ================= 1. 基础配置 =================

st.set_page_config(
    page_title="ChatBI", 
    layout="wide", 
    page_icon="🧬", 
    initial_sidebar_state="expanded"
)

# --- VI 体系与 UI 样式定义 ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* ================= VI 变量定义 (医药魔方风格) ================= */
        :root {
            --pc-primary-blue: #005ADE; /* 魔方蓝 - 主色调 */
            --pc-dark-blue: #004099;    /* 深蓝 - 用于强调 */
            --pc-bg-light: #F4F6F9;     /* 浅灰蓝背景 - 专业洁净 */
            --pc-text-main: #1A2B47;    /* 主要文字 */
            --pc-text-sub: #5E6D82;     /* 次要文字 */
        }

        /* 全局样式应用 */
        .stApp {
            background-color: var(--pc-bg-light);
            font-family: 'Inter', "Microsoft YaHei", sans-serif;
            color: var(--pc-text-main);
        }

        /* ================= 顶部固定导航栏 ================= */
        .fixed-header-container {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 64px;
            background-color: #FFFFFF;
            box-shadow: 0 2px 12px rgba(0, 90, 222, 0.08);
            z-index: 999999;
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 24px;
            border-bottom: 1px solid #E6EBF5;
        }
        .nav-left { display: flex; align-items: center; }
        .nav-logo-img { height: 32px; width: auto; margin-right: 12px; }
        .nav-brand-text { font-size: 18px; font-weight: 700; color: var(--pc-primary-blue); letter-spacing: 0.5px; }
        
        .nav-center { display: flex; gap: 32px; font-weight: 600; font-size: 15px; }
        .nav-item { color: var(--pc-text-sub); cursor: pointer; padding: 20px 4px; position: relative; transition: all 0.2s; }
        .nav-item:hover { color: var(--pc-primary-blue); }
        .nav-item.active { color: var(--pc-primary-blue); }
        .nav-item.active::after {
            content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 3px;
            background-color: var(--pc-primary-blue); border-radius: 2px 2px 0 0;
        }
        
        .nav-right { display: flex; align-items: center; gap: 16px; }
        .nav-avatar {
            width: 32px; height: 32px; background-color: var(--pc-primary-blue); color: white;
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-size: 12px; font-weight: bold; border: 2px solid #E6EBF5;
        }
        .nav-exit-btn {
            border: 1px solid #DCDFE6; padding: 5px 12px; border-radius: 4px;
            font-size: 13px; color: var(--pc-text-sub); background: white; cursor: pointer; transition: all 0.2s;
        }
        .nav-exit-btn:hover { border-color: var(--pc-primary-blue); color: var(--pc-primary-blue); background-color: #F0F7FF; }

        /* ================= 布局调整 (避让顶部导航栏) ================= */
        .block-container {
            padding-top: 80px !important; /* 关键：向下偏移，留出 Header 空间 */
            padding-bottom: 3rem !important;
            max-width: 1200px;
        }

        /* ================= 隐藏 Streamlit 原生元素 ================= */
        header[data-testid="stHeader"] { display: none !important; visibility: hidden !important; }
        [data-testid="stToolbar"] { display: none !important; visibility: hidden !important; height: 0 !important; }
        footer { display: none !important; }
        [data-testid="stStatusWidget"] { visibility: hidden !important; }

        /* ================= 组件风格微调 ================= */
        div.stButton > button {
            border: 1px solid #E6EBF5; color: var(--pc-text-main); background: white;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        }
        div.stButton > button:hover {
            border-color: var(--pc-primary-blue); color: var(--pc-primary-blue); background-color: #F0F7FF;
        }
        .summary-box {
            background-color: #FFFFFF; padding: 20px; border-radius: 8px;
            border: 1px solid #E6EBF5; border-left: 4px solid var(--pc-primary-blue); margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        }
        .tech-card {
            background-color: white; padding: 24px; border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02); margin-bottom: 20px;
            border: 1px solid #E6EBF5; transition: all 0.2s ease-in-out;
        }
        .tech-card:hover { transform: translateY(-2px); border-color: #B3C0D1; box-shadow: 0 8px 16px rgba(0,0,0,0.04); }
        .mini-insight {
            background-color: #F4F6F9; padding: 12px 16px; border-radius: 6px;
            font-size: 13px; color: var(--pc-text-sub); margin-top: 15px; border-left: 3px solid #909399;
        }
        .insight-box {
            background: white; padding: 24px; border-radius: 12px; position: relative;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02); border: 1px solid #E6EBF5;
        }
        .insight-box::before {
            content: ''; position: absolute; left: 0; top: 12px; bottom: 12px;
            width: 4px; background: linear-gradient(180deg, var(--pc-primary-blue) 0%, #00C853 100%);
            border-radius: 0 4px 4px 0;
        }
        .step-header {
            font-weight: 700; color: var(--pc-text-main); font-size: 16px; margin-top: 35px; 
            margin-bottom: 20px; display: flex; align-items: center;
        }
        .step-header::before {
            content: ''; display: inline-block; width: 4px; height: 18px;
            background: var(--pc-primary-blue); margin-right: 12px; border-radius: 2px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 配置读取 ---
try:
    FIXED_API_KEY = st.secrets["GENAI_API_KEY"]
except:
    FIXED_API_KEY = ""

FIXED_FILE_NAME = "hcmdata.xlsx" 
LOGO_FILE = "logo.png"

PREVIEW_ROW_LIMIT = 500
EXPORT_ROW_LIMIT = 5000   

# ================= 2. 核心逻辑函数 =================

@st.cache_resource
def get_client():
    if not FIXED_API_KEY: return None
    try:
        return genai.Client(api_key=FIXED_API_KEY, http_options={'api_version': 'v1beta'})
    except Exception as e:
        st.error(f"SDK 初始化失败: {e}")
        return None

def safe_generate_content(client, model_name, contents, config=None, retries=3):
    base_delay = 5 
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
                    time.sleep(base_delay * (2 ** i))
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

# ================= 3. 页面渲染函数 =================

def render_header_nav():
    logo_b64 = ""
    # ⚠️ 请确保你的 Logo 文件名为 logo.png，并且在项目根目录下
    if os.path.exists(LOGO_FILE):
        with open(LOGO_FILE, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
    
    logo_img_tag = f'<img src="data:image/png;base64,{logo_b64}" class="nav-logo-img">' if logo_b64 else ""
    user_initials = "PRO"

    # 关键修改：HTML 标签必须顶格写，不能有缩进，否则会被当成代码块显示
    st.markdown(f"""
<div class="fixed-header-container">
    <div class="nav-left">
        {logo_img_tag}
        <div class="nav-brand-text">医药魔方</div>
    </div>
    
    <div class="nav-center">
        <div class="nav-item">HCM</div> 
        <div class="nav-item active">ChatBI</div>
    </div>
    
    <div class="nav-right">
        <div class="nav-avatar" title="当前用户">{user_initials}</div>
        <button class="nav-exit-btn" onclick="alert('Web应用中无法直接退出浏览器，您可以直接关闭标签页。')">退出</button>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= 4. 主程序执行 =================

# 1. 注入样式
inject_custom_css()

# 2. 渲染顶部导航
render_header_nav()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_draft" not in st.session_state:
    st.session_state.last_query_draft = ""
if "is_interrupted" not in st.session_state:
    st.session_state.is_interrupted = False

client = get_client()

# 安全检查
if not client:
    st.warning("⚠️ 未检测到有效 API Key。")
    st.info("请在 Streamlit 后台 Secrets 中配置 `GENAI_API_KEY`。")
    st.stop()

df = load_data()

if df is not None:
    time_context = analyze_time_structure(df)
    meta_data = build_metadata(df, time_context)
    
    # Sidebar: 仅保留控制台功能，Logo 已移至顶部
    with st.sidebar:
        st.markdown("### 🛠️ 控制台")
        st.caption("状态: 在线 (Active)")
        st.info(f"📊 总行数: {len(df):,}")
        st.info(f"📅 时间跨度: {time_context.get('min_q')} ~ {time_context.get('max_q')}")
        st.divider()
        if st.button("🗑️ 清空会话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_query_draft = ""
            st.session_state.is_interrupted = False
            st.rerun()

    # 聊天记录渲染
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
                            <div class="summary-title">⚡ 取数执行协议</div>
                            <ul class="summary-list">
                                <li><span class="summary-label">意图</span> {s.get('intent', '-')}</li>
                                <li><span class="summary-label">指标</span> {s.get('metrics', '-')}</li>
                                <li><span class="summary-label">逻辑</span> {s.get('logic', '-')}</li>
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
                            if len(data_payload) > 1: st.markdown(f"**📄 {table_name}**")
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

    # 引导卡片
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

    # 中止 & 输入
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
        if query_input := st.chat_input("🔎 请输入问题..."):
            st.session_state.last_query_draft = query_input
            st.session_state.messages.append({"role": "user", "type": "text", "content": query_input})
            st.rerun()

    # 核心逻辑
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and not st.session_state.is_interrupted:
        current_query = st.session_state.messages[-1]["content"]
        history_context_str = get_history_context(st.session_state.messages, turn_limit=3)
        stop_btn_placeholder = st.empty()
        
        if stop_btn_placeholder.button("⏹️ 中止生成", type="primary", use_container_width=True):
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
                    1. "simple": 简单取数、排序、排名、计算基础指标。
                    2. "analysis": 开放式问题，寻求洞察、原因分析、市场格局。
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
                    st.warning("⚠️ 当前提问不在数据覆盖范围内")
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": "抱歉，当前提问与数据内容无关。"})

                # ================= [Simple Mode] =================
                elif intent_type == 'simple':
                    with st.spinner("⚡ 正在解析意图并生成代码..."):
                        simple_prompt = f"""
                        你是一位 Pandas 数据处理专家。用户需求："{current_query}"
                        【元数据】{meta_data}
                        【历史记录】{history_context_str}
                        【时间上下文】MAT: {mat_list}, YTD: {ytd_list}
                        
                        【关键指令 - 这里的规则必须遵守】
                        1. **唯一数据源**：环境中只有 `df`。不要假设存在 `df_sales`, `df_hainan` 等变量。
                        2. **必须自行筛选**：如果需要特定维度（如海南、2023年），必须在代码中显式筛选。例如：`df_sub = df[df['省份']=='海南']`。
                        3. **结果赋值**：将最终结果字典赋值给 `results`。
                        4. **严禁绘图**。
                        
                        输出 JSON: {{ 
                            "summary": {{ "intent": "意图描述", "scope": "数据范围", "metrics": "指标", "logic": "计算逻辑" }}, 
                            "code": "df_sub = df[...]\nresults = {{'标题': df_sub}}" 
                        }}
                        """
                        simple_resp = safe_generate_content(
                            client, "gemini-3-pro-preview", simple_prompt, config=types.GenerateContentConfig(response_mime_type="application/json")
                        )
                        simple_json = json.loads(simple_resp.text)
                        
                        # 纯净的执行上下文，防止 AI 幻觉
                        execution_context = {
                            'df': df, 
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
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": "未提取到有效数据。"})

                # ================= [Analysis Mode] =================
                else:
                    with st.spinner("🧠 正在拆解问题..."):
                        prompt_plan = f"""
                        你是一位医药行业 BI 专家。请将问题："{current_query}" 拆解为 2-5 个分析角度。
                        结合时间动态（MAT/YTD）和竞争视角进行分析。
                        
                        【元数据】{meta_data}
                        【历史记录】{history_context_str}
                        【时间上下文】MAT: {mat_list}, YTD: {ytd_list}
                        
                        【关键指令 - 这里的规则必须遵守】
                        0. **数据源唯一入口**：环境中**只有**一个名为 `df` 的 Pandas DataFrame。
                        1. **严禁使用未定义变量**：绝对不要直接使用 `df_hainan`, `df_2023` 这种变量，除非你在代码中第一行先定义了它（例如：`df_sub = df[df['省份']=='海南']`）。
                        2. **严禁绘图**：不要生成 fig, plt, sns 代码。
                        3. **结果赋值**：最终结果必须赋值给变量 `result`。
                        4. **语言**：中文。
                        
                        输出 JSON: {{ "intent_analysis": "意图深度解析(Markdown)", "angles": [ {{"title": "分析角度标题", "description": "描述", "code": "df_sub = df[...]\nresult = df_sub..."}} ] }}
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
                                
                                try:
                                    # 纯净的执行上下文
                                    execution_context = {
                                        'df': df, 
                                        'pd': pd, 'np': np,
                                        'result': None,
                                        'current_mat': mat_list, 'mat_list': mat_list, 'prior_mat': mat_list_prior,
                                        'mat_list_prior': mat_list_prior, 'ytd_list': ytd_list, 'ytd_list_prior': ytd_list_prior
                                    }
                                    exec(angle['code'], execution_context)
                                    
                                    # 智能抓取结果
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
                                            # 修复：使用 to_string() 替代 to_markdown() 避免依赖报错
                                            mini_prompt = f"""
                                            对数据进行深度解读（200字内）。
                                            数据预览：\n{res_df.head(20).to_string()}
                                            要求：提炼趋势/异常，结合业务含义，语言专业。
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
                                    st.error(f"代码执行逻辑有误: {e}")

                        if angles_data:
                            st.markdown('<div class="step-header">3. 综合业务洞察</div>', unsafe_allow_html=True)
                            with st.spinner("🤖 生成综述..."):
                                all_findings = "\n".join([f"[{ad['title']}]: {ad['explanation']}" for ad in angles_data])
                                final_prompt = f"""
                                问题: "{current_query}"
                                各角度发现: {all_findings}
                                生成最终洞察 (Markdown)。严禁建议，仅陈述事实。
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
            except Exception as e:
                st.error(f"系统错误: {e}")
            finally:
                stop_btn_placeholder.empty()


