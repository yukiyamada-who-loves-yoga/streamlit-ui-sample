import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import random

st.set_page_config(
    page_title="リアルタイム監視",
    page_icon="📡",
    layout="wide"
)

st.title("📡 リアルタイム監視ダッシュボード")

# リアルタイムデータの生成
@st.cache_data
def generate_realtime_data():
    """リアルタイムモックデータを生成"""
    np.random.seed(42)
    
    # 現在時刻から過去24時間のデータ
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # システムメトリクス
    cpu_usage = []
    memory_usage = []
    network_traffic = []
    active_users = []
    error_rate = []
    
    for i, timestamp in enumerate(timestamps):
        # CPU使用率（時間帯による変動）
        hour = timestamp.hour
        base_cpu = 30 + 20 * np.sin(2 * np.pi * hour / 24)
        cpu = base_cpu + np.random.normal(0, 5)
        cpu_usage.append(max(0, min(100, cpu)))
        
        # メモリ使用率
        memory = 60 + np.random.normal(0, 10)
        memory_usage.append(max(0, min(100, memory)))
        
        # ネットワークトラフィック
        base_traffic = 100 + 50 * np.sin(2 * np.pi * hour / 24)
        traffic = base_traffic + np.random.normal(0, 20)
        network_traffic.append(max(0, traffic))
        
        # アクティブユーザー数
        base_users = 50 + 30 * np.sin(2 * np.pi * hour / 24)
        users = base_users + np.random.poisson(10)
        active_users.append(max(0, users))
        
        # エラー率
        base_error = 0.5 + 0.3 * np.sin(2 * np.pi * hour / 24)
        error = base_error + np.random.exponential(0.1)
        error_rate.append(max(0, min(5, error)))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'network_traffic': network_traffic,
        'active_users': active_users,
        'error_rate': error_rate
    })

# データの読み込み
data = generate_realtime_data()

# リアルタイム更新の設定
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# 自動更新の設定
auto_refresh = st.sidebar.checkbox("🔄 自動更新", value=False)
refresh_interval = st.sidebar.slider("更新間隔（秒）", 1, 60, 5)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# アラート設定
st.sidebar.markdown("### 🚨 アラート設定")
cpu_threshold = st.sidebar.slider("CPU使用率閾値 (%)", 50, 95, 80)
memory_threshold = st.sidebar.slider("メモリ使用率閾値 (%)", 50, 95, 85)
error_threshold = st.sidebar.slider("エラー率閾値 (%)", 0.1, 5.0, 2.0)

# 現在のシステム状態
st.subheader("🖥️ 現在のシステム状態")

col1, col2, col3, col4 = st.columns(4)

with col1:
    current_cpu = data['cpu_usage'].iloc[-1]
    cpu_color = "red" if current_cpu > cpu_threshold else "green"
    st.metric(
        "CPU使用率", 
        f"{current_cpu:.1f}%",
        delta=f"{data['cpu_usage'].iloc[-1] - data['cpu_usage'].iloc[-2]:.1f}%"
    )

with col2:
    current_memory = data['memory_usage'].iloc[-1]
    memory_color = "red" if current_memory > memory_threshold else "green"
    st.metric(
        "メモリ使用率", 
        f"{current_memory:.1f}%",
        delta=f"{data['memory_usage'].iloc[-1] - data['memory_usage'].iloc[-2]:.1f}%"
    )

with col3:
    current_traffic = data['network_traffic'].iloc[-1]
    st.metric(
        "ネットワークトラフィック", 
        f"{current_traffic:.0f} Mbps",
        delta=f"{data['network_traffic'].iloc[-1] - data['network_traffic'].iloc[-2]:.0f} Mbps"
    )

with col4:
    current_users = data['active_users'].iloc[-1]
    st.metric(
        "アクティブユーザー", 
        f"{current_users:.0f}人",
        delta=f"{data['active_users'].iloc[-1] - data['active_users'].iloc[-2]:.0f}人"
    )

# アラート表示
st.subheader("🚨 システムアラート")

alerts = []
if current_cpu > cpu_threshold:
    alerts.append(f"⚠️ CPU使用率が閾値（{cpu_threshold}%）を超過: {current_cpu:.1f}%")

if current_memory > memory_threshold:
    alerts.append(f"⚠️ メモリ使用率が閾値（{memory_threshold}%）を超過: {current_memory:.1f}%")

if data['error_rate'].iloc[-1] > error_threshold:
    alerts.append(f"⚠️ エラー率が閾値（{error_threshold}%）を超過: {data['error_rate'].iloc[-1]:.2f}%")

if alerts:
    for alert in alerts:
        st.error(alert)
else:
    st.success("✅ すべてのシステムが正常に動作しています")

# リアルタイムチャート
st.subheader("📈 リアルタイムメトリクス")

# サブプロットの作成
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('CPU使用率', 'メモリ使用率', 'ネットワークトラフィック', 'アクティブユーザー'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# CPU使用率
fig.add_trace(
    go.Scatter(
        x=data['timestamp'],
        y=data['cpu_usage'],
        mode='lines',
        name='CPU使用率',
        line=dict(color='red', width=2)
    ),
    row=1, col=1
)

# メモリ使用率
fig.add_trace(
    go.Scatter(
        x=data['timestamp'],
        y=data['memory_usage'],
        mode='lines',
        name='メモリ使用率',
        line=dict(color='blue', width=2)
    ),
    row=1, col=2
)

# ネットワークトラフィック
fig.add_trace(
    go.Scatter(
        x=data['timestamp'],
        y=data['network_traffic'],
        mode='lines',
        name='ネットワークトラフィック',
        line=dict(color='green', width=2)
    ),
    row=2, col=1
)

# アクティブユーザー
fig.add_trace(
    go.Scatter(
        x=data['timestamp'],
        y=data['active_users'],
        mode='lines',
        name='アクティブユーザー',
        line=dict(color='purple', width=2)
    ),
    row=2, col=2
)

# レイアウトの更新
fig.update_layout(
    height=600,
    showlegend=False,
    title_text="24時間のシステムメトリクス"
)

# 閾値ラインの追加
for i in range(1, 3):
    for j in range(1, 3):
        if i == 1 and j == 1:  # CPU使用率
            fig.add_hline(y=cpu_threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"閾値: {cpu_threshold}%", row=i, col=j)
        elif i == 1 and j == 2:  # メモリ使用率
            fig.add_hline(y=memory_threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"閾値: {memory_threshold}%", row=i, col=j)

st.plotly_chart(fig, use_container_width=True)

# システムログ
st.subheader("📋 システムログ")

# モックログデータ
log_entries = [
    {"timestamp": datetime.now() - timedelta(minutes=5), "level": "INFO", "message": "データベース接続が正常に確立されました"},
    {"timestamp": datetime.now() - timedelta(minutes=4), "level": "WARNING", "message": "メモリ使用率が70%を超過しました"},
    {"timestamp": datetime.now() - timedelta(minutes=3), "level": "INFO", "message": "新しいユーザーセッションが開始されました"},
    {"timestamp": datetime.now() - timedelta(minutes=2), "level": "ERROR", "message": "API呼び出しでタイムアウトが発生しました"},
    {"timestamp": datetime.now() - timedelta(minutes=1), "level": "INFO", "message": "バックアップ処理が完了しました"},
    {"timestamp": datetime.now(), "level": "INFO", "message": "システム監視が正常に動作しています"}
]

# ログレベルでフィルタリング
log_level = st.selectbox("ログレベルでフィルタリング", ["すべて", "INFO", "WARNING", "ERROR"])

filtered_logs = log_entries
if log_level != "すべて":
    filtered_logs = [log for log in log_entries if log["level"] == log_level]

# ログテーブルの表示
log_df = pd.DataFrame(filtered_logs)
if not log_df.empty:
    for _, log in log_df.iterrows():
        if log["level"] == "ERROR":
            st.error(f"{log['timestamp'].strftime('%H:%M:%S')} - {log['level']}: {log['message']}")
        elif log["level"] == "WARNING":
            st.warning(f"{log['timestamp'].strftime('%H:%M:%S')} - {log['level']}: {log['message']}")
        else:
            st.info(f"{log['timestamp'].strftime('%H:%M:%S')} - {log['level']}: {log['message']}")

# パフォーマンス統計
st.subheader("📊 パフォーマンス統計")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("平均CPU使用率", f"{data['cpu_usage'].mean():.1f}%")
    st.metric("最大CPU使用率", f"{data['cpu_usage'].max():.1f}%")

with col2:
    st.metric("平均メモリ使用率", f"{data['memory_usage'].mean():.1f}%")
    st.metric("最大メモリ使用率", f"{data['memory_usage'].max():.1f}%")

with col3:
    st.metric("平均エラー率", f"{data['error_rate'].mean():.2f}%")
    st.metric("最大エラー率", f"{data['error_rate'].max():.2f}%")

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📡 リアルタイム監視ダッシュボード | 最終更新: {}</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True) 