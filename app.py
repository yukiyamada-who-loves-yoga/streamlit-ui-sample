import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
import json

# ページ設定
st.set_page_config(
    page_title="高機能データ分析ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "ダッシュボード"

# モックデータの生成
@st.cache_data
def generate_mock_data():
    """モックデータを生成する関数"""
    np.random.seed(42)
    
    # 日付データ
    dates = pd.date_range(start='2024-01-01', end='2025-06-30', freq='D')
    
    # 売上データ
    sales_data = []
    for date in dates:
        base_sales = 1000 + np.random.normal(0, 200)
        # 季節性を追加
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
        # 週末効果
        weekend_factor = 1.2 if date.weekday() >= 5 else 1.0
        sales = base_sales * seasonal_factor * weekend_factor + np.random.normal(0, 50)
        sales_data.append(max(0, sales))
    
    # 顧客データ
    customer_data = []
    for i in range(1000):
        customer_data.append({
            'customer_id': f'CUST_{i:04d}',
            'age': np.random.randint(18, 80),
            'income': np.random.normal(50000, 20000),
            'satisfaction_score': np.random.randint(1, 11),
            'purchase_frequency': np.random.poisson(3),
            'total_spent': np.random.exponential(1000),
            'region': np.random.choice(['東京', '大阪', '名古屋', '福岡', '札幌']),
            'gender': np.random.choice(['男性', '女性']),
            'membership_level': np.random.choice(['ブロンズ', 'シルバー', 'ゴールド', 'プラチナ'])
        })
    
    # 商品データ
    products = ['ノートPC', 'スマートフォン', 'タブレット', 'ヘッドフォン', 'キーボード', 
                'マウス', 'モニター', 'プリンター', 'スキャナー', '外付けHDD']
    
    product_data = []
    for i in range(1000):
        product_data.append({
            'order_id': f'ORD_{i:04d}',
            'product_name': np.random.choice(products),
            'quantity': np.random.randint(1, 10),
            'unit_price': np.random.uniform(1000, 100000),
            'date': np.random.choice(dates),
            'customer_id': f'CUST_{np.random.randint(0, 1000):04d}',
            'payment_method': np.random.choice(['クレジットカード', '銀行振込', 'コンビニ決済', '電子マネー']),
            'delivery_status': np.random.choice(['配送中', '配達完了', '返品', 'キャンセル'])
        })
    
    return {
        'sales_df': pd.DataFrame({'date': dates, 'sales': sales_data}),
        'customer_df': pd.DataFrame(customer_data),
        'product_df': pd.DataFrame(product_data)
    }

# データの読み込み
data = generate_mock_data()
st.session_state.data_loaded = True

# サイドバー
with st.sidebar:
    st.title("📊 ナビゲーション")
    
    page = st.selectbox(
        "ページを選択",
        ["ダッシュボード", "売上分析", "顧客分析", "商品分析", "予測分析", "AI アシスタント"],
        index=0
    )
    
    st.session_state.current_page = page
    
    st.markdown("---")
    st.markdown("### 📅 日付範囲")
    
    # 日付フィルター
    min_date = data['sales_df']['date'].min().date()
    max_date = data['sales_df']['date'].max().date()
    default_start = max(min_date, datetime.now().date() - timedelta(days=30))
    default_end = min(max_date, datetime.now().date())
    
    date_range = st.date_input(
        "分析期間を選択",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date
    )
    
    st.markdown("---")
    st.markdown("### 🎯 フィルター")
    
    # 地域フィルター
    regions = st.multiselect(
        "地域を選択",
        options=data['customer_df']['region'].unique(),
        default=data['customer_df']['region'].unique()
    )
    
    # 商品フィルター
    products = st.multiselect(
        "商品を選択",
        options=data['product_df']['product_name'].unique(),
        default=data['product_df']['product_name'].unique()
    )

# メインコンテンツ
if st.session_state.current_page == "ダッシュボード":
    st.markdown('<h1 class="main-header">📊 高機能データ分析ダッシュボード</h1>', unsafe_allow_html=True)
    
    # KPI メトリクス
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = data['sales_df']['sales'].sum()
        st.metric("総売上", f"¥{total_sales:,.0f}", f"{np.random.randint(-10, 20)}%")
    
    with col2:
        total_customers = len(data['customer_df'])
        st.metric("総顧客数", f"{total_customers:,}", f"{np.random.randint(-5, 15)}%")
    
    with col3:
        avg_satisfaction = data['customer_df']['satisfaction_score'].mean()
        st.metric("平均満足度", f"{avg_satisfaction:.1f}/10", f"{np.random.randint(-2, 5)}%")
    
    with col4:
        total_orders = len(data['product_df'])
        st.metric("総注文数", f"{total_orders:,}", f"{np.random.randint(-8, 12)}%")
    
    st.markdown("---")
    
    # チャートセクション
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 売上トレンド")
        
        # 売上トレンドチャート
        fig_sales = px.line(
            data['sales_df'], 
            x='date', 
            y='sales',
            title="日次売上推移"
        )
        fig_sales.update_layout(height=400)
        st.plotly_chart(fig_sales, use_container_width=True)
    
    with col2:
        st.subheader("👥 顧客分布")
        
        # 年齢分布
        fig_age = px.histogram(
            data['customer_df'],
            x='age',
            nbins=20,
            title="顧客年齢分布"
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    # 地域別売上
    st.subheader("🗺️ 地域別売上")
    region_sales = data['customer_df'].groupby('region')['total_spent'].sum().reset_index()
    
    fig_region = px.bar(
        region_sales,
        x='region',
        y='total_spent',
        title="地域別総売上"
    )
    fig_region.update_layout(height=400)
    st.plotly_chart(fig_region, use_container_width=True)

elif st.session_state.current_page == "売上分析":
    st.title("📈 売上分析")
    
    # 売上分析の詳細
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("月次売上")
        monthly_sales = data['sales_df'].set_index('date').resample('M')['sales'].sum().reset_index()
        
        fig_monthly = px.bar(
            monthly_sales,
            x='date',
            y='sales',
            title="月次売上推移"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        st.subheader("週次売上")
        weekly_sales = data['sales_df'].set_index('date').resample('W')['sales'].sum().reset_index()
        
        fig_weekly = px.line(
            weekly_sales,
            x='date',
            y='sales',
            title="週次売上推移"
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    # 売上統計
    st.subheader("📊 売上統計")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("平均日次売上", f"¥{data['sales_df']['sales'].mean():,.0f}")
    
    with col2:
        st.metric("最高日次売上", f"¥{data['sales_df']['sales'].max():,.0f}")
    
    with col3:
        st.metric("最低日次売上", f"¥{data['sales_df']['sales'].min():,.0f}")

elif st.session_state.current_page == "顧客分析":
    st.title("👥 顧客分析")
    
    # 顧客セグメンテーション
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("年齢層別分布")
        age_bins = [0, 25, 35, 45, 55, 65, 100]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        data['customer_df']['age_group'] = pd.cut(data['customer_df']['age'], bins=age_bins, labels=age_labels)
        
        age_dist = data['customer_df']['age_group'].value_counts()
        fig_age_group = px.pie(
            values=age_dist.values,
            names=age_dist.index,
            title="年齢層別顧客分布"
        )
        st.plotly_chart(fig_age_group, use_container_width=True)
    
    with col2:
        st.subheader("会員レベル別分布")
        membership_dist = data['customer_df']['membership_level'].value_counts()
        fig_membership = px.bar(
            x=membership_dist.index,
            y=membership_dist.values,
            title="会員レベル別顧客数"
        )
        st.plotly_chart(fig_membership, use_container_width=True)
    
    # 顧客満足度分析
    st.subheader("😊 顧客満足度分析")
    
    satisfaction_by_region = data['customer_df'].groupby('region')['satisfaction_score'].mean().reset_index()
    fig_satisfaction = px.bar(
        satisfaction_by_region,
        x='region',
        y='satisfaction_score',
        title="地域別平均満足度"
    )
    st.plotly_chart(fig_satisfaction, use_container_width=True)

elif st.session_state.current_page == "商品分析":
    st.title("🛍️ 商品分析")
    
    # 商品別売上
    product_sales = data['product_df'].groupby('product_name').agg({
        'quantity': 'sum',
        'unit_price': 'mean',
        'order_id': 'count'
    }).reset_index()
    product_sales['total_revenue'] = product_sales['quantity'] * product_sales['unit_price']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("商品別売上")
        fig_product_revenue = px.bar(
            product_sales,
            x='product_name',
            y='total_revenue',
            title="商品別総売上"
        )
        fig_product_revenue.update_xaxes(tickangle=45)
        st.plotly_chart(fig_product_revenue, use_container_width=True)
    
    with col2:
        st.subheader("商品別注文数")
        fig_product_orders = px.bar(
            product_sales,
            x='product_name',
            y='order_id',
            title="商品別注文数"
        )
        fig_product_orders.update_xaxes(tickangle=45)
        st.plotly_chart(fig_product_orders, use_container_width=True)
    
    # 支払い方法分析
    st.subheader("💳 支払い方法分析")
    payment_methods = data['product_df']['payment_method'].value_counts()
    
    fig_payment = px.pie(
        values=payment_methods.values,
        names=payment_methods.index,
        title="支払い方法別分布"
    )
    st.plotly_chart(fig_payment, use_container_width=True)

elif st.session_state.current_page == "予測分析":
    st.title("🔮 予測分析")
    
    if not SKLEARN_AVAILABLE:
        st.error("⚠️ scikit-learnが利用できません。予測分析機能は無効です。")
        st.info("scikit-learnをインストールしてから再度お試しください。")
        st.stop()
    
    # 機械学習モデルの準備
    st.subheader("📊 売上予測モデル")
    
    # 特徴量エンジニアリング
    sales_df = data['sales_df'].copy()
    sales_df['day_of_week'] = sales_df['date'].dt.dayofweek
    sales_df['month'] = sales_df['date'].dt.month
    sales_df['day_of_year'] = sales_df['date'].dt.dayofyear
    sales_df['is_weekend'] = sales_df['day_of_week'].isin([5, 6]).astype(int)
    
    # ラグ特徴量
    sales_df['sales_lag1'] = sales_df['sales'].shift(1)
    sales_df['sales_lag7'] = sales_df['sales'].shift(7)
    
    # 移動平均
    sales_df['sales_ma7'] = sales_df['sales'].rolling(window=7).mean()
    sales_df['sales_ma30'] = sales_df['sales'].rolling(window=30).mean()
    
    # 欠損値を削除
    sales_df = sales_df.dropna()
    
    # 特徴量とターゲット
    features = ['day_of_week', 'month', 'day_of_year', 'is_weekend', 
                'sales_lag1', 'sales_lag7', 'sales_ma7', 'sales_ma30']
    X = sales_df[features]
    y = sales_df['sales']
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # モデル訓練
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    
    # モデル評価
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R² スコア", f"{r2_score(y_test, y_pred):.3f}")
    
    with col2:
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    
    # 予測vs実際のプロット
    fig_pred = px.scatter(
        x=y_test,
        y=y_pred,
        title="予測 vs 実際の売上",
        labels={'x': '実際の売上', 'y': '予測売上'}
    )
    fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                  y=[y_test.min(), y_test.max()], 
                                  mode='lines', name='完全予測'))
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig_importance = px.barh(
        feature_importance,
        x='importance',
        y='feature',
        title="特徴量重要度"
    )
    st.plotly_chart(fig_importance, use_container_width=True)

elif st.session_state.current_page == "AI アシスタント":
    st.title("🤖 AI アシスタント")
    
    st.markdown("### 💬 データ分析アシスタント")
    st.markdown("データに関する質問をしてください。AIが分析結果を説明します。")
    
    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # ユーザー入力
    if prompt := st.chat_input("データについて質問してください..."):
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI応答の生成（モック）
        ai_response = generate_ai_response(prompt, data)
        
        # AIメッセージを追加
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.markdown(ai_response)

def generate_ai_response(prompt, data):
    """AI応答を生成する関数（モック）"""
    prompt_lower = prompt.lower()
    
    if "売上" in prompt_lower or "sales" in prompt_lower:
        total_sales = data['sales_df']['sales'].sum()
        avg_sales = data['sales_df']['sales'].mean()
        return f"""
        📊 **売上分析結果**
        
        - **総売上**: ¥{total_sales:,.0f}
        - **平均日次売上**: ¥{avg_sales:,.0f}
        - **最高売上日**: ¥{data['sales_df']['sales'].max():,.0f}
        - **最低売上日**: ¥{data['sales_df']['sales'].min():,.0f}
        
        売上は季節性があり、週末に若干の上昇が見られます。
        """
    
    elif "顧客" in prompt_lower or "customer" in prompt_lower:
        total_customers = len(data['customer_df'])
        avg_satisfaction = data['customer_df']['satisfaction_score'].mean()
        return f"""
        👥 **顧客分析結果**
        
        - **総顧客数**: {total_customers:,}人
        - **平均満足度**: {avg_satisfaction:.1f}/10
        - **平均年齢**: {data['customer_df']['age'].mean():.1f}歳
        - **平均所得**: ¥{data['customer_df']['income'].mean():,.0f}
        
        顧客の満足度は全体的に良好で、地域別では東京と大阪が高い傾向にあります。
        """
    
    elif "商品" in prompt_lower or "product" in prompt_lower:
        top_product = data['product_df']['product_name'].value_counts().index[0]
        return f"""
        🛍️ **商品分析結果**
        
        - **人気商品**: {top_product}
        - **総商品種類**: {data['product_df']['product_name'].nunique()}種類
        - **総注文数**: {len(data['product_df']):,}件
        - **平均単価**: ¥{data['product_df']['unit_price'].mean():,.0f}
        
        電子機器が人気で、特にノートPCとスマートフォンの売上が好調です。
        """
    
    else:
        return """
        🤖 **AIアシスタント**
        
        以下のような質問ができます：
        
        - 📊 売上について教えて
        - 👥 顧客分析をして
        - 🛍️ 商品の売れ筋は？
        - 📈 トレンド分析をして
        
        より具体的な質問をしていただけると、詳細な分析結果をお答えできます。
        """

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 高機能データ分析ダッシュボード | Streamlit 1.46.1 | 2025年7月1日</p>
</div>
""", unsafe_allow_html=True) 