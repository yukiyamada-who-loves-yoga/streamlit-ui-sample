import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64

st.set_page_config(
    page_title="データエクスプローラー",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 データエクスプローラー")

# モックデータの生成
@st.cache_data
def generate_explorer_data():
    """データエクスプローラー用のモックデータを生成"""
    np.random.seed(42)
    
    # 顧客データ
    n_customers = 1000
    customer_data = []
    
    for i in range(n_customers):
        customer_data.append({
            'customer_id': f'CUST_{i:04d}',
            'name': f'顧客{i+1}',
            'age': np.random.randint(18, 80),
            'gender': np.random.choice(['男性', '女性']),
            'income': np.random.normal(50000, 20000),
            'education': np.random.choice(['高校', '専門学校', '大学', '大学院']),
            'occupation': np.random.choice(['会社員', '自営業', '学生', '主婦', 'フリーランス']),
            'region': np.random.choice(['東京', '大阪', '名古屋', '福岡', '札幌', '仙台', '広島']),
            'satisfaction_score': np.random.randint(1, 11),
            'purchase_frequency': np.random.poisson(3),
            'total_spent': np.random.exponential(1000),
            'last_purchase_date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
            'membership_level': np.random.choice(['ブロンズ', 'シルバー', 'ゴールド', 'プラチナ']),
            'is_active': np.random.choice([True, False], p=[0.8, 0.2])
        })
    
    # 商品データ
    n_products = 500
    product_data = []
    
    categories = ['電子機器', '衣類', '食品', '書籍', 'スポーツ用品', '家具', '化粧品']
    brands = ['ブランドA', 'ブランドB', 'ブランドC', 'ブランドD', 'ブランドE']
    
    for i in range(n_products):
        category = np.random.choice(categories)
        brand = np.random.choice(brands)
        
        product_data.append({
            'product_id': f'PROD_{i:04d}',
            'name': f'{category}商品{i+1}',
            'category': category,
            'brand': brand,
            'price': np.random.uniform(100, 10000),
            'cost': np.random.uniform(50, 8000),
            'stock_quantity': np.random.randint(0, 1000),
            'rating': np.random.uniform(1, 5),
            'review_count': np.random.randint(0, 500),
            'is_featured': np.random.choice([True, False], p=[0.1, 0.9]),
            'created_date': datetime.now() - timedelta(days=np.random.randint(1, 1000))
        })
    
    # 売上データ
    n_sales = 2000
    sales_data = []
    
    for i in range(n_sales):
        customer = np.random.choice(customer_data)
        product = np.random.choice(product_data)
        
        sales_data.append({
            'sale_id': f'SALE_{i:04d}',
            'customer_id': customer['customer_id'],
            'product_id': product['product_id'],
            'quantity': np.random.randint(1, 10),
            'unit_price': product['price'],
            'total_amount': product['price'] * np.random.randint(1, 10),
            'sale_date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
            'payment_method': np.random.choice(['クレジットカード', '銀行振込', 'コンビニ決済', '電子マネー']),
            'delivery_status': np.random.choice(['配送中', '配達完了', '返品', 'キャンセル']),
            'region': customer['region']
        })
    
    return {
        'customers': pd.DataFrame(customer_data),
        'products': pd.DataFrame(product_data),
        'sales': pd.DataFrame(sales_data)
    }

# データの読み込み
data = generate_explorer_data()

# サイドバー - データセット選択
st.sidebar.title("📊 データセット選択")
dataset = st.sidebar.selectbox(
    "分析するデータセットを選択",
    ["顧客データ", "商品データ", "売上データ"]
)

# データセットの表示
if dataset == "顧客データ":
    df = data['customers']
    st.subheader("👥 顧客データ")
elif dataset == "商品データ":
    df = data['products']
    st.subheader("🛍️ 商品データ")
else:
    df = data['sales']
    st.subheader("💰 売上データ")

# データ概要
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("行数", f"{len(df):,}")
with col2:
    st.metric("列数", f"{len(df.columns)}")
with col3:
    st.metric("メモリ使用量", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# データプレビュー
st.subheader("📋 データプレビュー")
st.dataframe(df.head(10), use_container_width=True)

# データ型情報
st.subheader("🔧 データ型情報")
type_info = pd.DataFrame({
    '列名': df.columns,
    'データ型': df.dtypes,
    '非欠損値数': df.count(),
    '欠損値数': df.isnull().sum(),
    '欠損率(%)': (df.isnull().sum() / len(df) * 100).round(2)
})
st.dataframe(type_info, use_container_width=True)

# 基本統計
st.subheader("📊 基本統計")
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
else:
    st.info("数値型の列が見つかりません")

# データ可視化
st.subheader("📈 データ可視化")

# 可視化オプション
viz_type = st.selectbox(
    "可視化タイプを選択",
    ["分布図", "相関分析", "時系列分析", "カテゴリ分析", "散布図"]
)

if viz_type == "分布図":
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("分析する列を選択", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ヒストグラム
            fig_hist = px.histogram(
                df, 
                x=selected_col,
                nbins=30,
                title=f"{selected_col}の分布"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # ボックスプロット
            fig_box = px.box(
                df,
                y=selected_col,
                title=f"{selected_col}のボックスプロット"
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    else:
        st.warning("数値型の列が見つかりません")

elif viz_type == "相関分析":
    if len(numeric_cols) > 1:
        # 相関行列
        correlation_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title="相関行列",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # 相関係数の詳細
        st.subheader("相関係数の詳細")
        corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = correlation_matrix.iloc[i, j]
                corr_pairs.append({
                    '変数1': numeric_cols[i],
                    '変数2': numeric_cols[j],
                    '相関係数': corr_value
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.sort_values('相関係数', key=abs, ascending=False)
        st.dataframe(corr_df, use_container_width=True)
    
    else:
        st.warning("相関分析には2つ以上の数値型列が必要です")

elif viz_type == "時系列分析":
    # 日付列を探す
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                continue
    
    if date_cols:
        selected_date_col = st.selectbox("日付列を選択", date_cols)
        
        # 日付列を変換
        df_copy = df.copy()
        df_copy[selected_date_col] = pd.to_datetime(df_copy[selected_date_col])
        
        # 時系列データの集計
        if len(numeric_cols) > 0:
            selected_value_col = st.selectbox("集計する値の列を選択", numeric_cols)
            
            # 日付でグループ化
            time_series = df_copy.groupby(df_copy[selected_date_col].dt.date)[selected_value_col].sum().reset_index()
            time_series[selected_date_col] = pd.to_datetime(time_series[selected_date_col])
            
            fig_time = px.line(
                time_series,
                x=selected_date_col,
                y=selected_value_col,
                title=f"{selected_value_col}の時系列推移"
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.warning("数値型の列が見つかりません")
    else:
        st.warning("日付型の列が見つかりません")

elif viz_type == "カテゴリ分析":
    # カテゴリ列を探す
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        selected_cat_col = st.selectbox("カテゴリ列を選択", categorical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # カテゴリ別件数
            cat_counts = df[selected_cat_col].value_counts()
            fig_bar = px.bar(
                x=cat_counts.index,
                y=cat_counts.values,
                title=f"{selected_cat_col}の分布"
            )
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # 円グラフ
            fig_pie = px.pie(
                values=cat_counts.values,
                names=cat_counts.index,
                title=f"{selected_cat_col}の割合"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.warning("カテゴリ型の列が見つかりません")

elif viz_type == "散布図":
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X軸の列を選択", numeric_cols, key="x")
        
        with col2:
            y_col = st.selectbox("Y軸の列を選択", numeric_cols, key="y")
        
        if x_col != y_col:
            fig_scatter = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}",
                trendline="ols"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("X軸とY軸は異なる列を選択してください")
    
    else:
        st.warning("散布図には2つ以上の数値型列が必要です")

# データフィルタリング
st.subheader("🔍 データフィルタリング")

# フィルター条件の設定
filter_conditions = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        # 数値列の場合
        min_val = df[col].min()
        max_val = df[col].max()
        selected_range = st.slider(
            f"{col}の範囲",
            min_value=float(min_val),
            max_value=float(max_val),
            value=(float(min_val), float(max_val))
        )
        if selected_range != (min_val, max_val):
            filter_conditions.append(f"({col} >= {selected_range[0]}) & ({col} <= {selected_range[1]})")
    
    elif df[col].dtype == 'object':
        # カテゴリ列の場合
        unique_values = df[col].unique()
        if len(unique_values) <= 20:  # 選択肢が多すぎる場合はスキップ
            selected_values = st.multiselect(
                f"{col}を選択",
                options=unique_values,
                default=unique_values
            )
            if len(selected_values) != len(unique_values):
                filter_conditions.append(f"{col}.isin({selected_values})")

# フィルター適用
if filter_conditions:
    filter_query = " & ".join(filter_conditions)
    filtered_df = df.query(filter_query)
    st.success(f"フィルター適用後: {len(filtered_df):,}行（元のデータ: {len(df):,}行）")
    
    # フィルター適用後のデータプレビュー
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    # データエクスポート
    st.subheader("💾 データエクスポート")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("CSVとしてダウンロード"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="CSVファイルをダウンロード",
                data=csv,
                file_name=f"filtered_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Excelとしてダウンロード"):
            # Excelファイルの生成（簡易版）
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Data')
            output.seek(0)
            
            st.download_button(
                label="Excelファイルをダウンロード",
                data=output.getvalue(),
                file_name=f"filtered_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔍 データエクスプローラー | Streamlit 1.46.1 | 2025年7月1日</p>
</div>
""", unsafe_allow_html=True) 