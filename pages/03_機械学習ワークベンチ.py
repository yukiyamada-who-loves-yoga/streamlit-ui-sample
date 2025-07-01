import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        mean_squared_error, r2_score, classification_report, confusion_matrix
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("scikit-learnのインポートに失敗しました。機械学習機能は利用できません。")
import joblib
import io
import base64
from datetime import datetime

st.set_page_config(
    page_title="機械学習ワークベンチ",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 機械学習ワークベンチ")

# モックデータの生成
@st.cache_data
def generate_ml_data():
    """機械学習用のモックデータを生成"""
    np.random.seed(42)
    
    # 分類問題用データ
    n_samples = 1000
    
    # 特徴量
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    feature3 = np.random.normal(0, 1, n_samples)
    feature4 = np.random.normal(0, 1, n_samples)
    feature5 = np.random.normal(0, 1, n_samples)
    
    # 分類ターゲット（2クラス）
    # 非線形な決定境界を作成
    target_classification = ((feature1**2 + feature2**2 > 2) & 
                           (feature3 > 0) & 
                           (feature4 + feature5 > 0)).astype(int)
    
    # 回帰ターゲット
    target_regression = (2 * feature1 + 3 * feature2 - 1.5 * feature3 + 
                        0.5 * feature4 + 1.2 * feature5 + np.random.normal(0, 0.5, n_samples))
    
    # クラスタリング用データ
    cluster_data = np.column_stack([
        np.random.normal(0, 1, n_samples),
        np.random.normal(0, 1, n_samples),
        np.random.normal(0, 1, n_samples)
    ])
    
    # カテゴリ特徴量
    categories = np.random.choice(['A', 'B', 'C'], n_samples)
    
    return {
        'classification': pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4': feature4,
            'feature5': feature5,
            'category': categories,
            'target': target_classification
        }),
        'regression': pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4': feature4,
            'feature5': feature5,
            'category': categories,
            'target': target_regression
        }),
        'clustering': pd.DataFrame({
            'feature1': cluster_data[:, 0],
            'feature2': cluster_data[:, 1],
            'feature3': cluster_data[:, 2]
        })
    }

# データの読み込み
data = generate_ml_data()

# サイドバー - タスク選択
st.sidebar.title("🎯 機械学習タスク")

if not SKLEARN_AVAILABLE:
    st.error("⚠️ scikit-learnが利用できません。機械学習機能は無効です。")
    st.stop()

task = st.sidebar.selectbox(
    "タスクを選択",
    ["分類問題", "回帰問題", "クラスタリング", "特徴量エンジニアリング"]
)

if task == "分類問題":
    st.subheader("🔍 分類問題")
    
    df = data['classification']
    
    # データ概要
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("サンプル数", len(df))
    with col2:
        st.metric("特徴量数", len(df.columns) - 1)
    with col3:
        st.metric("クラス数", df['target'].nunique())
    
    # データプレビュー
    st.subheader("📋 データプレビュー")
    st.dataframe(df.head(10), use_container_width=True)
    
    # ターゲット分布
    col1, col2 = st.columns(2)
    
    with col1:
        target_counts = df['target'].value_counts()
        fig_target = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title="ターゲット分布"
        )
        st.plotly_chart(fig_target, use_container_width=True)
    
    with col2:
        # 特徴量の分布
        selected_feature = st.selectbox("特徴量を選択", ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        
        fig_dist = px.histogram(
            df,
            x=selected_feature,
            color='target',
            title=f"{selected_feature}の分布（ターゲット別）",
            barmode='overlay'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # モデル選択
    st.subheader("🤖 モデル選択")
    
    model_type = st.selectbox(
        "モデルを選択",
        ["ランダムフォレスト", "ロジスティック回帰", "SVM"]
    )
    
    # ハイパーパラメータ設定
    st.subheader("⚙️ ハイパーパラメータ設定")
    
    if model_type == "ランダムフォレスト":
        n_estimators = st.slider("n_estimators", 10, 200, 100)
        max_depth = st.slider("max_depth", 3, 20, 10)
        min_samples_split = st.slider("min_samples_split", 2, 20, 2)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    
    elif model_type == "ロジスティック回帰":
        C = st.slider("C (正則化強度)", 0.1, 10.0, 1.0)
        max_iter = st.slider("max_iter", 100, 1000, 100)
        
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    
    else:  # SVM
        C = st.slider("C (正則化強度)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("kernel", ['rbf', 'linear', 'poly'])
        
        model = SVC(C=C, kernel=kernel, random_state=42)
    
    # データ前処理
    st.subheader("🔧 データ前処理")
    
    # カテゴリ変数のエンコーディング
    if 'category' in df.columns:
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['category'])
        features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'category_encoded']
    else:
        features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    
    X = df[features]
    y = df['target']
    
    # データ分割
    test_size = st.slider("テストデータの割合", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # スケーリング
    use_scaling = st.checkbox("特徴量のスケーリング", value=True)
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        X_train_final = X_train
        X_test_final = X_test
    
    # モデル訓練
    if st.button("🚀 モデルを訓練"):
        with st.spinner("モデルを訓練中..."):
            # モデル訓練
            model.fit(X_train_final, y_train)
            
            # 予測
            y_pred = model.predict(X_test_final)
            y_pred_proba = model.predict_proba(X_test_final)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 評価指標
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # 結果表示
            st.subheader("📊 評価結果")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("正解率", f"{accuracy:.3f}")
            with col2:
                st.metric("適合率", f"{precision:.3f}")
            with col3:
                st.metric("再現率", f"{recall:.3f}")
            with col4:
                st.metric("F1スコア", f"{f1:.3f}")
            
            # 混同行列
            st.subheader("📈 混同行列")
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="混同行列",
                labels=dict(x="予測", y="実際")
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # 分類レポート
            st.subheader("📋 詳細レポート")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # 特徴量重要度（ランダムフォレストの場合）
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 特徴量重要度")
                
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig_importance = px.barh(
                    importance_df,
                    x='importance',
                    y='feature',
                    title="特徴量重要度"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # モデル保存
            st.subheader("💾 モデル保存")
            if st.button("モデルを保存"):
                model_filename = f"classification_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                joblib.dump(model, model_filename)
                st.success(f"モデルを {model_filename} として保存しました")

elif task == "回帰問題":
    st.subheader("📈 回帰問題")
    
    df = data['regression']
    
    # データ概要
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("サンプル数", len(df))
    with col2:
        st.metric("特徴量数", len(df.columns) - 1)
    with col3:
        st.metric("ターゲット範囲", f"{df['target'].min():.2f} - {df['target'].max():.2f}")
    
    # データプレビュー
    st.subheader("📋 データプレビュー")
    st.dataframe(df.head(10), use_container_width=True)
    
    # ターゲット分布
    col1, col2 = st.columns(2)
    
    with col1:
        fig_target = px.histogram(
            df,
            x='target',
            title="ターゲット分布"
        )
        st.plotly_chart(fig_target, use_container_width=True)
    
    with col2:
        # 特徴量とターゲットの関係
        selected_feature = st.selectbox("特徴量を選択", ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        
        fig_scatter = px.scatter(
            df,
            x=selected_feature,
            y='target',
            title=f"{selected_feature} vs ターゲット"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # モデル選択
    st.subheader("🤖 モデル選択")
    
    model_type = st.selectbox(
        "モデルを選択",
        ["ランダムフォレスト", "線形回帰", "SVR"]
    )
    
    # ハイパーパラメータ設定
    st.subheader("⚙️ ハイパーパラメータ設定")
    
    if model_type == "ランダムフォレスト":
        n_estimators = st.slider("n_estimators", 10, 200, 100)
        max_depth = st.slider("max_depth", 3, 20, 10)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    
    elif model_type == "線形回帰":
        model = LinearRegression()
    
    else:  # SVR
        C = st.slider("C (正則化強度)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("kernel", ['rbf', 'linear', 'poly'])
        
        model = SVR(C=C, kernel=kernel)
    
    # データ前処理
    st.subheader("🔧 データ前処理")
    
    # カテゴリ変数のエンコーディング
    if 'category' in df.columns:
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['category'])
        features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'category_encoded']
    else:
        features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    
    X = df[features]
    y = df['target']
    
    # データ分割
    test_size = st.slider("テストデータの割合", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # スケーリング
    use_scaling = st.checkbox("特徴量のスケーリング", value=True)
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        X_train_final = X_train
        X_test_final = X_test
    
    # モデル訓練
    if st.button("🚀 モデルを訓練"):
        with st.spinner("モデルを訓練中..."):
            # モデル訓練
            model.fit(X_train_final, y_train)
            
            # 予測
            y_pred = model.predict(X_test_final)
            
            # 評価指標
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # 結果表示
            st.subheader("📊 評価結果")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MSE", f"{mse:.3f}")
            with col2:
                st.metric("RMSE", f"{rmse:.3f}")
            with col3:
                st.metric("R²", f"{r2:.3f}")
            
            # 予測 vs 実際
            st.subheader("📈 予測 vs 実際")
            
            fig_pred = px.scatter(
                x=y_test,
                y=y_pred,
                title="予測 vs 実際の値",
                labels={'x': '実際の値', 'y': '予測値'}
            )
            fig_pred.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()], 
                y=[y_test.min(), y_test.max()], 
                mode='lines', 
                name='完全予測',
                line=dict(dash='dash')
            ))
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # 残差プロット
            residuals = y_test - y_pred
            
            fig_residual = px.scatter(
                x=y_pred,
                y=residuals,
                title="残差プロット",
                labels={'x': '予測値', 'y': '残差'}
            )
            fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residual, use_container_width=True)
            
            # 特徴量重要度（ランダムフォレストの場合）
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 特徴量重要度")
                
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig_importance = px.barh(
                    importance_df,
                    x='importance',
                    y='feature',
                    title="特徴量重要度"
                )
                st.plotly_chart(fig_importance, use_container_width=True)

elif task == "クラスタリング":
    st.subheader("🎯 クラスタリング")
    
    df = data['clustering']
    
    # データ概要
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("サンプル数", len(df))
    with col2:
        st.metric("特徴量数", len(df.columns))
    with col3:
        st.metric("データ次元", len(df.columns))
    
    # データプレビュー
    st.subheader("📋 データプレビュー")
    st.dataframe(df.head(10), use_container_width=True)
    
    # データ可視化
    st.subheader("📈 データ可視化")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter = px.scatter(
            df,
            x='feature1',
            y='feature2',
            title="Feature1 vs Feature2"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        fig_3d = px.scatter_3d(
            df,
            x='feature1',
            y='feature2',
            z='feature3',
            title="3D散布図"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # クラスタリング設定
    st.subheader("🤖 クラスタリング設定")
    
    n_clusters = st.slider("クラスタ数", 2, 10, 3)
    
    # K-meansクラスタリング
    if st.button("🚀 クラスタリング実行"):
        with st.spinner("クラスタリング中..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(df)
            
            # 結果表示
            st.subheader("📊 クラスタリング結果")
            
            # クラスタ分布
            cluster_counts = df['cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cluster_dist = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    title="クラスタ分布"
                )
                st.plotly_chart(fig_cluster_dist, use_container_width=True)
            
            with col2:
                fig_cluster_scatter = px.scatter(
                    df,
                    x='feature1',
                    y='feature2',
                    color='cluster',
                    title="クラスタリング結果（2D）"
                )
                st.plotly_chart(fig_cluster_scatter, use_container_width=True)
            
            # 3Dクラスタリング結果
            fig_cluster_3d = px.scatter_3d(
                df,
                x='feature1',
                y='feature2',
                z='feature3',
                color='cluster',
                title="クラスタリング結果（3D）"
            )
            st.plotly_chart(fig_cluster_3d, use_container_width=True)
            
            # クラスタ統計
            st.subheader("📋 クラスタ統計")
            cluster_stats = df.groupby('cluster').agg(['mean', 'std']).round(3)
            st.dataframe(cluster_stats, use_container_width=True)

elif task == "特徴量エンジニアリング":
    st.subheader("🔧 特徴量エンジニアリング")
    
    df = data['classification'].copy()
    
    st.subheader("📋 元データ")
    st.dataframe(df.head(10), use_container_width=True)
    
    # 特徴量エンジニアリング手法
    st.subheader("🛠️ 特徴量エンジニアリング手法")
    
    # 多項式特徴量
    if st.checkbox("多項式特徴量を追加"):
        df['feature1_squared'] = df['feature1'] ** 2
        df['feature2_squared'] = df['feature2'] ** 2
        df['feature1_feature2'] = df['feature1'] * df['feature2']
        st.success("多項式特徴量を追加しました")
    
    # 統計特徴量
    if st.checkbox("統計特徴量を追加"):
        df['feature_mean'] = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']].mean(axis=1)
        df['feature_std'] = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']].std(axis=1)
        df['feature_max'] = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']].max(axis=1)
        st.success("統計特徴量を追加しました")
    
    # カテゴリ変数のエンコーディング
    if st.checkbox("カテゴリ変数をエンコード"):
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['category'])
        st.success("カテゴリ変数をエンコードしました")
    
    # 特徴量選択
    st.subheader("🎯 特徴量選択")
    
    # 数値特徴量のみ選択
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_features:
        numeric_features.remove('target')
    
    selected_features = st.multiselect(
        "使用する特徴量を選択",
        numeric_features,
        default=numeric_features
    )
    
    if selected_features:
        X = df[selected_features]
        y = df['target']
        
        # 相関分析
        st.subheader("📊 特徴量相関分析")
        
        correlation_matrix = X.corr()
        fig_corr = px.imshow(
            correlation_matrix,
            title="特徴量相関行列",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # PCA分析
        st.subheader("📉 PCA分析")
        
        n_components = st.slider("主成分数", 2, min(len(selected_features), 10), 3)
        
        if st.button("PCA実行"):
            with st.spinner("PCA実行中..."):
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X)
                
                # 説明分散比
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_variance = px.bar(
                        x=list(range(1, n_components + 1)),
                        y=explained_variance_ratio,
                        title="各主成分の説明分散比"
                    )
                    st.plotly_chart(fig_variance, use_container_width=True)
                
                with col2:
                    fig_cumulative = px.line(
                        x=list(range(1, n_components + 1)),
                        y=cumulative_variance_ratio,
                        title="累積説明分散比"
                    )
                    st.plotly_chart(fig_cumulative, use_container_width=True)
                
                # PCA結果の可視化
                if n_components >= 2:
                    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
                    pca_df['target'] = y
                    
                    fig_pca = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color='target',
                        title="PCA結果（PC1 vs PC2）"
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                
                # 特徴量の寄与度
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'PC1_contribution': np.abs(pca.components_[0]),
                    'PC2_contribution': np.abs(pca.components_[1]) if n_components > 1 else 0
                })
                
                st.subheader("🎯 特徴量の主成分への寄与度")
                st.dataframe(feature_importance, use_container_width=True)

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 機械学習ワークベンチ | Streamlit 1.46.1 | 2025年7月1日</p>
</div>
""", unsafe_allow_html=True) 