import streamlit as st
import openai
from datetime import datetime
import json

st.set_page_config(
    page_title="AIチャットボット",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AIチャットボット")

# セッション状態の初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# サイドバー設定
st.sidebar.markdown("### ⚙️ 設定")

# APIキー入力
api_key = st.sidebar.text_input(
    "OpenAI APIキー",
    type="password",
    help="OpenAI APIキーを入力してください（必須）",
    value=st.session_state.openai_api_key
)

if api_key != st.session_state.openai_api_key:
    st.session_state.openai_api_key = api_key
    st.session_state.chat_history = []
    st.rerun()

# モデル選択
model = st.sidebar.selectbox(
    "使用モデル",
    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    index=0,
    help="使用するOpenAIモデルを選択してください"
)

# 温度設定
temperature = st.sidebar.slider(
    "創造性（温度）",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.1,
    help="値が高いほど創造的な回答になります"
)

# 最大トークン数
max_tokens = st.sidebar.slider(
    "最大トークン数",
    min_value=100,
    max_value=4000,
    value=1000,
    step=100,
    help="回答の最大長を設定します"
)

# チャット履歴クリアボタン
if st.sidebar.button("🗑️ チャット履歴をクリア"):
    st.session_state.chat_history = []
    st.rerun()

# メインコンテンツ
if not api_key:
    st.warning("⚠️ OpenAI APIキーを入力してください")
    st.info("""
    ### 使用方法
    1. サイドバーでOpenAI APIキーを入力
    2. 使用モデルとパラメータを設定
    3. チャットボックスに質問を入力
    4. AIアシスタントと対話を開始
    
    ### 対応機能
    - 一般的な質問への回答
    - プログラミングのサポート
    - データ分析のアドバイス
    - ビジネス戦略の提案
    - 学習支援
    """)
else:
    # チャット履歴の表示
    st.markdown("### 💬 チャット履歴")
    
    # チャット履歴を表示
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # チャット入力
    if prompt := st.chat_input("メッセージを入力してください..."):
        # ユーザーメッセージを履歴に追加
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.chat_history.append(user_message)
        
        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI応答を生成
        with st.chat_message("assistant"):
            with st.spinner("🤖 AIが考え中..."):
                try:
                    # OpenAIクライアントの設定
                    client = openai.OpenAI(api_key=api_key)
                    
                    # システムプロンプトの設定
                    system_prompt = """あなたは親切で知識豊富なAIアシスタントです。
以下の点に注意して回答してください：

1. 日本語で丁寧に回答する
2. 質問者のレベルに合わせて説明する
3. 必要に応じて具体例を挙げる
4. プログラミングやデータ分析の質問には実用的なアドバイスを提供する
5. ビジネス関連の質問には戦略的な視点で回答する
6. 学習支援では段階的な説明を心がける
7. 不明な点があれば質問して明確にする

常に建設的で役立つ回答を心がけてください。"""
                    
                    # メッセージ履歴の準備
                    messages = [{"role": "system", "content": system_prompt}]
                    
                    # 過去の会話履歴を追加（最新の10件まで）
                    recent_history = st.session_state.chat_history[-10:]
                    for msg in recent_history:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    # OpenAI APIを呼び出し
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # AI応答を取得
                    ai_response = response.choices[0].message.content
                    
                    # AI応答を履歴に追加
                    ai_message = {
                        "role": "assistant",
                        "content": ai_response,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_history.append(ai_message)
                    
                    # AI応答を表示
                    st.markdown(ai_response)
                    
                except openai.AuthenticationError:
                    st.error("❌ APIキーが無効です。正しいAPIキーを入力してください。")
                except openai.RateLimitError:
                    st.error("❌ APIレート制限に達しました。しばらく待ってから再試行してください。")
                except openai.APIError as e:
                    st.error(f"❌ APIエラーが発生しました: {str(e)}")
                except Exception as e:
                    st.error(f"❌ 予期しないエラーが発生しました: {str(e)}")
    
    # チャット履歴の統計情報
    if st.session_state.chat_history:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_messages = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
            st.metric("ユーザーメッセージ数", user_messages)
        
        with col2:
            ai_messages = len([msg for msg in st.session_state.chat_history if msg["role"] == "assistant"])
            st.metric("AI応答数", ai_messages)
        
        with col3:
            total_messages = len(st.session_state.chat_history)
            st.metric("総メッセージ数", total_messages)
    
    # よくある質問の例
    st.markdown("---")
    st.markdown("### 💡 よくある質問の例")
    
    example_questions = [
        "Pythonでデータ分析を始めるにはどうすればいいですか？",
        "Streamlitアプリのパフォーマンスを改善する方法を教えてください",
        "機械学習モデルの精度を向上させるコツはありますか？",
        "ビジネスデータの可視化で効果的なグラフの選び方を教えてください",
        "データベース設計のベストプラクティスを教えてください"
    ]
    
    cols = st.columns(len(example_questions))
    for i, question in enumerate(example_questions):
        with cols[i]:
            if st.button(f"例{i+1}", key=f"example_{i}"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question,
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()

# フッター情報
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>🤖 AIチャットボット - 質問や疑問を気軽に相談してください</p>
    <p>Powered by OpenAI GPT Models | Streamlit 1.46.1</p>
</div>
""", unsafe_allow_html=True) 