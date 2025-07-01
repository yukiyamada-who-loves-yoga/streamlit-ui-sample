import streamlit as st
import openai
from datetime import datetime
import json
import time

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

# 文字数制限設定
st.sidebar.markdown("### 📝 文字数制限設定")
char_limit_enabled = st.sidebar.checkbox(
    "文字数制限を有効にする", 
    value=True, 
    help="入力と出力を指定文字数以内に制限します"
)

if char_limit_enabled:
    char_limit = st.sidebar.slider(
        "最大文字数",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="入力と出力の最大文字数を設定します"
    )
else:
    char_limit = 1000

# ストリーミング出力の設定
st.sidebar.markdown("### 📡 ストリーミング設定")
stream_enabled = st.sidebar.checkbox(
    "ストリーミング出力", 
    value=True, 
    help="リアルタイムでAI応答を表示します"
)

if stream_enabled:
    stream_speed = st.sidebar.slider(
        "表示速度",
        min_value=0.001,
        max_value=0.05,
        value=0.01,
        step=0.001,
        help="ストリーミング表示の速度を調整します"
    )
else:
    stream_speed = 0.01

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
    - リアルタイムストリーミング出力
    """)
else:
    # チャット履歴の表示
    st.markdown("### 💬 チャット履歴")
    
    # チャット履歴を表示
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # 文字数制限が有効な場合は表示内容を制限
            display_content = message["content"]
            if char_limit_enabled and len(display_content) > char_limit:
                display_content = display_content[:char_limit] + "..."
            st.markdown(display_content)
    
    # チャット入力
    if prompt := st.chat_input("メッセージを入力してください..."):
        # 文字数制限のチェック
        if char_limit_enabled and len(prompt) > char_limit:
            st.error(f"❌ 入力が長すぎます。{char_limit}文字以内で入力してください。（現在: {len(prompt)}文字）")
        else:
            # 文字数制限が有効な場合は、入力内容を制限
            if char_limit_enabled:
                prompt = prompt[:char_limit]
            
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
            try:
                # OpenAIクライアントの設定
                client = openai.OpenAI(api_key=api_key)
                
                # システムプロンプトの設定
                char_limit_instruction = f"回答は必ず{char_limit}文字以内で簡潔にまとめてください。" if char_limit_enabled else ""
                system_prompt = f"""あなたは親切で知識豊富なAIアシスタントです。
以下の点に注意して回答してください：

1. 日本語で丁寧に回答する
2. 質問者のレベルに合わせて説明する
3. 必要に応じて具体例を挙げる
4. プログラミングやデータ分析の質問には実用的なアドバイスを提供する
5. ビジネス関連の質問には戦略的な視点で回答する
6. 学習支援では段階的な説明を心がける
7. 不明な点があれば質問して明確にする
8. {char_limit_instruction}

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
                
                if stream_enabled:
                    # ストリーミング出力
                    with st.spinner("🤖 AIがストリーミング中..."):
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        # ストリーミングAPIを呼び出し
                        stream = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=True
                        )
                        
                        # ストリーミング応答を処理
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                # リアルタイムで表示を更新（カーソル付き）
                                message_placeholder.markdown(full_response + "▌")
                                # 設定された速度で待機してスムーズな表示を実現
                                time.sleep(stream_speed)
                        
                        # 文字数制限を適用
                        if char_limit_enabled and len(full_response) > char_limit:
                            full_response = full_response[:char_limit] + "..."
                        
                        # 最終的な応答を表示（カーソルを削除）
                        message_placeholder.markdown(full_response)
                        
                        # AI応答を履歴に追加
                        ai_message = {
                            "role": "assistant",
                            "content": full_response,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.chat_history.append(ai_message)
                else:
                    # 通常の出力
                    with st.spinner("🤖 AIが考え中..."):
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        # AI応答を取得
                        ai_response = response.choices[0].message.content
                        
                        # 文字数制限を適用
                        if char_limit_enabled and len(ai_response) > char_limit:
                            ai_response = ai_response[:char_limit] + "..."
                        
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
        "Pythonでデータ分析を始めるには？",
        "Streamlitアプリの改善方法は？",
        "機械学習の精度向上のコツは？",
        "効果的なグラフの選び方は？",
        "データベース設計のベストプラクティスは？"
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
    <p>🤖 AIチャットボット - 簡潔な質問と回答</p>
    <p>Powered by OpenAI GPT Models | Streamlit 1.46.1 | 文字数制限対応</p>
</div>
""", unsafe_allow_html=True) 