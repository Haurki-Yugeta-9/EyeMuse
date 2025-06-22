# ライブラリのインポート
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import colorsys
# from dotenv import load_dotenv
# import os
# from openai import OpenAI


# # 環境変数の読み込み
# load_dotenv()
# client = OpenAI()

# # OpenAI APIキーの取得
# # Local用
# OpenAI.api_key = os.getenv("OPENAI_API_KEY")
# # # Host用
# # OpenAI.api_key = st.secrets["OPENAI_API_KEY"]


# サイドバーの設定
with st.sidebar:
    # サイドバーのタイトル
    st.header("操作パネル")
       
    # 画像のアップロード
    file_upload = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    
    # メイクスタイルの選択
    style_options = [
        "---メイクスタイルの選択---",
        "クール",
        "キュート",
        "エレガント",
        "ナチュラル",
        "セクシー",
        "カジュアル",
        "フェミニン",
        "ボーイッシュ",
        "ゴージャス"
    ]
    selected_style = st.selectbox("メイクスタイルを選択してください", style_options)
    
    # 眼の形の選択 
    eye_shape_options = [
        "---眼の形の選択---",
        "一重まぶた",
        "二重まぶた",
        "奥二重まぶた",
    ]
    selected_eye_shape = st.selectbox("眼の形を選択してください", eye_shape_options)

    # 送信ボタン
    # いずれか未選択の場合、送信ボタンを無効化
    is_disabled = (
        file_upload is None or
        selected_style == "---メイクスタイルの選択---" or
        selected_eye_shape == "---眼の形の選択---"
    )
    submit_button = st.button("送信", disabled=is_disabled)


# 中央画面の設定
# アプリタイトル
st.title("EyeMuse")

# アプリの説明
st.caption("好きな写真をアップして、あなたらしいアイメイクをプロデュース！")


# アップロード画像の解析
# アップロードされたらfile_uploadがNoneではなくなるので、実行される
if (file_upload !=None):
    # 分析する画像を表示
    st.image(file_upload, caption="アップロードされた画像", use_container_width=True) 

# 送信ボタンPB後の処理
if submit_button and file_upload is not None:
    # アップロードされた画像のサイズを取得
    image = Image.open(file_upload)
    width, height = image.size
    print(width, height)
    
    # 空のNumpy配列を作成
    # 高さ、幅、RGBの3チャンネルを持つ配列を作成
    # dtype=intは整数型の配列を作成するため
    image_array = np.empty((height, width, 3), dtype=int)
    
    # 画像の各ピクセルのRGB値を配列に格納
    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            image_array[y, x] = [r, g, b]
    print(image_array)


    # メイン・ベース・アクセントカラーの取得    
    # K-meansクラスタリング
    n_colors = 4
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(image_array.reshape(-1, 3))
    # クラスタリング結果の色を取得
    colors = kmeans.cluster_centers_.astype(int)
    
    # 各クラスタの割合を計算
    counts = np.bincount(kmeans.labels_)
    ratios = counts / counts.sum()
    # 割合の大きい順にインデックスを取得
    sorted_idx = np.argsort(ratios)[::-1]
    # # メインカラーの取得
    # main_color = colors[sorted_idx[0]]  # 最も割合の高いクラスタをメインカラーとする
    # ベースカラーの取得
    base_color = colors[sorted_idx[0:]] # 全クラスタをベースカラーとする

    # # メインカラーの表示
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    # main_hex = rgb_to_hex(main_color)
    # st.subheader("メインカラー")
    # st.color_picker("", value=main_hex, key="main_color", disabled=True)
    

    # ベースカラーの表示（円で表示）
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.imshow([base_color], aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    st.subheader("ベースカラー")
    st.pyplot(fig)
   

    # アクセントカラーの取得
    # 画像の全ピクセルからHSV彩度最大の色を探す
    flat_pixels = image_array.reshape(-1, 3)
    max_s = -1
    accent_rgb = None
    for rgb in flat_pixels:
        r, g, b = rgb / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # ベースカラーから離れた色を候補にする
        is_contrast = True
        for base in np.vstack([base_color]):
            if np.linalg.norm(rgb - base) < 120: # 120未満は距離が近いとする
                is_contrast = False
                break
        if is_contrast and s > max_s:
            max_s = s
            accent_rgb = rgb

    if is_contrast and s > max_s:
        accent_rgb = accent_rgb.astype(int)
        accent_hex = rgb_to_hex(accent_rgb)
        st.subheader("アクセントカラー")
        st.color_picker("", value=accent_hex, key="accent_color", disabled=True)
    else:
        # アクセントカラーが見つからなかった場合の処理
        # 原色に最も近いRGB値を取得
        def get_accent_color(colors):
            # RGBの原色
            primary_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
            # 各クラスタリング結果の色と原色の距離を計算
            distances = np.linalg.norm(colors[:, np.newaxis] - primary_colors, axis=2)
            # 最小距離のインデックスを取得
            min_index = np.argmin(distances, axis=1)
            return primary_colors[min_index]
        accent_colors = get_accent_color(colors)
        st.subheader("アクセントカラー")
        # RGBタブルを16進数に変換
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(*rgb)
        accent_hex = rgb_to_hex(accent_colors[0])
        st.color_picker("", value=accent_hex, key="accent_color_1")


    # メイクアドバイスの設定
    # 眼の選択内容確認
    if selected_eye_shape == "一重まぶた":
        eye_pattern = "一重"
    elif selected_eye_shape == "二重まぶた":
        eye_pattern = "二重"
    else:
        eye_pattern = "奥二重"
    
    # ベースカラーのパターン決定
    # ベースカラーの平均RGBを計算
    base_rgb = np.mean(base_color, axis=0).astype(int)
    print("ベースカラーの平均RGB:", base_rgb)
    R, G, B = base_rgb

    # ベースカラーのパターン判定
    if R==G==B or (R>G and R>B and (R<=160 or G-B>=30)):
        base_pattern = "茶"
    elif R>G and R>B:
        base_pattern = "赤"
    elif G>R and G>B:
        base_pattern = "緑" 
    elif B>R and B>G:
        base_pattern = "青"
    else:
        base_pattern = "黄"
    
    # アクセントカラーのパターン決定
    AR, AG, AB = accent_colors[0]
    print("アクセントカラーのRGB:", AR, AG, AB)
    # アクセントカラーのパターン判定
    if AR>AG and AR>AB:
        accent_pattern = "赤"
    elif AG>AR and AG>AB:
        accent_pattern = "緑" 
    else:
        accent_pattern = "青"


    # 画像のファイルパスの取得
    # 画像ファイル名を組み立て
    image_path = f"{eye_pattern}/ベース{base_pattern}/{accent_pattern}.jpg"
    
    # 画像の表示
    st.subheader("メイクイメージ")
    st.image(image_path, caption=f"{eye_pattern}/ベース{base_pattern}/{accent_pattern}")
    

    # # 日本人女性のすっぴん画像の生成
    # st.subheader("メイク前")
    # prompt_plain = "A typical Japanese woman's face without makeup, studio photo, neutral background"
    # response_plain = client.images.generate(prompt=prompt_plain, n=1, size="512x512")
    # image_plain_url = response_plain.data[0].url
    # st.image(image_plain_url, use_container_width=True)

    # # メイク後の画像生成
    # st.subheader("メイク後")
    # prompt_makeup = (
    #     f"A typical Japanese woman's face with {selected_style} makeup, "
    #     f"main color: {main_hex}, base colors: {', '.join(rgb_to_hex(c) for c in base_color)}, "
    #     f"accent color: {accent_hex}, studio photo, neutral background"
    # )
    # response_makeup = client.images.generate(prompt=prompt_makeup, n=1, size="512x512")
    # image_makeup_url = response_makeup.data[0].url
    # st.image(image_makeup_url, use_container_width=True)