import streamlit as st

# 假设你有三张图片，存放在同一目录下，名为 'image1.jpg', 'image2.jpg', 'image3.jpg'
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
image_values = [10, 20, 30]  # 每张图片点击后返回的值

# 使用列来并排显示图片
cols = st.columns(len(image_paths))
selected_value = None

for col, img_path, val in zip(cols, image_paths, image_values):
    # 使用列来并排显示图片，图片作为按钮
    with col:
        # 使用图片路径作为按钮，点击按钮后设置 selected_value
        if st.button("", key=img_path):  # 用图片路径作为唯一的 key
            selected_value = val
        # 显示图片
        st.image(img_path, use_column_width=True)

if selected_value is not None:
    st.write(f"你点击的图片对应的值是: {selected_value}")
