import streamlit as st
from PIL import Image

# 定义图片路径和相应的返回值
image_options = {
    "image1.jpg": 1,
    "image2.jpg": 2,
    "image3.jpg": 3
}

# 创建列布局来展示图片
columns = st.columns(len(image_options))

# 初始化一个变量来存储用户选择的图片
selected_image = None

# 在每个列中显示一张图片和一个按钮
for i, (image_path, number) in enumerate(image_options.items()):
    with columns[i]:
        # 显示图片
        image = Image.open(image_path)
        st.image(image, caption=f"图片 {number}", use_column_width=True)
        
        # 显示按钮
        if st.button(f"选择图片 {number}"):
            selected_image = image_path

# 如果用户选择了某个图片，显示对应的数字
if selected_image:
    st.write(f"你选择了图片 {selected_image}，返回的数字是: {image_options[selected_image]}")
