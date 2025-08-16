import streamlit.components.v1 as components
from PIL import Image

def gender_cause_states_clean(st, gender):
    # 如果性别发生变化，清空相关状态
    if gender != st.session_state.last_gender:
        st.session_state.last_gender = gender
        st.session_state.selected_face = None
        st.session_state.face_name = "未选择"
        st.session_state.my_hair_suggest = ""
        st.session_state.selected_hair = None
        st.session_state.hair_name = "未选择"
        st.session_state.selected_body = None
        st.session_state.body_type = "未选择"

def choose_face_type(st):
    """
    选择脸型

    return: 返回选中的脸型、对应的发型建议
        str: face_name
        str: my_hair_suggest
    """

    # 定义一个缓存函数，用于加载图片
    @st.cache_data
    def load_image(image_file):
        return Image.open(image_file)

    st.write("脸型选择")
    # 定义图片路径和相应的返回值
    face_img_paths = ["img/OvalFace.png",
                    "img/SquareFace.png",
                    "img/OblongFace.png",
                    "img/TriangularFace.png",
                    "img/RoundFace.png",
                    "img/DiamondFace.png",
                    "img/heartFace.png",
                    ]
    face_name_list = ["椭圆形脸", "方形脸", "长形脸", "三角形脸", "圆形脸", "钻石形脸", "心形脸"]

    suggest_hair = ["几乎所有发型都适合，因为椭圆形脸被认为是最理想的脸型。无论是长发、中发、短发、直发还是卷发都能很好地与椭圆形脸相搭配。",
                    "可以选择柔和的卷发或波浪发，来柔化脸部的棱角。长发或侧分发型能够有效平衡脸部的线条。",
                    "选择有层次感的中长发或波浪发，可以增加脸部的宽度。刘海也是一个不错的选择，能让脸部显得短一些。",
                    "应选择在耳朵上方有较多体积的发型，如蓬松的短发或大卷发，以平衡较窄的额头和较宽的下颌。",
                    "选择有层次的长发或中发，可以拉长脸型。避免过短的发型，因为它们会让脸看起来更圆。",
                    "可以选择露出额头的发型，如偏分、侧分发型，以突出额头的宽度。中长发或卷发可以柔化脸部线条。",
                    "下巴较尖，建议选择蓬松的下巴长度的发型，或者大卷发来平衡额头和下巴的比例。刘海和中分发型也是不错的选择。",
                    ]

    # 使用 st.session_state 缓存图片
    if "face_images" not in st.session_state:
        st.session_state.face_images = {}

    # 使用 st.session_state 缓存选择结果
    if "selected_face" not in st.session_state:
        st.session_state.selected_face = None
        st.session_state.face_name = "未选择"
        st.session_state.my_hair_suggest = ""

    # 如果缓存中没有图片，则加载图片
    for path in face_img_paths:
        if path not in st.session_state.face_images:
            image = Image.open(path)
            st.session_state.face_images[path] = image
    
    # 创建列布局来展示图片
    columns = st.columns(len(face_img_paths))

    # 在每个列中显示一张图片和一个按钮
    for i, path in enumerate(face_img_paths):
        with columns[i]:
            # 从缓存中获取图片并显示
            st.image(st.session_state.face_images[path], caption=f"", use_column_width=True)
            
            # 显示按钮
            if st.button(f"{face_name_list[i]}"):
                st.session_state.selected_face = path
                st.session_state.face_name = face_name_list[i]
                st.session_state.my_hair_suggest = suggest_hair[i]
    # 显示已选中的脸型和发型建议
    if st.session_state.selected_face: 
        st.markdown("你选择了:red[" + st.session_state.face_name + "]")
        st.markdown(":blue[发型推荐]: :gray[" + st.session_state.my_hair_suggest + "]")
    
    return [st.session_state.face_name, st.session_state.my_hair_suggest]

def choose_hair_type(st, gender):
    """
    根据性别，选择发型

    return: 返回选中的发型
        str: hair_type
    """
    # 定义一个缓存函数，用于加载图片
    @st.cache_data
    def load_image(image_file):
        return Image.open(image_file)

    boy_img_paths = ["img/boys_hair9.png",
                    "img/boys_hair1.png",
                    "img/boys_hair2.png",
                    "img/boys_hair3.png",
                    "img/boys_hair4.png",
                    "img/boys_hair5.png",
                    "img/boys_hair6.png",
                    "img/boys_hair7.png",
                    "img/boys_hair8.png",
                    ]
    boy_hair_names = ["阳光寸头",
                    "长发",
                    "自然卷带刘海",
                    "经典刺头",
                    "中风",
                    "美式前刺",
                    "斜分刘海",
                    "大背头",
                    "乖巧学生头",
                    ]

    girl_img_paths = ["img/girl_hair1.png",
                    "img/girl_hair2.png",
                    "img/girl_hair3.png",
                    "img/girl_hair4.png",
                    "img/girl_hair5.png",
                    "img/girl_hair6.png",
                    "img/girl_hair7.png",
                    "img/girl_hair8.png",
                    "img/girl_hair9.png",
                    ]
    girl_hair_names = ["齐肩短发",
                    "长发刘海",
                    "长直发",
                    "双马尾",
                    "单马尾",
                    "侧麻花辫",
                    "双丸子头",
                    "低丸子头",
                    "单丸子头",
                    ]

    hair_img_paths = []
    hair_names = []

    if gender == "男性":
        hair_img_paths = boy_img_paths
        hair_names = boy_hair_names
    else:
        hair_img_paths = girl_img_paths
        hair_names = girl_hair_names
        
    st.write("发型选择")

    # 使用 st.session_state 缓存图片
    if "hair_images" not in st.session_state:
        st.session_state.hair_images = {}
    
    # 使用 st.session_state 缓存选择结果
    if "selected_hair" not in st.session_state:
        st.session_state.selected_hair = None
        st.session_state.hair_name = "未选择"

    # 如果缓存中没有图片，则加载图片
    for path in hair_img_paths:
        if path not in st.session_state.hair_images:
            image = Image.open(path)
            st.session_state.hair_images[path] = image

    # 创建列布局来展示图片
    hair_columns = st.columns(len(hair_img_paths))

    # 在每个列中显示一张图片和一个按钮
    for i, path in enumerate(hair_img_paths):
        with hair_columns[i]:
            # 从缓存中获取图片并显示
            st.image(st.session_state.hair_images[path], caption=f"", use_column_width=True)
            
            # 显示按钮
            if st.button(f"{hair_names[i]}"):
                st.session_state.selected_hair = path
                st.session_state.hair_name = hair_names[i]

    # 如果用户选择了某个图片，显示对应的数字
    if st.session_state.selected_hair:
        st.markdown("你的发型是:red[" + st.session_state.hair_name + "]")

    return st.session_state.hair_name

def choose_body_type(st, gender):
    """
    根据性别，选择体型

    return: 返回选中的体型
        str: body_type_name
    """
    # 定义一个缓存函数，用于加载图片
    @st.cache_data
    def load_image(image_file):
        return Image.open(image_file)

    boy_body_paths = ["img/boy_body_X.png",
                    "img/boy_body_T.png",
                    "img/boy_body_H.png",
                    "img/boy_body_A.png",
                    "img/boy_body_O.png",
                    ]
    boy_body_names = ["倒梯形",
                    "倒三角形",
                    "矩形",
                    "三角形",
                    "椭圆形",
                    ]

    girl_body_paths = ["img/girl_body_O.png",
                    "img/girl_body_X.png",
                    "img/girl_body_H.png",
                    "img/girl_body_A.png",
                    "img/girl_body_T.png",
                    ]
    girl_body_names = ["苹果形",
                    "沙漏形",
                    "直筒形",
                    "梨形",
                    "倒三角形",
                    ]

    body_img_paths = []
    body_names = []

    if gender == "男性":
        body_img_paths = boy_body_paths
        body_names = boy_body_names
    else:
        body_img_paths = girl_body_paths
        body_names = girl_body_names
        
    st.write("体型选择")

    # 使用 st.session_state 缓存图片
    if "body_images" not in st.session_state:
        st.session_state.body_images = {}

    # 使用 st.session_state 缓存选择结果
    if "selected_body" not in st.session_state:
        st.session_state.selected_body = None
        st.session_state.body_name = "未选择"
    
    # 如果缓存中没有图片，则加载图片
    for path in body_img_paths:
        if path not in st.session_state.body_images:
            image = Image.open(path)
            st.session_state.body_images[path] = image

        # 创建列布局来展示图片
    body_columns = st.columns(len(body_img_paths))

    # 在每个列中显示一张图片和一个按钮
    for i, path in enumerate(body_img_paths):
        with body_columns[i]:
            # 从缓存中获取图片并显示
            st.image(st.session_state.body_images[path], caption=f"", use_column_width=True)
            
            # 显示按钮
            if st.button(f"{body_names[i]}"):
                st.session_state.selected_body = path
                st.session_state.body_name = body_names[i]

    # 如果用户选择了某个图片，显示对应的数字
    if st.session_state.selected_body:
        st.markdown("你的身材是:red[" + st.session_state.body_name + "]")

    return st.session_state.body_name

def getColorChooser(st):
    """
        取色器
    """
    # 定义一个缓存函数，用于加载图片
    @st.cache_data
    def load_image(image_file):
        return Image.open(image_file)

    # HTML代码
    html_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>取色器</title>
        <style>
            body {
                text-align: center;
                font-family: Arial, sans-serif;
            }
            #color {
                width: 200px;
                height: 200px;
                border: none;
                cursor: pointer;
                border-radius: 10px;
                display: block;
                margin: 0 auto;  /* 居中显示 */
            }
            .color-info {
                margin-top: 20px;
            }
            input[type="text"] {
                cursor: pointer;
                padding: 10px;
                font-size: 16px;
                border: 2px solid #ddd;
                border-radius: 5px;
                height: 25px;
                line-height: 20px;
                text-align: center;
                color: #333;
                background-color: #f9f9f9;
                width: 100%;
                max-width: 300px;  /* 限制文本框的最大宽度 */
                margin: 10px auto;  /* 让文本框也居中 */
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                display: block;  /* 确保每个文本框占据一行 */
            }
            b {
                font-size: 18px;
                color: #FC5531;
                display: block;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div id="color-picker">
            <input type="color" id="color" value="#ff0000">
        </div>
        <div class="color-info">
            <b>点击文本框复制颜色值</b>
            <input type="text" id="hex" value="#ff0000" readonly>
            <input type="text" id="rgb" value="rgb(255, 0, 0)" readonly>
        </div>
        <script>
            const colorObj = document.getElementById('color');
            const hexObj = document.getElementById('hex');
            const rgbObj = document.getElementById('rgb');

            colorObj.oninput = function () {
                hexObj.value = colorObj.value;
                rgbObj.value = hexToRGB(colorObj.value);
            }

            hexObj.onclick = function () {
                copyToClipboard(hexObj.value);
            }

            rgbObj.onclick = function () {
                copyToClipboard(rgbObj.value);
            }

            function hexToRGB(hex) {
                const red = parseInt(hex.substring(1, 3), 16);
                const green = parseInt(hex.substring(3, 5), 16);
                const blue = parseInt(hex.substring(5, 7), 16);
                return `rgb(${red}, ${green}, ${blue})`;
            }

            function copyToClipboard(str) {
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    return navigator.clipboard.writeText(str);
                } else {
                    const textarea = document.createElement('textarea');
                    textarea.value = str;
                    document.body.appendChild(textarea);
                    textarea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textarea);
                }
            }
        </script>
    </body>
    </html>
    """

    # 在Streamlit中嵌入HTML代码
    # 创建两个并排的列
    col1, col2 = st.columns([3, 3])  # 调整列的比例，使得图片和取色器看起来平衡
    # 在左边的列中放置取色器
    with col1:
        st.markdown("##### :blue[取色器使用方法]")
        st.markdown("1. 在右边上传一张图片")
        st.markdown("2. 利用左边取色器查看颜色")
        components.html(html_code, height=400)
    # 在右边的列中放置图片上传组件和显示图片
    with col2:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = load_image(uploaded_file)
            base_height = 400
            w_percent = (base_height / float(img.size[1]))
            w_size = int((float(img.size[0]) * float(w_percent)))
            img = img.resize((w_size, base_height), Image.Resampling.LANCZOS)
            st.image(img, caption='', use_column_width=True)
        else:
            st.markdown(":gray[请上传一张待取色的图片]")
