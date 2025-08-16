# 设置OpenMP线程数为8
import os
# os.environ["OMP_NUM_THREADS"] = "2"

from typing import Any, List, Optional

import streamlit as st

# 从llama_index库导入HuggingFaceEmbedding类，用于将文本转换为向量表示
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# 从llama_index库导入ChromaVectorStore类，用于高效存储和检索向量数据
from llama_index.vector_stores.chroma import ChromaVectorStore
# 从llama_index库导入PyMuPDFReader类，用于读取和解析PDF文件内容
from llama_index.readers.file import PyMuPDFReader
# 从llama_index库导入NodeWithScore和TextNode类
# NodeWithScore: 表示带有相关性分数的节点，用于排序检索结果
# TextNode: 表示文本块，是索引和检索的基本单位。节点存储文本内容及其元数据，便于构建知识图谱和语义搜索
from llama_index.core.schema import NodeWithScore, TextNode
# 从llama_index库导入RetrieverQueryEngine类，用于协调检索器和响应生成，执行端到端的问答过程
from llama_index.core.query_engine import RetrieverQueryEngine
# 从llama_index库导入QueryBundle类，用于封装查询相关的信息，如查询文本、过滤器等
from llama_index.core import QueryBundle
# 从llama_index库导入BaseRetriever类，这是所有检索器的基类，定义了检索接口
from llama_index.core.retrievers import BaseRetriever
# 从llama_index库导入SentenceSplitter类，用于将长文本分割成句子或语义完整的文本块，便于索引和检索
from llama_index.core.node_parser import SentenceSplitter
# 从llama_index库导入VectorStoreQuery类，用于构造向量存储的查询，支持语义相似度搜索
from llama_index.core.vector_stores import VectorStoreQuery
# 向量数据库
import chromadb
from ipex_llm.llamaindex.llms import IpexLLM

from zhipuai import ZhipuAI
import urllib.request
from PIL import Image
from tools import choose_face_type, choose_hair_type, choose_body_type, getColorChooser, gender_cause_states_clean

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context
client = ZhipuAI(api_key="xx")

class Config:
    """配置类,存储所有需要的参数"""
    model_path = "qwen2chat_int8"
    tokenizer_path = "qwen2chat_int8"
    question = "我今天该穿什么？"
    data_path = "./doc"
    persist_dir = "./chroma_db_test"
    embedding_model_path = "qwen2chat_src/AI-ModelScope/bge-small-zh-v1___5"
    max_new_tokens = 8192

def load_vector_database(persist_dir: str) -> ChromaVectorStore:
    """
    加载或创建向量数据库
    
    Args:
        persist_dir (str): 持久化目录路径
    
    Returns:
        ChromaVectorStore: 向量存储对象
    """
    # 检查持久化目录是否存在
    if os.path.exists(persist_dir):
        print(f"正在加载现有的向量数据库: {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_collection("dress")
    else:
        print(f"创建新的向量数据库: {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.create_collection("dress")
    print(f"Vector store loaded with {chroma_collection.count()} documents")
    return ChromaVectorStore(chroma_collection=chroma_collection)

def load_data(data_path: str) -> List[TextNode]:
    """
    加载并处理PDF数据
    
    Args:
        data_path (str): PDF文件路径
    
    Returns:
        List[TextNode]: 处理后的文本节点列表
    """
    loader = PyMuPDFReader()
    documents = loader.load(file_path=data_path)

    text_parser = SentenceSplitter(chunk_size=384)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)
        print("idx: ", idx, "\nnode: ", node)
    return nodes

class VectorDBRetriever(BaseRetriever):
    """向量数据库检索器"""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        检索相关文档
        
        Args:
            query_bundle (QueryBundle): 查询包
        
        Returns:
            List[NodeWithScore]: 检索到的文档节点及其相关性得分
        """
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        print(f"Retrieved {len(nodes_with_scores)} nodes with scores")
        return nodes_with_scores

def completion_to_prompt(completion: str) -> str:
    """
    将完成转换为提示格式
    
    Args:
        completion (str): 完成的文本
    
    Returns:
        str: 格式化后的提示
    """
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

def messages_to_prompt(messages: List[dict]) -> str:
    """
    将消息列表转换为提示格式
    
    Args:
        messages (List[dict]): 消息列表
    
    Returns:
        str: 格式化后的提示
    """
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    prompt = prompt + "<|assistant|>\n"

    return prompt

def setup_llm(config: Config) -> IpexLLM:
    """
    设置语言模型
    
    Args:
        config (Config): 配置对象
    
    Returns:
        IpexLLM: 配置好的语言模型
    """
    return IpexLLM.from_model_id_low_bit(
        model_name=config.model_path,
        tokenizer_name=config.tokenizer_path,
        context_window=16384,
        max_new_tokens=config.max_new_tokens,
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        model_kwargs={},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="cpu",
    )

def generate_response(message):
    config = Config()
    embed_model = HuggingFaceEmbedding(model_name=config.embedding_model_path)
    llm = setup_llm(config)
    vector_store = load_vector_database(persist_dir=config.persist_dir)

    # 直接初始化检索器（复用已实现的检索逻辑）
    retriever = VectorDBRetriever(
        vector_store, 
        embed_model, 
        query_mode="default", 
        similarity_top_k=2  # 这里可根据需求调整返回数量，原手动查询用了2
    )

    # 通过检索器获取查询结果（检索器内部已完成向量查询和结果整理）
    query_bundle = QueryBundle(query_str=message)  # 封装查询文本
    nodes_with_scores = retriever.retrieve(query_bundle)  # 调用检索器的检索方法s
    print(f"Retrieved {len(nodes_with_scores)} nodes with scores")  # 保留日志输出
    print(nodes_with_scores)

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm, streaming=True)
    response = query_engine.query(message)
    return response

def main():
    """主函数"""
    config = Config()
    
    # 数据库插入下载一次
    if os.path.exists(config.persist_dir) and os.listdir(config.persist_dir):
        print("数据库已加载")
    else:
        config = Config() 
        embed_model = HuggingFaceEmbedding(model_name=config.embedding_model_path)
        # 加载向量数据库
        vector_store = load_vector_database(persist_dir=config.persist_dir)
        for filename in os.listdir(config.data_path):
            file_path = os.path.join(config.data_path, filename)    
            nodes = load_data(data_path=file_path)
            for node in nodes:
                node_embedding = embed_model.get_text_embedding(
                    node.get_content(metadata_mode="all")
                )
                node.embedding = node_embedding
            vector_store.add(nodes)

    st.title("魔镜魔镜，什么是世界上最好的穿搭")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    ## 单次函数执行项
    #  提交表单
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if not st.session_state.submitted:
        # 用户信息输入
        gender = st.selectbox("性别", ["女性", "男性"])
        # 性别修改则清除有关st缓存
        if "last_gender" not in st.session_state:
            st.session_state.last_gender = gender
        gender_cause_states_clean(st, gender)
        # body_type = st.selectbox("体型", ["H", "X", "O"])
        body_type = choose_body_type(st, gender)
        # face_type = st.selectbox("脸型", ["长脸", "方脸", "圆脸"])
        face_type, my_hair_suggestion = choose_face_type(st)
        # 增加发型
        hair_type = choose_hair_type(st, gender)
        shoulder_type = st.selectbox("肩型", ["宽肩", "窄肩"])
        arm_type = st.selectbox("臂型", ["短臂", "长臂"])
        height = st.text_input("身高 (cm)", "170")
        weight = st.text_input("体重 (kg)", "60")
        occasion = st.text_input("场景", "参加朋友的婚礼")
        custom_color = st.text_input("服装颜色偏好")
        # 获取肤色、或其他颜色的 rgb 和 hex颜色编码
        getColorChooser(st)
        body_color = st.text_input("肤色取色器取色值（需要手动复制取色值粘贴到此处）")
        style_preference = st.multiselect("风格偏好", ["简约", "经典", "复古", "明亮"], default=["简约", "经典"])
        specific_requirements = st.text_input("具体要求", "舒适性，适合迅速抢到捧花")

        submit_button = st.button("提交")
        if submit_button:
            st.session_state.gender = gender
            st.session_state.face_type = face_type
            st.session_state.hair_type = hair_type
            st.session_state.shoulder_type = shoulder_type
            st.session_state.arm_type = arm_type
            st.session_state.body_type = body_type
            st.session_state.height = height
            st.session_state.weight = weight
            st.session_state.occasion = occasion
            st.session_state.custom_color = custom_color
            st.session_state.style_preference = style_preference
            st.session_state.specific_requirements = specific_requirements
            st.session_state.body_color = body_color
            st.session_state.submitted = True
            st.rerun()  # 重新运行脚本，隐藏输入框
    #  查看表单的 prompt
    if st.session_state.submitted:
        init_info = f"用户的性别是{st.session_state['gender']}，身材是{st.session_state['body_type']}，肤色是{st.session_state['body_color']}，脸型是{st.session_state['face_type']}，{st.session_state['shoulder_type']}、{st.session_state['arm_type']}、{st.session_state['body_type']}， 发型是{st.session_state['hair_type']}，身高是{st.session_state['height']}cm，体重是{st.session_state['weight']}kg，目标场景在{st.session_state['occasion']}，服装的颜色偏好为{st.session_state['custom_color']}，喜爱的风格偏好为{', '.join(st.session_state['style_preference'])}，另外你推荐的穿搭还有{st.session_state['specific_requirements']}的要求。现在你是一位虚拟的时尚顾问，请需要根据上面提供的这些信息给出具体的搭配建议，你需要根据用户提供的场景信息，推荐一套完整的服饰搭配，包括上衣、下装、鞋子和配件等，并简要说明每个搭配的原因，一条一条来。"
        st.session_state.messages.append({"role": "system", "content":init_info})

        pic_info_fir = f"根据这些要求画一幅画：用户的性别是{st.session_state['gender']}，身材是{st.session_state['body_type']}，肤色是{st.session_state['body_color']}，脸型是{st.session_state['face_type']}，{st.session_state['shoulder_type']}、{st.session_state['arm_type']}、{st.session_state['body_type']}，发型是{st.session_state['hair_type']}，身高是{st.session_state['height']}cm，目标场景在{st.session_state['occasion']}，"

        answer = generate_response(init_info)
        answer_str = answer.response_txt
        st.write_stream(answer.response_gen)
        st.markdown("示例图生成中...")
        pic_info = f"{pic_info_fir}并且该用户的服装搭配选择是{answer_str}，生成相应的图片"
        response = client.images.generations(
            # model="cogview-3",
            model="cogview-4",
            prompt=pic_info,
        )
        web_image_url=response.data[0].url
        urllib.request.urlretrieve(
            str(web_image_url),
            'img/cloth.png'
        )
        img = Image.open('img/cloth.png')
        st.image(img)

if __name__ == "__main__":

    main()