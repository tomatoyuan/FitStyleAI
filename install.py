import os
from modelscope import snapshot_download
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

def install_model():
    # 下载模型并设置缓存路径
    model_path = snapshot_download('Qwen/Qwen2-7B-Instruct', cache_dir='qwen2chat_src', revision='master')
    
    # 加载模型并应用低精度量化
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int8', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 保存低精度量化模型和分词器
    model.save_low_bit('qwen2chat_int8')
    tokenizer.save_pretrained('qwen2chat_int8')

def download_FlagEmbedding():
    # 保存 AI-ModelScope/bge-small-zh-v1.5
    embedding_path = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir='qwen2chat_src', revision='master')


if __name__ == "__main__":
    install_model()
    print("7B模型下载完成")

    download_FlagEmbedding()
    print("FlagEmbedding模型下载完成")
