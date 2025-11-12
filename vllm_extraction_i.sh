# 初始化 Conda
conda init bash > /dev/null 2>&1
source ~/.bashrc

# 激活指定 Conda 环境
conda activate gemma  # 替换为你的 Conda 环境名

# 执行 Python 脚本
python code/vllm_keyword_extraction.py --OA_SECTION "04"       # 替换为你的 Python 脚本名
