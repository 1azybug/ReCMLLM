

# a directory for data and outputs
mkdir ../ReCMLLM_outputs
mkdir ../ReCMLLM_outputs/BucketSortText
mkdir ../ReCMLLM_outputs/data

# download data
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download HuggingFaceFW/fineweb --exclude "*/*" --local-dir ../ReCMLLM_outputs/HuggingFaceFW/fineweb
# huggingface-cli download --repo-type dataset --resume-download HuggingFaceFW/fineweb --include "sample/350BT/*" --local-dir ../ReCMLLM_outputs/HuggingFaceFW/fineweb

# 因为网络问题，可能会中断下载，所以多次尝试。
# 设置循环次数
max_attempts=100

# 循环执行命令
for (( i=1; i<=max_attempts; i++ )); do
    echo "尝试下载 $i /$max_attempts..."
    huggingface-cli download --repo-type dataset --resume-download HuggingFaceFW/fineweb --include "sample/350BT/*" --local-dir ../ReCMLLM_outputs/HuggingFaceFW/fineweb
    echo "尝试 $i 执行完毕。"
done


