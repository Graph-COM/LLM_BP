from huggingface_hub import login
from huggingface_hub import upload_folder

login(token='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')

repo_id = 'Graph-COM/Text-Attributed-Graphs'  # 替换为你的 repo
local_dir = './results'   # 替换为本地文件夹路径

upload_folder(
    folder_path=local_dir,
    path_in_repo='results',
    repo_id=repo_id,
    repo_type='dataset',  # 可选："model", "dataset", "space"
    commit_message='upload result of GPT-4o in classification and GPT-4o-mini in predicting r'
)