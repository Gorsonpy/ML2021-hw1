import torch
print("torch version: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())


myseed = 45617
np.random.seed(myseed)

# 确保卷积运算结果固定
torch.backends.cudnn.deterministic = True
# 禁止自动选择最快卷积算法的功能
torch.backends.cudnn.benchmark = False
