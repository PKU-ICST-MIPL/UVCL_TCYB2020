from vocab import Vocabulary
import evaluation

evaluation.evalrank("ckpt-large-direct-attention/model_best.pth.tar", data_path="/media/zhuoyunkan/unsupervised2019/data/", split="test")