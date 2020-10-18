import numpy as np

IOU_all=np.load('IOU_flickr_train_862_idf.npy');
#train_list='/media/zhuoyunkan/unsupervised2019/data/f30k_precomp/train_caps_ori.txt'
train_list='/media/zhuoyunkan/unsupervised2019/data/flickr_caps_entities/train_caps_filtered.txt'
out_caps='flickr_top_1_IOU_caps_862_idf.txt'
out_f = open(out_caps,'w')
cap=[]
for line in open(train_list,'r'):
    cap.append(line.strip())
print(len(cap))
for IOU in IOU_all:
    idxs=np.argsort(IOU)[-1:]
    for idx in idxs:
        #out_f.write(str(IOU[idx])+' '+cap[idx]+'\n')
        out_f.write(cap[idx]+'\n')
        
out_f.flush()
out_f.close()