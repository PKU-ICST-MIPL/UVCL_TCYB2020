import numpy as np
import datetime
import scipy.io as sio


def IOU(image_entities,caps_entities,idf_scores):
    caps_entities=caps_entities*idf_scores
    intersect=np.sum(caps_entities*image_entities,axis=1)
    caps_sum=np.sum(caps_entities,axis=1)
    image_entities=image_entities*idf_scores
    imgs_sum=np.sum(image_entities,axis=1)
    union=caps_sum+imgs_sum-intersect
    return intersect*100.0/union
    
caps_entities=np.load('/media/zhuoyunkan/unsupervised2019/data/flickr_caps_entities/train_caps_entities_one_hot_filtered.npy')
image_entities=np.load('/media/zhuoyunkan/unsupervised2019/data/flickr_image_entities/train_img_entities_one_hot.npy')
idf_scores_ori=np.load('/media/zhuoyunkan/unsupervised2019/data/flickr_caps_entities/flickr_idf_score.npy')
out_caps=open('/media/zhuoyunkan/unsupervised_IOU/IOU_test/flickr_top_IOU_caps_862_idf.txt','w')
out_caps_with_confidence=open('/media/zhuoyunkan/unsupervised_IOU/IOU_test/flickr_top_IOU_caps_862_idf_with_confidence.txt','w')

print(caps_entities.shape)
print(image_entities.shape)
print(idf_scores_ori.shape)

np.set_printoptions(threshold='nan')
IOU_all=np.zeros([len(image_entities),len(caps_entities)],dtype=np.int8)

train_list='/media/zhuoyunkan/unsupervised2019/data/flickr_caps_entities/train_caps_filtered.txt'
cap=[]
for line in open(train_list,'r'):
    cap.append(line.strip())
print(len(cap))
#long running

starttime = datetime.datetime.now()

for i in range(len(image_entities)):
    if i%10==0:
        print('processing',i)
    image_entities_rep=np.repeat(image_entities[np.newaxis,i],len(caps_entities),axis=0)
    idf_scores_rep=np.repeat(idf_scores_ori[np.newaxis,:],len(caps_entities),axis=0)
    IOU_all[i]=IOU(image_entities_rep,caps_entities,idf_scores_rep)
    IOU_i=IOU_all[i]
    idxs=np.argsort(IOU_i)[-5:]
    for idx in idxs:
        out_caps_with_confidence.write(str(IOU_i[idx])+' '+cap[idx]+'\n')
        out_caps.write(cap[idx]+'\n')

endtime = datetime.datetime.now()
duration = (endtime - starttime).seconds
print('duration',duration)
IOU_all=IOU_all.astype(np.int8)
np.set_printoptions(threshold='nan')
print(IOU_all[0,:100])
np.save('IOU_flickr_train_862_idf.npy',IOU_all)

