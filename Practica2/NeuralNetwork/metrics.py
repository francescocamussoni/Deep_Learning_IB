import numpy as np

def MSE(s, yt): #s=scores yt=y_true
    return np.mean(np.sum((s-yt)**2, axis=1))

def Acc_img(s, yt): #yp=y_predicted
    yp=np.argmax(s, axis=1)
    yt=np.argmax(yt, axis=1)
    return np.mean(yp==yt)

def Acc_xor(s, yt):
    s[s>0.5]=1
    s[s<=0.5]=-1
    return np.mean(s==yt)
