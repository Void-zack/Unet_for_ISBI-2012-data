import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils.dice_coef import dice_coef
from utils.save_info import save_dice

def bina(Img):
    for i in range(len(Img)):
        Img[i][Img[i] >= 0.5] = 1
        Img[i][Img[i] < 0.5] = 0
    return Img.astype('uint8')

def Training_visualization(history,epoch,path):
    iters = range(1,epoch+1)
    plt.plot(iters,history.history['binary_accuracy'],'r',label='train acc (=%.2f)'%history.history['binary_accuracy'][-1])
    plt.plot(iters,history.history['loss'],'g',label='train loss (=%.2f)'%history.history['loss'][-1])
    plt.plot(iters,history.history['val_binary_accuracy'],'b',label='val acc (=%.2f)'%history.history['val_binary_accuracy'][-1])
    plt.plot(iters,history.history['val_loss'],'k',label='val loss (=%.2f)'%history.history['val_loss'][-1])
    plt.title('Model accuracy')
    plt.ylabel('Acc&Loss')
    plt.xlabel('Epoch')
    plt.ylim(0,1.05)
    plt.legend(loc='best')
    plt.savefig(path+'/Training.png',facecolor='white',figsize=(12,8))
    
def Dice_visualization(Predicted_mask_list,mask_list,test_path):
    pred = bina(np.squeeze(Predicted_mask_list))
    #================ dice_coef ================#
    dice_list=[]
    for i in range(len(pred)):
        dice_list.append(dice_coef(pred[i],mask_list[i]))
    save_dice(dice_list,test_path)
    #================ dice_plot ================#
    plt.plot(np.arange(len(dice_list)),dice_list,linewidth=1,color='r',marker='o',markerfacecolor='blue',markersize=3)
    plt.xlabel('picture')
    plt.ylabel('dice_coef')
    plt.xticks(np.arange(0,len(dice_list)+2,5))
    plt.yticks(np.arange(0,1.2,0.5))
    meanDice = sum(dice_list)/len(dice_list)
    plt.title('Mean of Dice_Coef:%.4f'%meanDice)
    plt.grid()
    plt.savefig(test_path+'dice_plot.png',facecolor='white')
    
def Compare_results(raw_,label_,pred_,test_path,ID):
    dice_=(pred_^label_)
    fig = plt.figure(figsize=(8,8),facecolor='green')
    raw_img = fig.add_subplot(221)
    pred_img = fig.add_subplot(222)
    label_img = fig.add_subplot(223)
    dice_img = fig.add_subplot(224)
    raw_img.set_title('Raw Image')
    raw_img.axis('off')
    raw_img.imshow(raw_,cmap='gray')
    pred_img.set_title('Pred Image')
    pred_img.axis('off')
    pred_img.imshow(pred_, cmap='gray')
    label_img.set_title('Label Image')
    label_img.axis('off')
    label_img.imshow(label_, cmap='gray')
    dice_img.set_title('XOR Image')
    dice_img.axis('off')
    dice_img.imshow(dice_, cmap='gray')
    plt.savefig(test_path+str(ID)+'_compare.png',facecolor='green')