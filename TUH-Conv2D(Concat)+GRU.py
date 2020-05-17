# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 18:02:51 2018

@author: QCY
"""

import numpy as np
import keras
import math,random,string
import sklearn
import matplotlib.pyplot as plt
import datetime as dt  
from keras.models import Model
from keras.layers import Dense, Dropout,TimeDistributed,Conv2D,Reshape,GRU,Input
from keras import regularizers,optimizers
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,classification_report

NUM_CLASSES=2
BATCH_SIZE=150
NB_FOLD=10
Train_Accuracy=np.zeros((NB_FOLD,))
Test_Accuracy=np.zeros((NB_FOLD,))
Best_Test_Accuracy=np.zeros((NB_FOLD,))
def lr(epoch):
    initial_lrate = 0.01
    drop = 0.6
    epochs_drop = 20
    learning_rate = round(initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop)),6)
    print("Learning rate is {}.".format(learning_rate))
    return learning_rate

def savefig(hist,lb,modelname):
    if lb=='loss':
        plt.figure(figsize=(18,12))
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./lossfig/'+modelname+'.jpg')
        
    elif lb=='accuracy':
        plt.figure(figsize=(18,12))
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./accfig/'+modelname+'.jpg')
        
    else:
        print('error param.')

    
Dataset=np.fromfile('E:/Python/workspace/TUH_EEG/dataset/1000Samples/Dataset_Train.dump')
Label=np.fromfile('E:/Python/workspace/TUH_EEG/dataset/1000Samples/Label_Train.dump')
Dataset=Dataset.reshape((-1,21,1000))

#再次乱序
Index=np.arange(Dataset.shape[0])
np.random.shuffle(Index)
Dataset=Dataset[Index]
Label=Label[Index]

Dataset=Dataset[:1500,:,:]
Label=Label[:1500]        

#采用maxabs归一化
for i in range(Dataset.shape[0]):
    for j in range(Dataset.shape[1]):
        Dataset[i][j]=sklearn.preprocessing.maxabs_scale(Dataset[i][j])

#对数据集进行reshape，变为(1500,21,1000,1),作为conv2d的输入
Dataset=Dataset.reshape((-1,21,1000,1))

#onehot
Label=keras.utils.to_categorical(Label,NUM_CLASSES)

Label_New = np.zeros((Label.shape[0],20,2))
for i in range(Label.shape[0]):
    Label_New[i,:] = Label[i]


def createModel():
    inputs = Input(batch_shape=(BATCH_SIZE,Dataset.shape[1],Dataset.shape[2],1))
    conv2d_1_1=Conv2D(filters=6,kernel_size=(2,1),strides=(2,1),padding='same',activation='relu')(inputs)
    conv2d_1_2=Conv2D(filters=6,kernel_size=(4,1),strides=(2,1),padding='same',activation='relu')(inputs)
    conv2d_1_3=Conv2D(filters=6,kernel_size=(8,1),strides=(2,1),padding='same',activation='relu')(inputs)
    concat_1 =keras.layers.Concatenate(axis=3)([conv2d_1_1,conv2d_1_2,conv2d_1_3])
    conv2d_2_1=Conv2D(filters=3,kernel_size=(2,1),strides=(2,1),padding='same',activation='relu')(concat_1)
    conv2d_2_2=Conv2D(filters=3,kernel_size=(4,1),strides=(2,1),padding='same',activation='relu')(concat_1)
    conv2d_2_3=Conv2D(filters=3,kernel_size=(8,1),strides=(2,1),padding='same',activation='relu')(concat_1)   
    concat_2=keras.layers.Concatenate(axis=3)([conv2d_2_1,conv2d_2_2,conv2d_2_3])
    conv2d_3=Conv2D(filters=1,kernel_size=(6,1),strides=(1,1))(concat_2)
    dropout_1=Dropout(0.5)(conv2d_3)
    rnn_inputs=Reshape(target_shape=(20,50))(dropout_1)
    gru1 = GRU(100,stateful=True,kernel_regularizer=regularizers.l2(0.02),recurrent_dropout=0.25,return_sequences=True)(rnn_inputs)
    fc = TimeDistributed(Dense(100,activation='relu'))(gru1)
    dropout_2=Dropout(0.5)(fc)
    gru2 = GRU(100,stateful=True,kernel_regularizer=regularizers.l2(0.02),recurrent_dropout=0.25,return_sequences=True)(dropout_2)
    outputs = Dense(2, activation='sigmoid')(gru2)
    model = Model(inputs=inputs, outputs=outputs)
    adam = optimizers.adam(lr=0.01,clipvalue=0.5)
    #编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


def trainModel(sample_train,label_train_new,sample_test,label_test_new):

    model=createModel()
    model_filename=str(dt.datetime.now())
    model_filename=model_filename[:10]
    model_filename=model_filename+''.join(random.sample(string.ascii_letters, 5))
    
    history=model.fit(sample_train, 
                      label_train_new,
                      epochs=100,
                      verbose=1, 
                      batch_size=BATCH_SIZE,
                      validation_data=(sample_test,label_test_new),
                      callbacks=[keras.callbacks.LearningRateScheduler(lr),
                      keras.callbacks.ModelCheckpoint('./weights/' + model_filename +'.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)])
    
    model.save('./model/'+model_filename+'.h5')
    
    return model,history,model_filename



def computeOverallAcc(sample='test',sample_train=None,label_train=None,sample_test=None,label_test=None,model=None):
    if sample is 'test':
        x_data = sample_test
        y_ans = label_test
        output = 'Test'
    else:
        x_data =sample_train
        y_ans = label_train
        output = 'Train'
    y_pred_subs = model.predict(x_data,batch_size=BATCH_SIZE) #y_pred_subs.shape=(N, 20, 2)
    y_pred_result = np.zeros(y_pred_subs.shape)
    for i in range(y_pred_subs.shape[0]):
        for j in range(y_pred_subs.shape[1]):
            #返回最大值下标.argmax()
            max_id = y_pred_subs[i][j].argmax()
            y_pred_result[i][j][max_id] = 1
    #统计20个单元中[0,1]和[1,0]的标签数量，输出结果如：[1,19],[15,5]..etc.
    y_pred_result= y_pred_result.sum(axis=1)    # y_pred_result.shape=(N,2)
    y_pred_final = np.zeros(y_pred_result.shape)#y_pred_final.shape=(N,2)
    for i in range(y_pred_final.shape[0]):
        #投票法，将[0,1]和[1,0]中标签数量较多的置为1
        max_id = y_pred_result[i].argmax()
        y_pred_final[i][max_id] = 1
    #将统计所得结果与原始标签进行对比,若与标签符合则为全0，若不符合则为非0
    error = y_pred_final - y_ans  #error.shape=(N,2)
    #先对结果逐个检测是否为全0，若是全0则返回True,若为非0则返回False；再对该结果取反，取出非0的结果置入Error中
    error = error[~(error==0).all(axis=1)]   #error.shape=(N,2),N为样本中检测不一致的个数
    overall_acc = (y_ans.shape[0]-error.shape[0])/y_ans.shape[0]
    print("{} prediction accuracy: {:.4f}".format(output,overall_acc))
    #np.argmax()返回最大值的下标，最大值下标为0表示[1,0]即正常状态，最大值下标为1表示[0,1]即癫痫状态
    #confusion_matrix(label_true,label_pred),此处把onehot转换成了最初的1维标签，即0或1
    cm = confusion_matrix(np.argmax(y_ans,axis=1),np.argmax(y_pred_final,axis=1))#混淆矩阵
    precision,recall,fbeta_score,support = precision_recall_fscore_support(np.argmax(y_ans,axis=1),np.argmax(y_pred_final,axis=1))
    print("Confusion Matrix: \n",cm)
    print("Classification Report:\n",classification_report(np.argmax(y_ans,axis=1),np.argmax(y_pred_final,axis=1)))
    return overall_acc,cm,precision,recall,fbeta_score,support



#10折交叉验证，求平均准确率
for i in range(NB_FOLD):
    fold=len(Dataset)//NB_FOLD
    Sample_Train=np.vstack((Dataset[:i*fold],Dataset[(i+1)*fold:]))
    Sample_Test=Dataset[i*fold:(i+1)*fold]
    
    Label_Train_New=np.vstack((Label_New[:i*fold],Label_New[(i+1)*fold:]))
    Label_Test_New=Label_New[i*fold:(i+1)*fold]
    
    Label_Train=np.vstack((Label[:i*fold],Label[(i+1)*fold:]))
    Label_Test=Label[i*fold:(i+1)*fold]
    
    Models,History,Model_Filename=trainModel(Sample_Train,Label_Train_New,Sample_Test,Label_Test_New)
    savefig(History,'loss',Model_Filename)
    savefig(History,'accuracy',Model_Filename)
    # Compute overall prediction accuracy (after vote) 计算模型整体分类准确度（“投票法”最终结果）
    print ("======== The {}th fold's training epochs finished ========".format(i))
    train_acc = computeOverallAcc(sample='train',sample_test=Sample_Test,label_test=Label_Test,sample_train=Sample_Train,label_train=Label_Train,model=Models)
    test_acc = computeOverallAcc(sample='test',sample_test=Sample_Test,label_test=Label_Test,sample_train=Sample_Train,label_train=Label_Train,model=Models)
        #调用验证集准确率最高的权重模型
    print ("====== Reload the best weights for val_acc ======")
    Models.load_weights('./weights/'+ Model_Filename+'.h5')
    best_acc = computeOverallAcc(sample='test',sample_test=Sample_Test,label_test=Label_Test,sample_train=Sample_Train,label_train=Label_Train,model=Models)


    Train_Accuracy[i]=train_acc[0]
    Test_Accuracy[i]=test_acc[0]  
    Best_Test_Accuracy[i]=best_acc[0]


print("The final average Test_Accuracy is {:.2f}%".format(Test_Accuracy.mean()*100))
print("The final average Train_Accuracy is {:.2f}%".format(Train_Accuracy.mean()*100))
print("The final average Best_Test_Accuracy is {:.2f}%".format(Best_Test_Accuracy.mean()*100))









    
