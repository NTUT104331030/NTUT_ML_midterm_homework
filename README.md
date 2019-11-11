# 北科大 機器學習 期中作業

----
## 作法說明

**剔除不需要的輸入參數**

這邊選擇剔除掉銷售日期是因為相對於其他輸入參數，銷售日期的雜訊相對較多，也比較難看出日期跟房價之間的回歸關係

![image](https://github.com/NTUT104331030/NTUT_ML_midterm_homework/blob/master/Pic/截圖%202019-11-11%20下午3.29.26.png)

**使用MLP來訓練模型**

一開始我使用線性回歸的方式來做預測模型，但後來發現預測精準度非常有限於事改成MLP來做預測其結果提升了8倍之多

**多神經元深度模型**

![image](https://github.com/NTUT104331030/NTUT_ML_midterm_homework/blob/master/Pic/截圖%202019-11-11%20下午3.53.22.png)

**使用交叉驗證來提高準確度**
![image](https://github.com/NTUT104331030/NTUT_ML_midterm_homework/blob/master/Pic/截圖%202019-11-11%20下午4.26.38.png)


## 程式流程圖

![image](https://github.com/NTUT104331030/NTUT_ML_midterm_homework/blob/master/Pic/flow.png)


**匯入函示庫**

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd 
    import matplotlib as mpl
    
    from keras.models import Sequential
    from keras.callbacks import TensorBoard
    from keras.layers import Dense, Activation, Input

    from sklearn import preprocessing
    
**讀取資料**

    #read train data 
    data_1 = pd.read_csv('./machine-learning-realtek-regression/train-v3.csv')
    X_train = data_1.drop(['price','id','sale_yr','sale_month','sale_day','zipcode'],axis=1).values
    y_train = data_1['price'].values

    #read valid data
    data_2 = pd.read_csv('./machine-learning-realtek-regression/valid-v3.csv')
    X_valid = data_2.drop(['price','id','sale_yr','sale_month','sale_day','zipcode'],axis=1).values
    y_valid = data_2['price'].values

    #read test data
    data_3 = pd.read_csv('./machine-learning-realtek-regression/test-v3.csv')
    X_test = data_3.drop(['id','sale_yr','sale_month','sale_day','zipcode'],axis=1).values
    
**正規化數據**

    X_train = preprocessing.scale(X_train)
    X_valid = preprocessing.scale(X_valid)
    X_test  = preprocessing.scale(X_test)
    
**模型設定**   

    def build_model() : 
        model = Sequential()

        #add hidden layer with relu activation function 
        model.add(Dense(units = 32,input_dim=X_train.shape[1],activation='relu'))
        model.add(Dense(units = 128,activation='relu'))
        model.add(Dense(units = 128,activation='relu'))
        model.add(Dense(units = 32,activation='relu'))
        model.add(Dense(X_train.shape[1],activation='relu'))
    
        #output layer
        model.add(Dense(1))
        model.compile(loss='mae',optimizer='adam')
    
        return model
**交叉驗證** 
    
    #k-fold number
    k = 4
    
    #每一折樣本數
    nb_val_samples = len(X_train)//k 
    
    #訓練次數
    epochs = 300
    
    #批次訓練
    batch_size = 50#every bacth number 
    
    for i in range(k):
        print("processing Fold #" + str(i))
        print((i+1)*nb_val_samples)
        
        X_val = X_train[i*nb_val_samples: (i+1)*nb_val_samples]
        Y_val = y_train[i*nb_val_samples: (i+1)*nb_val_samples]
    
        X_train_p = np.concatenate(
            [X_train[:i*nb_val_samples],
             X_train[(i+1)*nb_val_samples:]], axis = 0)
    
        Y_train_p = np.concatenate(
            [y_train[:i*nb_val_samples],
             y_train[(i+1) * nb_val_samples:]], axis = 0)
            
        model = build_model()
        model.fit(X_train_p,Y_train_p,batch_size=batch_size,epochs=epochs,verbose=1)

**輸出模型**

    fn = str(epochs) + '_' + str(batch_size)
    model.save(fn + '.h5')

**輸出預測值**

    y_predict = model.predict(X_test)
    
    #填入id行
    y_predict = np.insert(y_predict, 0, values=data_3["id"].values.astype(int),axis=1)
    
    #寫出成csv檔
    np.savetxt('test_first_try_5.csv' ,y_predict,delimiter=',',header="id,price")
    
    #檢查輸出值
    y_preview = pd.read_csv('./test_first_try_5.csv')
    
    
----
## 結果分析
![image](https://github.com/NTUT104331030/NTUT_ML_midterm_homework/blob/master/Pic/score.png)
![image](https://github.com/NTUT104331030/NTUT_ML_midterm_homework/blob/master/Pic/截圖%202019-11-11%20下午4.35.29.png)

----
## 為什麼誤差值很大？（猜測）
** 資料預處理**

* 沒有檢查數據有沒有缺失
* 沒有針對離散資料做處理
* 沒有用例外狀況去處理極端值等雜訊


----
## 改進方法
* 第一次嘗試：我把全部的參數餵進去模型沒有做任何處理
* 第二次嘗試：我參考書上的步驟加入了交叉驗證方法
* 第三 ~ 五次嘗試：打開Excel查看個別參數跟價格之間的關係，測試不同輸入對模型的影響
* 第六 ~ 八次嘗試：嘗試調整批次數量、訓練數量對模型的影響
![image](https://github.com/NTUT104331030/NTUT_ML_midterm_homework/blob/master/Pic/severalTry.png)
