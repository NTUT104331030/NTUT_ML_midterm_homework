# 北科大 機器學習 期中作業

----
## 作法說明
see [Wikipedia](https://en.wikipedia.org/wiki/Markdown)

> 
----
## 程式流程圖
1. Write markdown text in this textarea.
2. Click 'HTML Preview' button.

----
## 結果分析
# headers

*emphasis*

**strong**
    

* list

>block quote

    code (4 spaces indent)
[links](https://wikipedia.org)

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
