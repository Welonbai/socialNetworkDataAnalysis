# socialNetworkDataAnalysis

Link Prediction 演算法流程與實現:  
我選擇用networkx套件建立graph，並且是將train和test的node一起放進graph，才不會發生test的node不在graph裡的情況。  
根據建立出的graph，我計算出common_neighbor，Jaccard_coefficients，shortest_path_length，Adamic_Adar作為額外feature，來和原先的node1和node2一起作為features放進model訓練。  
我用了Ensemble的方法，使用了randomForestClassifier, SVM, logisticRegression做結合，並使用votingClassifier的soft voting設定來做ensemble。  
(我發現若用五種classifier包括adaboost, gradientboost, decisionTree, extraTree, KNN，來做ensemble的話，效果極差，故選擇了SVM和logisticRegression)  
我另外分出了資料的一部分來做accuracy測試，避免kaggle的提交次數用完。  

如何跑我的檔案:  
請將三個csv檔案和py檔放在同一個folder，再run py檔即可得到答案  
