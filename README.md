# WDD_T2DMprediction
The code in this repository is for predicting T2DM. K-nearest neighbor imputation and two missing value tolerance weighted diversity density algorithms (WDD-KNN, MVT-WDD-DI and MVT-WDD-BF) were included.

For training code, the basic usage is:
'python <trainingFile> <hyper-parameters> <dataFile>' 

  For example:

WDD-KNN: python KNNtrain.py --gamma=0.125 --data_path=yourdata.csv --out_Path=/anyPath
  
MVT-WDD-DI: python DItrain.py --gamma=1 --delta=0.25 --data_path=yourdata.csv --out_Path=/anyPath
  
MVT-WDD-BF: python BFtrain.py --gamma=8 --lam=32 --data_path=yourdata.csv --out_Path=/anyPath

After training, the output will be stored in a .pk file called <trained-parametersFile>. The performance of the 10-fold cross validation will be stored in score.txt.  All of these output files could be found in out_Path.

For prediction code, the basic usage is:
'python <predictionFile> <hyper-parameters><trained-parametersFile> <dataFile>' 
For example:
WDD-KNN: python KNNprediction.py --gamma=0.125 --parapath=/your_path/parameter.pk --datapath=templatedata/template_data_fill.csv
MVT-WDD-DI: python DIprediction.py --gamma=1 --delta=0.25 --parapath=/your_path/DIpara.pk --datapath=templatedata/template_data_nofill.csv
MVT-WDD-BF: python BFprediction.py --gamma=8 --lam=32 --parapath=/your_path/BFpara.pk --datapath=templatedata/template_data_nofill.csv

After executing the script, the predcition will be printed at STDOUT as follows:
> python KNNprediction.py --gamma=0.125 --parapath=/your_path/parameter.pk --datapath=templatedata/template_data_fill.csv

>y_true: tensor([1, 0])
>y_pred: tensor([1, 0])

Below are the brief explanation of the parameters:
trainingFile: the file contains the training steps, currently the <Modelname>train.py files  in the root path of the repository are available. Besides, <Modelname>model.py files  in the root path is used as a library for  trainingFile.
dataFile : for both training and prediction steps, the data format should be the same as the .csv file in the folder templatedata.
trained-parametersFile：these parameters are obtained after model training, which are saved as the .pk file. These files can be found in folder parameter. 
hyper-parameters: gamma, delta, lam are the hyper-parameters for our algorithms, these parameters can expand and shrink the feature space. Therefore, finding appropriate hyperparameters can help us search for the optimal point. The values of these parameters may need to be adjusted again for different datasets and tasks. In our experience, the scope of parameter tuning is 2^-10 to 2^10.

Required extra packages: 
Pytorch>=1.12.0 
scikit-learn

