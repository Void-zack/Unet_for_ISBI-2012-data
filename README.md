# Unet_for_ISBI-2012-data

## About the data & result
- All of the data are in the **/ISBI_data** directory,It contains two directories：**/Train** , **/Test** .
This folders should have the following tree:
```
    ISBI_data
     │
     ├───train
     │ ├───raw.tif
     │ └───mask.tif
     │
     └───test
       ├───raw.tif
       └───mask.tif
``` 
- The result will be saved in the **/result** directory.

## How to use?
- Run the **data_train.py** to train: 
`python data_train.py`

- Run the **data_predict.py** to predict: 
`python data_predict.py`

- Run the **data_compare.py** with the 'imageID' to compare the Ground Truth with the Result: 
`python data_compare.py imageID`

## About the configuration

- You can change all the configuration in **cofig.txt** .

- If you want to use your own dataset，please change the **train_path** , **test_path** , **raw_file** and **mask_file**
