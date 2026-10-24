import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

input_dir=r"MS2_feature_processed_I_1_down_sampling_result_12_to_1"
output_dir=r"MS2_dataset_split_SMILES_type5_12_to_1_CV5"
databases=["GNPS","MONA","NIST20","NIST23"]
random_state=42

def add_split_label(df):
    df=df.copy()
    df["Scaffold_type"]=df["Scaffold"].apply(lambda x:"Acyclic" if str(x)=="Acyclic" else "Other")
    df["Feature_Mass_scale"]=df["Feature_Mass_scale"].astype(str)
    df["split_label"]=df["CF2"].astype(str)+"_"+df["Scaffold_type"].astype(str)+"_"+df["Feature_Mass_scale"]
    return df

def split_test_cv(df):
    df=add_split_label(df)
    sgkf_test=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=random_state)
    splits=list(sgkf_test.split(df,y=df["split_label"],groups=df["Canonical_SMILES"]))
    _, test_idx = splits[9]
    test_df=df.iloc[test_idx].copy()
    remain_df=df.drop(df.index[test_idx]).reset_index(drop=True)
    sgkf_cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=random_state)
    folds=[]
    for fold,(train_idx,val_idx) in enumerate(sgkf_cv.split(remain_df,y=remain_df["split_label"],groups=remain_df["Canonical_SMILES"]),1):
        train=remain_df.iloc[train_idx].drop(columns=["split_label"])
        val=remain_df.iloc[val_idx].drop(columns=["split_label"])
        folds.append((fold,train,val))
    test_df=test_df.drop(columns=["split_label"])
    return folds,test_df

def check_split(train,val,test):
    train_s=set(train["Canonical_SMILES"])
    val_s=set(val["Canonical_SMILES"])
    test_s=set(test["Canonical_SMILES"])
    print("SMILES overlap:")
    print("train-val:",len(train_s&val_s),"train-test:",len(train_s&test_s),"val-test:",len(val_s&test_s))
    total=len(train)+len(val)+len(test)
    print("Ratio:",round(len(train)/total*100,2),round(len(val)/total*100,2),round(len(test)/total*100,2))
    for name,df in {"train":train,"val":val,"test":test}.items():
        print(name,"PFAS:",(df["CF2"]==1).sum(),"Non-PFAS:",(df["CF2"]==0).sum())

def save_csv(df,split_name,ion_mode):
    save_dir=os.path.join(output_dir,split_name,ion_mode)
    os.makedirs(save_dir,exist_ok=True)
    for i,start in enumerate(range(0,len(df),100000),1):
        df.iloc[start:start+100000].to_csv(os.path.join(save_dir,f"{i}.csv"),index=False)

for ion_mode in ["N","P"]:
    print("\nProcessing:",ion_mode)
    dfs=[]
    for db in databases:
        path=os.path.join(input_dir,ion_mode,db)
        if not os.path.exists(path):
            continue
        for file in sorted(os.listdir(path)):
            if file.endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(path,file),low_memory=False))
    data=pd.concat(dfs,ignore_index=True).dropna(subset=["Canonical_SMILES"])
    print("Total:",len(data))
    folds,test_df=split_test_cv(data)
    print("Test:",len(test_df))
    save_csv(test_df,"test",ion_mode)
    for fold,train,val in folds:
        print("\nFold",fold,"Train:",len(train),"Val:",len(val))
        check_split(train,val,test_df)
        save_csv(train,f"fold{fold}/train",ion_mode)
        save_csv(val,f"fold{fold}/val",ion_mode)

print("\nAll finish")