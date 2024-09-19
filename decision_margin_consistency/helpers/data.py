import os
import pandas as pd
import json
from glob import glob

def load_data(data_dir, nTrials=160):
    files = glob(os.path.join(data_dir, '*.txt'))
    print(f"Number of files: {len(files)}")
    df = None 
    for file in files:      
        with open(file) as f:
            data = json.loads(f.read())

        print("==> ", file)
        for key,value in data.items():
            if key != "trialData":
                if key == "totalTime":
                    print(f"{key}: {value/1000/60:4.2f}min")
                elif key in ['studyID','workerID','comments']:
                    print(f"{key}: {value}")

        df_ = pd.DataFrame(data['trialData'])
        assert len(df_)==nTrials, f"Ooops, expected {nTrials} trials, got {len(df_)}"
        df_['repeatNum'] = df_['trialsSinceLastPresented'].apply(lambda x: 1 if x==-1 else 2)
        df = pd.concat([df, df_])
    return df,data