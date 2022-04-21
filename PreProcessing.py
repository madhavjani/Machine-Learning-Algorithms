import numpy as np
import pandas as pd
import os

class Preprocessing:
    df = pd.read_csv("breast-cancer-wisconsin.data")
    df.rename(columns={"1000025": "ID_Number", "5":"Radius", "1":"Texture", "1.1":"Perimeter" , "1.2":"Area" , "2":"smoothness" , "1.3":"compаctness", "3":"concavity", "1.4":"concave_points", "1.5":"symmetry", "2.1":"fractal_dimension"}, inplace=True)
    df['compаctness'] = np.where(df['compаctness'] == "?",0 ,df.compаctness)
    df['compаctness'] = np.where(df['compаctness'] == 0,round(df['compаctness'].astype(str).astype(int).mean()) ,df.compаctness)
    # df.loc[df['compаctness'] == None]
    # df=df[df['compаctness']== None]
    а=1+2
    print(а)

print(Preprocessing.df)

