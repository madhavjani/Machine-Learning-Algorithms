import numpy as np
import pandas as pd
import sys

df = pd.read_csv("ecoli.data",sep='  ',names=["sequence_names", "mcg","gvh","lip","chg","aac","alm1","alm2","decision"])
df = df.drop("sequence_names", axis="columns")
df["decision"].replace(["cp", "im", "imU", "imS", "imL", "om", "omL", "pp"], [
                                  0, 1, 2, 3, 4, 5, 6, 7], inplace=True)
print(df)