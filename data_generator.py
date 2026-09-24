import sdv
import pandas as pd

df = pd.read_csv(r"C:\Users\110095\Desktop\HR_Data\HR_data_real.csv") 
print(df.shape) 
print(df.dtypes) 
print(df.isnull().sum())