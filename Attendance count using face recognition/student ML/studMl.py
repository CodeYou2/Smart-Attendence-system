import pandas as pd
df=pd.read_csv('D:\Works\Opencv project\Attendance count using face recognition\student ML\stuData.csv').dropna(axis=1)
def  name():
    name=input('Enter the name of the employee: ')
    mask=df['Name']==name
    df1=df[mask]
    df1.sort_values('Date',ascending=True)
    print(df1)
    return df1
name()