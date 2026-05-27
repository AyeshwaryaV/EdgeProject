import pandas as pd
files=['engine_data_synthetic.csv','engine_data.csv','engine_data_realistic.csv']
for fn in files:
    try:
        df = pd.read_csv(fn)
        print(fn, df.shape)
        print(df.columns.tolist())
        print(df.head(2).to_string(index=False))
        print('---')
    except Exception as e:
        print(fn,'ERROR',e)
