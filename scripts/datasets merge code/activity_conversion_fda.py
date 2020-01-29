import pandas as pd

filename = 'FDA_femaleFertility_rat'

df = pd.read_csv(filename + '.csv')
fda_actiivity_col = 'ACTIVITY'
activity = []

for i in range(len(df)):
    if df.loc[i, fda_actiivity_col]  > 30:
        activity.append(1)
    elif df.loc[i, fda_actiivity_col]  < 20:
        activity.append(0)
    else:
        activity.append('inconclusive')

df['Activity_final'] = activity

df.to_csv(filename + 'with_activity.csv')