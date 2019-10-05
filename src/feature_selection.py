import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF_THRESHOLD = 5

df = pd.read_pickle("../data/sanitized.pkl")

# Find (better) response variable
pearson_correlation = df['Value'].corr(df['Wage'], method="pearson")
if pearson_correlation > 0.8:
    df = df.drop('Wage', axis=1)


positions = df.loc[:, 'LS':'RB']

attack_positions = positions.loc[:, 'LS':'RM']

attack_positions = add_constant(attack_positions)
attack_vifs = pd.Series([variance_inflation_factor(attack_positions.values, i)
                         for i in range(attack_positions.shape[1])], index=attack_positions.columns)


min_vif = attack_vifs.min()
max_vif = attack_vifs.idxmax()
if min_vif > VIF_THRESHOLD:
    df = df.drop(list(attack_positions.drop(['LS', 'const'], 1)), 1)
else:
    df = df.drop(list(attack_positions.drop(max_vif, 1)), 1)

defense_positions = positions.loc[:, 'LWB':'RB']

defense_positions = add_constant(defense_positions)
defense_vifs = pd.Series([variance_inflation_factor(defense_positions.values, i)
                         for i in range(defense_positions.shape[1])], index=defense_positions.columns)


min_vif = defense_vifs.min()
max_vif = defense_vifs.idxmax()
if min_vif > VIF_THRESHOLD:
    df = df.drop(list(defense_positions.drop(['LB', 'const'], 1)), 1)
else:
    df = df.drop(list(defense_positions.drop(max_vif, 1)), 1)

# No need for goalkeeping stats
df = df.drop(list(df.loc[:, 'GKDiving':'GKReflexes']), 1)

df.to_pickle("../data/pruned.pkl")

