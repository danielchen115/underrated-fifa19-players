import pandas as pd


def calculate_score(equation):
    return sum(map(int, (equation).split('+')))


def price_to_float(price):
    price = price[1:]
    try:
        price = float(price)
    except ValueError:
        price = float(price[:-1]) * (1000000 if price[-1] == 'M' else 1000 if price[-1] == 'K' else 1)
    return price


if __name__ == '__main__':
    df = pd.read_csv("../data/data.csv")

    df['Value'] = df['Value'].apply(lambda x: price_to_float(x))
    df['Wage'] = df['Wage'].apply(lambda x: price_to_float(x))

    # Only deal with outfield players for now
    players_df = df[df['Position'] != 'GK'].dropna()

    for column in range(10, 36):
        players_df[players_df.columns[column]] = players_df[players_df.columns[column]].apply(
            lambda x: calculate_score(x))

    players_df.to_pickle('../data/sanitized.pkl')
