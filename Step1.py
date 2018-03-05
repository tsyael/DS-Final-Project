import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv("hotels_data.csv")
    print(df.describe())
    print(df.columns)
    print(df.head(5))
    df['Snapshot Date'] = pd.to_datetime(df['Snapshot Date'],format="%m/%d/%Y %H:%M", utc=True)
    df['Checkin Date'] = pd.to_datetime(df['Checkin Date'], format="%m/%d/%Y %H:%M", utc=True)
    df['DayDiff'] = (df['Checkin Date'] - df['Snapshot Date']).astype('timedelta64[D]')
    print(df.head(5)['DayDiff'])
    df['WeekDay'] = df['Checkin Date'].dt.dayofweek
    print(df.head(5)['WeekDay'])
    df['DiscountDiff'] = df['Original Price'] - df['Discount Price']
    df['DiscountPerc'] = (df['Discount Price']/df['Original Price'])
    print(df.head(5)['DiscountDiff'])
    print(df.head(5)['DiscountPerc'])

    df.to_csv("Hotels_data_Changed.csv", date_format='%m/%d/%Y')

