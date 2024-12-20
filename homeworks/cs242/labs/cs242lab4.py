import pandas as pd 

df = pd.read_csv("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/employees.98.7.csv", parse_dates=["Start Date"])

def calc_years_working(dataframe):
    years = pd.DatetimeIndex(df["Start Date"]).year
    print(years)
    working = [2021] * len(years) - years
    print(working) 
    df["Years of Working"] = working
    return df
calc_years_working(df)


def bin_years_working(dataframe):
    # working = df["Years of Working"]
    df["Years of Working"] = pd.cut(x=df["Years of Working"], bins=[0, 5, 10, 20, 40, 2147483647], labels=["less than 5 years", "5 to 10 years", 
                                                                                               "10 to 20 years", "20 to 40 years", "more than 40 years"])
    return df
bin_years_working(calc_years_working(df))


def find_counts(dataframe):
    twenty_forty = dataframe["Years of Working"].value_counts("20 to 40 years")
    ten_twenty = dataframe["Years of Working"].value_counts("10 to 20 years")
    five_ten = dataframe["Years of Working"].value_counts("5 to 10 years")
    forty = dataframe["Years of Working"].value_counts("more than 40 years")
    five = dataframe["Years of Working"].value_counts("less than 5 years")
    return pd.value_counts(dataframe["Years of Working"])
    
find_counts(bin_years_working(calc_years_working(df)))


weather_ds1 = pd.read_csv("WLAF_weather_jan_apr.120.6.csv")
weather_ds2 = pd.read_csv("WLAF_weather_jan_apr_avg.120.3.csv")
weather_ds3 = pd.read_csv("WLAF_weather_mar_diu.31.2.csv")
weather_ds4 = pd.read_csv("WLAF_weather_may.31.7.csv")