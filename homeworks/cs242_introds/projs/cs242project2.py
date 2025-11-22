# Refer to .ipynb file

import pandas as pd
from plotnine import *

# Heart Rate Data
h_df = pd.read_csv("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/Project 2/Fitabase Data 4.12.16-5.12.16/heartrate_seconds_merged.csv")

# Unique Ids
users = h_df['Id'].unique()[:5]

#heart rate data for each user
h_df2 = h_df[h_df['Id'].isin(users)]

# Plot heart rate data for each user
plot = ggplot(h_df2) + geom_point(aes(x='Time', y='Value', group='Id')) + facet_wrap('~Id')
plot 