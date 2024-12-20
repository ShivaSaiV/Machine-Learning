# Reading the dataset (make sure to change the file path here)
# https://www.kaggle.com/datasets/maso0dahmed/netflix-movies-and-shows/ 


netflix_data = read.csv("/Users/shivasaivummaji/Downloads/imdb_movies_shows.csv")
View(netflix_data)

summary(netflix_data$title)

# Creating a new dataframe for just movies
movie_data = netflix_data[netflix_data$type == "MOVIE", ]
View(movie_data)

# Creating a new dataframe for just shows
show_data = netflix_data[netflix_data$type == "SHOW", ]
View(show_data)

# Relationship between number of movies and shows
counts = table(netflix_data$type)
barplot(counts, col = c("brown", "royalblue"))




# Release Year Histogram
hist(netflix_data$release_year, xlab = "Year")

# Release Year Boxplot
boxplot(netflix_data$release_year, main = "Boxplot for Release Year (All Data)")

# Release Year Histogram for movies
hist(movie_data$release_year, xlab = "Year")

# Release Year Histogram for shows
hist(show_data$release_year, xlab = "Year")

# Summary Statistics for release year overall
summary(netflix_data$release_year)

# Summary Statistics for release year movies
summary(movie_data$release_year)

# Summary Statistics for release year shows
summary(show_data$release_year)




# Age distribution overall
ages = table(netflix_data$age_certification)
barplot(ages, main = "Age Rating")
pie(ages)




# Runtime Histogram Overall
hist(netflix_data$runtime, xlab = "Minutes")

# Runtime Boxplot Overall
boxplot(netflix_data$runtime)

# Runtime Histogram for Movies
hist(movie_data$runtime, xlab = "Minutes")

# Boxplot for Runtime Movies
boxplot(movie_data$runtime)

# Runtime Histogram for Shows
hist(show_data$runtime, xlab = "Minutes")

# Boxplot for Runtime Shows
boxplot(show_data$runtime)

# Summary Statistics for runtime overall
summary(netflix_data$runtime)

# Summary Statistics for runtime of movies
summary(movie_data$runtime)

# Summary Statistics for runtime of shows
summary(show_data$runtime)




# Genres 
g = unlist(strsplit(gsub("\\[|\\]|'", "", netflix_data$genres), ", "))
g_df = data.frame(genre = g)
View(g_df)
g_c = table(g_df$genre)
barplot(g_c, cex.names = 0.6)
pie(g_c, cex = 0.7)
print(g_c)




# Production Countries
p = unlist(strsplit(gsub("\\[|\\]|'", "", netflix_data$production_countries), ", "))
p_df = data.frame(country = p)
p_c = table(p_df$country)
barplot(p_c, cex.names = 0.7)
pie(p_c, cex = 0.7)
print(p_c)




# Histogram for seasons for shows
hist(show_data$seasons)

# Boxplot for seasons for shows
boxplot(show_data$seasons)

# Summary Statistics for seasons for shows
summary(show_data$seasons)




# Histogram for imdb score
hist(netflix_data$imdb_score, xlab = "Score")

# Boxplot for imdb score
boxplot(netflix_data$imdb_score)

# Summary Statistics for imdb score
summary(netflix_data$imdb_score)




# Summary Statistics for imdb votes
summary(netflix_data$imdb_votes)

# Scatter plot imdb score vs. imdb votes
plot(netflix_data$imdb_votes, netflix_data$imdb_score, xlab = "IMDB Votes", ylab = "IMDB Score")

# Relationships between movies and shows

# IMDB Score
boxplot(netflix_data$imdb_score ~ netflix_data$type, xlab = "Type", ylab = "Score")

# IMDB Votes
boxplot(netflix_data$imdb_votes ~ netflix_data$type, xlab = "Type", ylab = "Votes")

# Age Rating vs. IMDB Score
boxplot(netflix_data$imdb_score ~ netflix_data$age_certification, xlab = "Age Certification", ylab = "Score", las = 1)

# Runtime vs. IMDB Score
plot(netflix_data$runtime, netflix_data$imdb_score, xlab = "Runtime", ylab = "IMDB Score")


# Research Question 1: Is there a significant difference in IMDB scores between movies and shows?

# I will use a independent two-sample t-test in order to answer this question
# as I have two independent groups (movies and TV shows), and I want to compare 
# their IMDB scores. The assumptions made in applying this method is that 
# the two groups are independent, and the IMDB score data is normally distributed with equal variances. 

# I can check the validity of the assumptions by examining the QQ plots for normality. 
imdb_movies = netflix_data$imdb_score[netflix_data$type == "MOVIE"]
imdb_shows = netflix_data$imdb_score[netflix_data$type == "SHOW"]
qqnorm(imdb_movies, main = "QQ Plot for IMDB Scores (Movies) ")
qqline(imdb_movies)
qqnorm(imdb_shows, main = "QQ Plot for IMDB Scores (Shows")
qqline(imdb_shows)
# As it is clear from the graphs, both the scatter plots follow similar to a straight line, which supports
# the fact that the data sets are normal. 

q1 = t.test(imdb_movies, imdb_shows)
print(q1)





# Research Question 2: 

# How do Runtime, Release Year, and the number of IMDB votes contribute to IMDB scores? 

# I will use Multiple Linear Regression in order to answer this question because I want to 
# understand the relationship between IMDB scores (continuous dependent variable) and 
# Runtime, Release Year, and the number of IMDB votes (independent variables). 
# The assumptions made in applying this method is the independence of variables,
# and normality (for residuals).
reg_model = lm(netflix_data$imdb_score ~ 
                 netflix_data$runtime + netflix_data$release_year + 
                 netflix_data$imdb_votes)
summary(reg_model)

# Checking for assumptions
res2 = residuals(reg_model)
qqnorm(res2, main = "QQPlot of Residuals (Question 2)")



# Research Question 3: 
# Is there a correlation between runtime and IMDb score?

# I will use Pearson Correlation in order to answer this question as I have continuous
# variables (the IMDB score is a rounded continuous), and I want to test their
# linear association. The assumptions made in applying this method is that 
# the relationship between the variables is linear, and that both variables are normally distributed. 

new_netflix = na.omit(netflix_data[c("runtime", "imdb_score")])
cor3 = cor(new_netflix$runtime, new_netflix$imdb_score, method = "pearson")
plot(new_netflix$runtime, new_netflix$imdb_score, xlab = "Runtime", 
     ylab = "IMDB score", cex = 0.7)
fit_line = lm(new_netflix$imdb_score ~ new_netflix$runtime)
abline(fit_line, col = "blue")
cor3

# Assumption Checks
qqnorm(new_netflix$runtime, main = "QQPlot for Runtime")
qqnorm(new_netflix$imdb_score, main = "QQPlot for IMDB Score")

total = nrow(new_netflix)
t_value = cor3 * sqrt((n - 2) / (1 - cor3^2))
p_value = 2 * pt(-abs(t_value), df = n - 2)
p_value
