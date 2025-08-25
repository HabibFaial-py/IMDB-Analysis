# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression

# Upload CSV
df = pd.read_csv("imdb_top_1000.csv")
df.head() # first 5 lines
df.info() # datatypes
df.describe() # basic stats for numeric columns
df.isnull().sum() # look out for missing values

# Clean Data
df["Gross"] = df["Gross"].astype(str).str.replace("," , "",regex = True).astype(float) # removing "," from a number
df["No_of_Votes"] = pd.to_numeric(df["No_of_Votes"],errors="coerce") # converting no of votes to numeric
df["Released_Year"] = pd.to_numeric(df["Released_Year"],errors="coerce") # converting released year to numeric
df.dropna(inplace = True) # removing missing values

# Advanced Cleaning
df["Genre"] = df["Genre"].fillna("").str.split(",") #removing "," from genre column
df = df.explode("Genre") # splitting the genres for a movie
df["Decade"] = (df["Released_Year"]//10)*10 # creating a decade column
df = df[df["No_of_Votes"]>50000] #filter lowvotes movies

# Analysis
top_movies = df.sort_values("IMDB_Rating" , ascending = False).head(10) # top 10 movies by imdb rating
avg_by_genre = df.groupby("Genre")["IMDB_Rating"].mean().sort_values(ascending = False) # avg rating of all genres
avg_by_decade = df.groupby("Decade")["IMDB_Rating"].mean() # avg rating by decades
df[["No_of_Votes" , "IMDB_Rating" , "Gross" , "Runtime"]].corr() # creating a corelation bw Votes and Rating

# Visualization
# (a)
plt.figure(figsize=(10,5)) # create size of chart
sns.barplot(x = avg_by_genre.index[:10] , y = avg_by_genre.values[:10] , palette = "magma") # plot the top 10 values 
plt.xticks(rotation = 45) # rotate xticks by 45 degrees
plt.ylim(0,10)
plt.title("Top Genres by Avg IMDB Rating") # display title of chart
plt.show() # discplay chart
# (b)
plt.figure(figsize = (8,5)) # create size of chart
sns.lineplot(x = avg_by_decade.index , y = avg_by_decade.values , marker = "o") # create a line chart
plt.title("Average Rating By Decade") # title 
plt.show() # display
# (c)
plt.figure(figsize = (7,5)) # create size of chart
sns.scatterplot(data = df , x="No_of_Votes" , y = "IMDB_Rating" , alpha = 0.5) # create a scatterplt
plt.title("Votes vs IMDB Rating") # title 
plt.show() # display
# (d)
plt.figure(figsize = (8,6))# create size of chart
sns.heatmap(df[["IMDB_Rating","No_of_Votes","Gross","Runtime"]].corr(), annot = True , cmap = "coolwarm" , fmt = ".2f") # create a heatmap of the correlation
plt.title("Correlation Heatmap") # title 
plt.show() # display
# (e)
text = " ".join(df["Series_Title"].dropna()) # joining titles 
wordcloud = WordCloud(width=800, height=400, background_color="black").generate(text) # determining design
plt.figure(figsize = (10,5)) # set fig size
plt.imshow(wordcloud , interpolation = "bilinear") # create visualiztion
plt.axis("off") # hide axis
plt.title("Most Common Words in Movie Titles") # title
plt.show()  # display

# Mini Predictive Model (ML)
x = df[["Gross","No_of_Votes","Runtime"]].fillna(0) #select 3 columns, empty cols are 0
y = df["IMDB_Rating"].fillna(0) # make ratings as independant variable
model = LinearRegression() # create Linear Regression Model
model.fit(x,y) # train the model 
print("R^2 Score" , model.score(x,y)) # prints how well the data is set


