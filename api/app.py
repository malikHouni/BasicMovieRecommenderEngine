import numpy as np
import pandas as pd
from scipy.sparse.linalg import *


movies=pd.read_csv('https://raw.githubusercontent.com/ChicagoBoothML/DATA___MovieLens___1M/master/movies.dat',sep="::",encoding='latin',header=None,names=["MovieID","Title","Genres"])
ratings=pd.read_csv('https://raw.githubusercontent.com/ChicagoBoothML/DATA___MovieLens___1M/master/ratings.dat',sep="::",encoding='latin',header=None,names=["UserID","MovieID","Rating","Timestamp"])
users=pd.read_csv('https://raw.githubusercontent.com/ChicagoBoothML/DATA___MovieLens___1M/master/users.dat',sep="::",encoding='latin',header=None,names=["UserID","Gender","Age","Occupation","Zip-code"])

def recommendation(person,df_ratings):
  df_beforePivot= pd.merge(movies, df_ratings,on="MovieID")
  df_users_ratings_films = df_beforePivot.pivot_table(values="Rating",index="UserID",columns="Title").fillna(0)
  #getting R
  R=df_users_ratings_films.values
  User_ratings_mean = np.mean(R,axis=1)
  R_demeaned = R - User_ratings_mean.reshape(-1,1)
  #Matrix Factorization with S.V.D
  U,sigma,Vt=svds(R,k=50)
  sigma =np.diag(sigma)
  all_user_predicted_ratings=np.dot(np.dot(U,sigma),Vt) + User_ratings_mean.reshape(-1,1)
  df_predictions=pd.DataFrame(all_user_predicted_ratings,index=df_users_ratings_films.index,columns=df_users_ratings_films.columns)
  #Getting the recommendation of the person

  currentListToRecommend={}
  for filmId in df_users_ratings_films.columns:
      if df_users_ratings_films.loc[person][filmId] == 0.0:
          currentListToRecommend.setdefault(filmId)
          currentListToRecommend[filmId]=df_predictions.loc[person][filmId]
          
  FinalListRecommendation=[(value,key)  for key,value in currentListToRecommend.items()]
  FinalListRecommendation.sort()
  FinalListRecommendation.reverse()
  return FinalListRecommendation[0:10]

def addUser(Name,Gender,Occupation,ZipCode,DataFrameUsers,currentUserId):
  currentUserId=users.count()[0]+1
  DataFrameUsers=DataFrameUsers.append({"UserID":users.count()[0]+1,"Gender":Gender,"Age":15,"Occupation":10,"Zip-code":ZipCode},ignore_index=True)
  return [DataFrameUsers,currentUserId]


def getAllMovies():
  return movies['Title'].values.tolist()

def addNewUserRatings(listOfMoviesLiked,DataFrameUsersRatings,dfUser):
  for i in range(0,len(listOfMoviesLiked)):
    DataFrameUsersRatings=DataFrameUsersRatings.append({"UserID":dfUser.count()[0]+1,"MovieID":movies[movies["Title"]==listOfMoviesLiked[i]]["MovieID"].values[0],"Rating":6,"Timestamp":10},ignore_index=True)
  return DataFrameUsersRatings

def removeStupidSpaceInJson(currArray):
  newArray=[]
  for i in range(0,len(currArray)):
    currElem=currArray[i].lstrip(currArray[i][0])#remove first elem in string
    currElem=currElem[:-1]#remove last elem in string
    newArray.append(currElem)
  return newArray


from flask import Flask, render_template, request, jsonify
app = Flask(__name__, template_folder='./') 
  
@app.route("/")
def home():
    return render_template('index.html',data=getAllMovies())

@app.route("/routeListOfMoviesLiked" ,methods=["GET", "POST"])
def getMyRecommendation():
    if request.method == 'POST':
      data = request.json
      currentUserId=users.count()[0]+1
      dfUsers=addUser("8888","F","Journalist","72000",users,currentUserId)[0]
      dfRatings=addNewUserRatings(removeStupidSpaceInJson(data),ratings,dfUsers)
      resRecommendation=recommendation(dfUsers.count()[0]+1,dfRatings)
      return jsonify(resRecommendation)
    return render_template("index.html")

app.run()
