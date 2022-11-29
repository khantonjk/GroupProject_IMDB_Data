# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:39:07 2022
Machine Learning Project
@author: Sebastian Clyne
"""


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#If in different place than same directory as code
Directory_Training_Data = r"C:\Users\anton\OneDrive\Dokument\Uppsala Universitet\SML\Project_main_rep\GroupProject_IMDB_Data\train.csv"
#Directory_Training_Data = "train.csv" 
Training_Data_df = pd.read_csv(Directory_Training_Data)
#Import directory for train data


#All Categories except "Lead"
Categories = ["Number words female", "Total words", "Number of words lead", \
              "Difference in words lead and co-lead", "Number of male actors",\
              "Year", "Number of female actors", "Number words male", "Gross",\
              "Mean Age Male", "Mean Age Female", "Age Lead", "Age Co-Lead"]


AllCategories = ["Number words female", "Total words", "Number of words lead", \
              "Difference in words lead and co-lead", "Number of male actors",\
              "Year", "Number of female actors", "Number words male", "Gross",\
              "Mean Age Male", "Mean Age Female", "Age Lead", "Age Co-Lead"\
              , "Lead"]






#Female respresentation in lead and nonlead roles over time
Years = Training_Data_df["Year"].unique()
Years.sort()

PercentageFemaleLead = []
PercentageFemaleActors = []

for i in Years:
    Data_To_Be_Used = Training_Data_df.loc[(Training_Data_df["Year"] == i), \
                                           AllCategories]

    PercentageLead = Data_To_Be_Used[Data_To_Be_Used["Lead"] == "Female"].shape[0] \
                    /len(Data_To_Be_Used)*100
    PercentageFemaleLead.append(PercentageLead) 

    PercentageFemale = 100*Data_To_Be_Used["Number of female actors"].sum() \
                             /(Data_To_Be_Used["Number of female actors"].sum()+
                             Data_To_Be_Used["Number of male actors"].sum())
    PercentageFemaleActors.append(PercentageFemale)
    
    
    


plt.figure(1)
plt.ylabel("Percentage(%)")
plt.xlabel("Year")
plt.plot(Years, PercentageFemaleActors,'.', label="Percentage Female Actors")
plt.plot(Years, PercentageFemaleLead,'.', label="Percentage Female Lead Actors")    
plt.legend()









Category_xaxis = "Year"
Category_yaxis = "Gross"

FemaleDF = Training_Data_df.loc[(Training_Data_df["Number words male"] <= 
                           Training_Data_df["Number words female"]), \
                                           AllCategories]
MaleDF = Training_Data_df.loc[(Training_Data_df["Number words female"] < 
                           Training_Data_df["Number words male"]), \
                             AllCategories]

print(f'The average gross for a movie were males speak more or the same amount is: {MaleDF["Gross"].mean()}')
print(f'The Percentage of movies with males speak more is: {100*len(MaleDF)/len(Training_Data_df)}%')
print(f'The average gross for a movie were females speak more is: {FemaleDF["Gross"].mean()}')


plt.figure(2)
plt.xlabel(Category_xaxis)
plt.ylabel(Category_yaxis)
plt.plot(MaleDF[Category_xaxis],MaleDF[Category_yaxis],'.', \
         label='Males Speak More')
plt.plot(FemaleDF[Category_xaxis],FemaleDF[Category_yaxis],'.', \
         label='Females Speak More')

plt.legend()







FemaleDF = Training_Data_df[Training_Data_df["Lead"] == "Female"]
MaleDF = Training_Data_df[Training_Data_df["Lead"] == "Male"]


print(f'The average gross for a movie with a male lead is: {MaleDF["Gross"].mean()}')
print(f'The average gross for a movie were female lead is: {FemaleDF["Gross"].mean()}')







#Plot data where female or male are the lead actors
plt.figure(3)

# Choose Data to plot
Category_xaxis = "Number of female actors"
Category_yaxis = "Gross"


# Extract Male and Female Data
Male_Lead = Training_Data_df.loc[(Training_Data_df["Lead"] == "Male"), \
                                 Categories]
Female_Lead = Training_Data_df.loc[(Training_Data_df["Lead"] == "Female"), \
                                 Categories]

plt.xlabel(Category_xaxis)
plt.ylabel(Category_yaxis)
plt.plot(Male_Lead[Category_xaxis],Male_Lead[Category_yaxis],'.', \
         label='Male Lead')
plt.plot(Female_Lead[Category_xaxis],Female_Lead[Category_yaxis],'.', \
         label='Female Lead')

plt.legend()


plt.show() # Använd om du vill se graferna


