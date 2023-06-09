# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 20:35:13 2023

@author: shrey
"""

import numpy as np
import pickle as pkl
import streamlit as st
filepath="C:/Users/shrey/AI PROJECT/saved_model.sav"
load_model=pkl.load(open(filepath,"br"))
def pred(x):
    x=np.asarray(x).reshape(1,-1)
    result=load_model.predict(x)
    if result[0]==0:
        return("satosa")
    elif(result[0]==1):
        return("versicolor")
    else:
        return("verginica")
    
    
def main():
  sl=st.number_input("sepal-lenght")
  sw=st.number_input("sepal-width")    
  pl=st.number_input("petal-lenght") 
  pw=st.number_input("petal-width")
  data=[sl,sw,pl,pw]
  if st.button("Predict"):
      st.write(pred(data))
  if pred=="satosa":
      st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg")
  elif pred=="versicolor":
      st.image("https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/images/iris_versicolor/iris_versicolor_18_lg.jpg")
  else:
      st.image("https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_shrevei_tb2_lg.jpg")
if __name__ == "__main__":      
     main()