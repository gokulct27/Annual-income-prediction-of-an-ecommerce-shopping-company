#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import streamlit as st
from pickle import load
from pickle import dump
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# In[2]:


loaded_model=load(open('svm_model.sav','rb'))


# In[3]:


df=pd.read_csv("Ecommerce.csv")


# In[4]:


df=df.drop("Customer ID",axis=1)


# In[34]:


df=df.rename({'Avg Session length':'Avg_Session_length','Time on App':'Time_on_App','Time on Website':'Time_on_Website','Length of MemberShip':'Length_of_MemberShip','Yealy amount spent':'Yealy_amount_spent'},axis=1)


# In[ ]:


st.subheader("sample data")


# In[ ]:


st.write(df)


# In[36]:


def welcome():
    return 'welcome all'


# In[37]:


def prediction(Avg_Session_length,Time_on_App,Time_on_Website,Length_of_MemberShip):  
   
    prediction = loaded_model.predict(
        [[Avg_Session_length,Time_on_App,Time_on_Website,Length_of_MemberShip]])
    print(prediction)
    return prediction


# In[6]:


def main():
    st.title('Model Deployment: Support Vector Regresor')
    
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">ecommerce annual income prediction </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    
    
    Avg_Session_length= st.text_input("Avg Session length", "")
    Time_on_App= st.text_input("Time on App", "")
    Time_on_Website= st.text_input("Time on Website", "")
    Length_of_MemberShip= st.text_input("Length of MemberShip", "")
    result =""
    
    if st.button("Predict"):
        result = prediction(Avg_Session_length,Time_on_App,Time_on_Website,Length_of_MemberShip)
    st.success('The output is {}'.format(result))
if __name__=='__main__':
    main()


# In[ ]:




