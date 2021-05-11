import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)

# Load the pickled model
model = pickle.load(open('logisticmodel.pkl', 'rb')) 
dataset= pd.read_csv('Jatin_Tak__Classification_Dataset1.csv')

x = dataset.iloc[:,1:10].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
imputer = imputer.fit(x[:, 3: ]) 
x[:, 3: ]= imputer.transform(x[:, 3: ])  


imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Female', verbose=1, copy=True) 
imputer = imputer.fit(x[:, 2:3]) 
x[:, 2:3]= imputer.transform(x[:, 2:3])  

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 1] = labelencoder_x.fit_transform(x[:, 1])

labelencoder_x = LabelEncoder()
x[:, 2] = labelencoder_x.fit_transform(x[:, 2])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

def predict_note_authentication(UserID, CreditScore, Geography, Gender, Age, Tenure, Balance, HasCrCard, IsActiveMember ,EstimatedSalary):
  output= model.predict(sc.transform([[CreditScore, Geography, Gender, Age, Tenure, Balance, HasCrCard, IsActiveMember ,EstimatedSalary]]))
  print("Exited", output)
  if output==[1]:
    prediction="Customer will be Exited."
  else:
    prediction="Customer will not be Exited."
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Customer Prediction using Logistic Algorithm")
    UserID = st.text_input("UserID","")
    CreditScore = st.number_input('Insert Credit Score')
    Geography = st.number_input('Insert Geography France:0 Spain:1')
    Gender = st.number_input('Insert Gender Male:1 Female:0')
    Age = st.number_input('Insert Age')
    Tenure =  st.number_input('Insert Tenure')
    Balance =st.number_input('Insert Balance')
    HasCrCard = st.number_input('Credit Card Yes:1 No:0')
    IsActiveMember = st.number_input('Active Member Yes:1 No:0')
    EstimatedSalary =st.number_input('Insert Estimated Salary')
    resul=""
    if st.button("Prediction"):
      result=predict_note_authentication(UserID, CreditScore, Geography, Gender, Age, Tenure, Balance, HasCrCard, IsActiveMember ,EstimatedSalary)
      st.success('Model has predicted: {}'.format(result))  
    if st.button("About"):
      st.header("Developed by Jatin Tak")
      st.subheader("Student, Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Model Prediction</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()