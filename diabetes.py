#pip install streamlit
#pip install pandas
#pip install sklearn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle



df = pd.read_csv('diabetes.csv')


col1, col2, col3 = st.columns([10, 6, 10])
with col1:
    st.image("undraw_medicine_b-1-ol.svg")
with col2:
    st.write("")
with col3:
    st.write("")



st.markdown("<h1 style='text-align: center; color:#99ffff;'>Cardify:health  Disease Diagonistic App</h1>", unsafe_allow_html = True)
st.markdown("<h3 style='text-align: center; color:#99ffff;'>Check your Diabetic health for free💝💗.</h3>", unsafe_allow_html = True)

st.markdown("<h3 style='text-align: center; color:#9854ff;'>Its good to know your body and mind to have healthy life💝💗.</h3>", unsafe_allow_html = True)


# HEADINGS
# st.title('Diabetes Checkup')
# st.sidebar.header('Patient Data')
# st.subheader('Training Data Stats')
# st.write(df.describe())

# st.write(""" Visualization""")
# st.bar_chart(df)
# X AND Y DATA
# x = df.drop(['Outcome'], axis = 1)
# y = df.iloc[:, -1]
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

st.sidebar.markdown("""
Input your data here .
It is already set to normal values.
""")
# FUNCTION

pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
glucose = st.sidebar.slider('Glucose', 0,200, 120 )
bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
insulin = st.sidebar.slider('Insulin', 0,846, 79 )
bmi = st.sidebar.slider('BMI', 0,67, 20 )
dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
age = st.sidebar.slider('Age', 21,88, 33 )


st.markdown("<h3 style='text-align: center; color:#4dffa6;'>Update your details in the sidebar</h3>", unsafe_allow_html = True)
st.markdown("<h3 style='text-align: center; color:#4dffa6;'><----</h3>", unsafe_allow_html = True)
if st.sidebar.button('Submit'):
    datas = {
            'pregnancies':pregnancies,
            'glucose':glucose,
            'bp':bp,
            'skinthickness':skinthickness,
            'insulin':insulin, 
            'bmi':bmi,
            'dpf':dpf,
            'age':age}
  
    features = pd.DataFrame(datas, index=[0])
    st.markdown("<h2 style='text-align: center; color:#000066;'>Data gathered........</h2>", unsafe_allow_html = True)
    st.markdown("<h2 style='text-align: center; color:#000066;'>Processing Results........</h2>", unsafe_allow_html = True)

    # Reads in saved classification model
    load_clf = pickle.load(open('trained_model_dia.sav', 'rb'))
    # Apply model to make predictions
    prediction = load_clf.predict(features)
    prediction_proba = load_clf.predict_proba(features).reshape(2,)
    yes = prediction_proba[1]
    no = prediction_proba[0]

    # COLOR FUNCTION
    if prediction_proba[0]==0:
        color = '#99ffff'
    else:
        color = 'red'     
  





# PATIENT DATA
# user_datas = user_report()
# st.subheader('Patient Data')
# st.write(datas)




# MODEL
# rf  = RandomForestClassifier()
# rf.fit(x_train, y_train)
# user_resul = rf.predict(user_datas)


# user_result = user_resul.predict_proba(user_data).reshape(2,)
# yes = user_result[1]
# no = user_result[0]


# VISUALISATIONS
st.title('Visualised Patient Report')



# COLOR FUNCTION
# if user_result[0]==0:
#   color = 'blue'
# else:
#   color = 'red'


# Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = datas['age'], y = datas['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)



# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = datas['age'], y = datas['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)



# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = datas['age'], y = datas['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = datas['age'], y = datas['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = datas['age'], y = datas['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = datas['age'], y = datas['bmi'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


# Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = datas['age'], y = datas['dpf'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)



# OUTPUT


st.markdown("<h2 style='text-align: center; color:#99ffff;'><u>Prediction of your cardiac health </u></h2>", unsafe_allow_html = True)
pred1, pred2, pred3 = st.columns([12, 6, 14])
if prediction==0:
    st.markdown("<h1 style='text-align: center; color:#006600;'>You don't have any heart problem.</h1>", unsafe_allow_html = True)
    with pred1:
        st.write("")
    with pred2:
        st.image("smile_emo.png")
    with pred3:
        st.write("")
else:
    st.markdown("<h1 style='text-align: center; color:#cc0000;'>Go to a doctor.You may have heart problems.</h1>", unsafe_allow_html = True)
    st.markdown("<h1 style='text-align: center; color:#cc0000;'>From now on keep an healthy lifestyle and be happy.</h1>", unsafe_allow_html = True)

    with pred1:
        st.write("")
    with pred2:
        st.image("amb.png")
    with pred3:
        st.write("")
        
        
import time

my_bar = st.progress(0)

for percent_complete in range(100):
    time.sleep(0.001)
    my_bar.progress(percent_complete + 1)    
st.markdown("<h1 style='text-align: center; color:#7f7bf2;'><u>Prediction Probability</u></h1>", unsafe_allow_html = True)
fig,ax=plt.subplots(figsize=(10,8))
axes=plt.bar(['Chances of being healthy\n{} %'.format(no*100),'You are Diabetic\n{} %'.format(yes*100)], [no, yes])
axes[0].set_color('#99ffff')
axes[1].set_color('r')
st.pyplot(fig)
        
# vid1, vid2, vid3 = st.columns([100, 100, 100])
st.markdown("<h1 style='text-align: center; color:#99ffff;'>Always be happy like the small childrean who always laugh</h1>", unsafe_allow_html = True)
# with vid1:
st.video("production ID_4982409.mp4")
st.markdown("<h1 style='text-align: center; color:#99ffff;'>Have an healthy life style, eat healthy!</h1>", unsafe_allow_html = True)

# with vid2:
st.video("pexels-polina-kovaleva-5645055.mp4")
st.markdown("<h1 style='text-align: center; color:#99ffff;'>Do exercise regularly do some activity.</h1>", unsafe_allow_html = True)
# with vid2:
st.video("video (2).mp4")






st.image("undraw_doctor_kw-5-l.svg")
st.markdown("<h2>Developed with ❤ by Xavier Fernandes <a style='display: block; text-align: center;' href='https://www.linkedin.com/in/xavier-fernandes-938b6b223/' target='_blank'>Xavier Fernandes</a></h2>",unsafe_allow_html = True)


st.sidebar.markdown("""Follow me on [Kaggle](https://www.kaggle.com/xfflives) , [Instagram](https://www.instagram.com/Xavi_matirxx) , [Github](https://github.com/Xavi143)""")
st.sidebar.markdown("""Know more about me [Xavier Fernandes]()
For any queries email me on ***fernandescity143@gmail.com***
         
         All rights reserved.""")

# st.subheader('Your Report: ')
# output=''
# if user_result[0]==0:
#   output = 'You are not Diabetic'
# else:
#   output = 'You are Diabetic'
# st.title(output)
