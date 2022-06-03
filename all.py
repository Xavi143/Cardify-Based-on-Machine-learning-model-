import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# 
st.markdown("""
<nav class="navbar fixed-bottom navbar-expand-lg navbar-dark" style="background-color: black;">
  <a class="navbar-brand" href="" target="_blank">CARDIFY</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
       <li class="nav-item">
        <a class="nav-link" href="#heart-prediction">Heart Disease prediction</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#diabetes-prediction">Diabetes prediction</a>
      </li>
    </ul>
   </div>
</nav>
""", unsafe_allow_html=True)

# 
st.markdown('''
## Heart prediction
''')
st.image('heartt.svg')
st.markdown("<h3 style='text-align:; color:#99ffff;'>World Health Organization has  estimated 12 million deaths occur worldwide, every year due to Heart diseases. Half the deaths in the United States and other developed countries are due to cardio vascular diseases. The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications. This research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using logistic regression.</h3>", unsafe_allow_html = True)

df = pd.read_csv('processed heart disease.csv')
# 
def value(lst, string):
    for i in range(len(lst)):
        if lst[i] == string:
            return i
sex=['Female', 'Male']
edu=['10th pass', '12th pass/Diploma', 'Bachelors', 'Masters or Higher']
yn=['NO', 'YES']

col1, col2, col3 = st.columns([10, 6, 10])
with col1:
    st.image("undraw_medicine_b-1-ol.svg")
with col2:
    st.write("")
with col3:
    st.write("")
    
st.video('https://www.youtube.com/watch?v=xBAvxnT0ZvI') 
st.markdown("<h1 style='text-align: center; color:#99ffff;'>Cardify:Heart Disease Diagonistic App</h1>", unsafe_allow_html = True)
st.markdown("<h3 style='text-align: center; color:#99ffff;'>Check your cardiac health for free💝💗.</h3>", unsafe_allow_html = True)

st.markdown("<h3 style='text-align: center; color:#9854ff;'>Its good to know your body and mind to have healthy life💝💗.</h3>", unsafe_allow_html = True)
st.markdown("""

#### Each attribute is a potential risk factor. There are both demographic, behavioural and medical risk factors.

### Demographic: sex: male or female;(Nominal)

### age: age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
## Behavioural

### currentSmoker: whether or not the patient is a current smoker (Nominal)

### cigsPerDay: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarretts, even half a cigarette.)

## Medical( history):

### BPMeds: whether or not the patient was on blood pressure medication (Nominal)

### prevalentStroke: whether or not the patient had previously had a stroke (Nominal)

### prevalentHyp: whether or not the patient was hypertensive (Nominal)

### diabetes: whether or not the patient had diabetes (Nominal)

## Medical(current):

### totChol: total cholesterol level (Continuous)

### sysBP: systolic blood pressure (Continuous)

### diaBP: diastolic blood pressure (Continuous)

### BMI: Body Mass Index (Continuous)

### heartRate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)

### glucose: glucose level (Continuous)

## Predict variable (desired target):

### 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)""")


main_bg = "undraw_healthy_habit_bh-5-w"
main_bg_ext = "svg"
st.markdown("""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.header('User Input Features')



st.sidebar.markdown("""
Input your data here .
It is already set to normal values.
""")
male = st.sidebar.selectbox('Sex', ('Female', 'Male'))
age= st.sidebar.slider('Age', 5.0, 100.0, 30.0)
education = st.sidebar.selectbox('Education', ('10th pass', '12th pass/Diploma', 'Bachelors', 'Masters or Higher'))
current_smoker = st.sidebar.selectbox('Current Smoker', ('NO', 'YES'))
cigsPerDay = st.sidebar.slider('Cigarettes per Day', 0, 100, 20)
BPMeds = st.sidebar.selectbox('Takes BP medicines', ('NO', 'YES'))
prevstrk = st.sidebar.selectbox('Had any prevalent Stroke', ('NO', 'YES'))
prevhyp = st.sidebar.selectbox('Had any prevalent Hypertension', ('NO', 'YES'))
diabetes = st.sidebar.selectbox('Have diabetes', ('NO', 'YES'))
chol = st.sidebar.slider('Cholesterol (mg/dl)', 0.0, 700.0, 230.0)
highbp = st.sidebar.slider('Blood Pressure(upper value) (mmHg)', 100.0, 250.0, 120.0)
lowbp = st.sidebar.slider('Blood Pressure(Lower Value) (mmHg)', 50.0, 180.0, 80.0)
BMI = st.sidebar.slider('BMI (kg/m^2)', 15.0, 70.0, 23.0)
heart_rate = st.sidebar.slider('Heart Rate (per minute)', 30.0, 130.0, 40.0)
glucose = st.sidebar.slider('Glucose (mg/dl)', 100.0, 500.0, 110.0)

st.markdown("<h3 style='text-align: center; color:#4dffa6;'>Update your details in the sidebar</h3>", unsafe_allow_html = True)
st.markdown("<h3 style='text-align: center; color:#4dffa6;'><----</h3>", unsafe_allow_html = True)
if st.sidebar.button('Submit'):
        datas = {'male': value(sex, male),
                'age': age,
                'education': value(edu, education),
                'currentSmoker': value(yn, current_smoker),
                'cigsPerDay': cigsPerDay,
                'BPMeds': value(yn, BPMeds),
                'prevalentStroke': value(yn, prevstrk),
                'prevalentHyp': value(yn, prevhyp),
                'diabetes': value(yn, diabetes),
                'totChol': chol,
                'sysBP': highbp,
                'diaBP': lowbp,
                'BMI': BMI,
                'heartRate': heart_rate,
                'glucose': glucose}
        features = pd.DataFrame(datas, index=[0])

        st.markdown("<h2 style='text-align: center; color:#000066;'>Data gathered........</h2>", unsafe_allow_html = True)
        st.markdown("<h2 style='text-align: center; color:#000066;'>Processing Results........</h2>", unsafe_allow_html = True)
        
        # Reads in saved classification model
        # load_clf = pickle.load(open('trained_model.sav', 'rb'))
        



        # Logistic Regression Model
        load_clf = pickle.load(open('heart_disease.pkl', 'rb'))



        # 

        # Apply model to make predictions
        prediction = load_clf.predict(features)
        prediction_proba = load_clf.predict_proba(features).reshape(2,)
        yes = prediction_proba[1]
        no = prediction_proba[0]
        
        # VIsualization 
        # st.title('Visualised Patient Report')
        # COLOR FUNCTION
        if prediction_proba[0]==0:
            color = '#99ffff'
        else:
            color = 'red'


        
            # Age vs Pregnancies
        # st.header('cigerates Per Day count Graph (Others vs Yours)')
        # fig_cig = plt.figure()
        # ax1 = sns.scatterplot(x = 'age', y = 'BPMeds', data = df, hue = 'TenYearCHD', palette = 'Greens')
        # ax2 = sns.scatterplot(x = data['age'], y = data['BPMeds'], s = 150, color = color)
        # plt.xticks(np.arange(10,100,5))
        # plt.yticks(np.arange(0,20,2))
        # plt.title('0 - Healthy & 1 - Unhealthy')
        # st.pyplot(fig_cig)
                # Age vs Glucose
        # st.header('Glucose Value Graph (Others vs Yours)')
        # fig_glucose = plt.figure()
        # ax3 = sns.scatterplot([{ 'x':age ,'y': glucose}], data = df, hue = 'TenYearCHD' , palette='magma')
        # ax4 = sns.scatterplot(x = datas['age'], y = datas['glucose'], s = 150, color = color)
        # plt.xticks(np.arange(10,100,5))
        # plt.yticks(np.arange(0,220,10))
        # plt.title('0 - Healthy & 1 - Unhealthy')
        # st.pyplot(fig_glucose)

                
        
        
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
        axes=plt.bar(['Chances of being healthy\n{} %'.format(no*100),'Chances of getting cardiac diseases\n{} %'.format(yes*100)], [no, yes])
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

#-----------------------------------------------------------------
  

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
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle


df = pd.read_csv('diabetes.csv')
col1, col2, col3 = st.columns([10, 6, 10])
with col1:
    st.image("undraw_workout_gcgu.svg")
with col2:
    st.write("")
with col3:
    st.write("")
# HEADINGS
st.markdown('''
## Diabetes prediction
''')
st.video('https://www.youtube.com/watch?v=wZAjVQWbMlE')
st.image('https://trends.google.com/trends/explore?geo=IN&q=cardiac%20disease,diabetes') 
st.markdown("<h1 style='text-align: center; color:#99ffff;'>Cardify:Health & Disease Diagonistic App</h1>", unsafe_allow_html = True)
st.markdown("<h3 style='text-align: center; color:#99ffff;'>Check your Diabetic health for free💝💗.</h3>", unsafe_allow_html = True)

st.markdown("<h3 style='text-align: center; color:#9854ff;'>Its good to know your body and mind to have healthy life💝💗.</h3>", unsafe_allow_html = True)


st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
# st.subheader('Training Data Stats')
# st.write(df.describe())


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
st.sidebar.markdown("""
Input your data here .
It is already set to normal values.
""")

# FUNCTION
def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data
  


st.markdown("<h2 style='text-align: center; color:#000066;'>Data gathered........</h2>", unsafe_allow_html = True)
st.markdown("<h2 style='text-align: center; color:#000066;'>Processing Results........</h2>", unsafe_allow_html = True)




# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)




# MODEL
# rf  = RandomForestClassifier()
rf  = LogisticRegression(solver='liblinear')

rf.fit(x_train, y_train)
user_result = rf.predict(user_data)
predict_proba = rf.predict_proba(user_data).reshape(2,)
yes = predict_proba[1]
no = predict_proba[0]


# VISUALISATIONS
st.title('Visualised Patient Report')



# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


# Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)



# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)



# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['age'], y = user_data['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


# Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['age'], y = user_data['dpf'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)



# OUTPUT


st.markdown("<h2 style='text-align: center; color:#99ffff;'><u>Prediction of your Diabetic health </u></h2>", unsafe_allow_html = True)
pred1, pred2, pred3 = st.columns([12, 6, 14])
if user_result ==0:
    st.markdown("<h1 style='text-align: center; color:#006600;'>You don't have any Diabetes problem.</h1>", unsafe_allow_html = True)
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

st.image("undraw_indoor_bike_pwa4.svg")

     
# vid1, vid2, vid3 = st.columns([100, 100, 100])
st.markdown("<h1 style='text-align: center; color:#99ffff;'>Always be happy like the small childrean who always laugh</h1>", unsafe_allow_html = True)

# with vid1:
st.video("production ID_4982409.mp4")
st.markdown("<h1 style='text-align: center; color:#99ffff;'>Have an healthy life style, eat healthy!</h1>", unsafe_allow_html = True)
st.image("undraw_hamburger_-8-ge6.svg") 
# with vid2:
st.video("pexels-polina-kovaleva-5645055.mp4")
st.markdown("<h1 style='text-align: center; color:#99ffff;'>Do exercise regularly do some activity.</h1>", unsafe_allow_html = True)
# with vid2:
st.image("undraw_healthy_habit_bh-5-w.svg")
st.video("video (2).mp4")






st.image("undraw_doctor_kw-5-l.svg")
st.markdown("<h2>Developed with ❤ by Xavier Fernandes <a style='display: block; text-align: center;' href='https://www.linkedin.com/in/xavier-fernandes-938b6b223/' target='_blank'>Xavier Fernandes</a></h2>",unsafe_allow_html = True)


st.sidebar.markdown("""Follow me on [Kaggle](https://www.kaggle.com/xfflives) , [Instagram](https://www.instagram.com/Xavi_matirxx) , [Github](https://github.com/Xavi143)""")
st.sidebar.markdown("""Know more about me [Xavier Fernandes]()
For any queries email me on ***fernandescity143@gmail.com***
         
         All rights reserved.""")


st.markdown("<h1 style='text-align: center; color:#99ffff;'>STAY HEALTHY</h1>", unsafe_allow_html = True)
