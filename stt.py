import streamlit as st
import pandas as pd
import pickle
import shap
from streamlit_shap import st_shap

from PIL import Image


st.set_page_config(page_title="LoL Game Strategy Estimator",layout="centered") 

image = Image.open('leageu.jpg')
st.image(image)
st.title('🎮  Game Strategy Estimator Demo')
left,mid,right=st.columns(3)
with left:
     kda=st.number_input('KDA')
     champExperience_min=st.number_input('champExperience Min')
     totalDamageDealt_min=st.number_input('totalDamageDealt Min')
with mid:       
     dragonKills=st.number_input('dragonKills')
     damageDealtToBuildings=st.number_input('damageDealtToBuildings')
     turretsLost= st.number_input('turret lost')
with right:      
     turretTakedowns=st.number_input('turret Takedowns')
     inhibitorsLost=st.number_input('inhibitors Lost')
     inhibitorTakedowns=st.number_input('inhibitor Takedowns')

data = {'turretsLost': turretsLost,
            'turretTakedowns': turretTakedowns,
            'inhibitorsLost': inhibitorsLost,
            'inhibitorTakedowns': inhibitorTakedowns,
            'kda': kda,
            'champExperience': champExperience_min,
            'totalDamageDealt':totalDamageDealt_min ,
            'dragonKills': dragonKills,
            'damageDealtToBuildings': damageDealtToBuildings,
            }
df = pd.DataFrame(data, index=[0]) 

with open('cbt.pkl', 'rb') as f:
     model = pickle.load(f)

prediction=model.predict_proba(df)

if st.button('Predict Win Probabilities'):
            st.success("The Probability of Winning is " + str(round(prediction[0][1],4)))
            explainer = shap.Explainer(model)
            shap_values = explainer(df)
            st_shap(shap.plots.waterfall(shap_values[0]))

