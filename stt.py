import streamlit as st
import pandas as pd
import pickle
import shap
from streamlit_shap import st_shap
from PIL import Image


st.set_page_config(page_title="LoL Game Strategy Estimator",layout="centered") 

image = Image.open('leageu.jpg')
st.image(image)
st.title('🎮 Game Strategy Estimator Demo')
left,mid=st.columns(2)
with left:
     st.write("#### Player Performance")
     kda=st.number_input('KDA')
     timePlayed=st.number_input('time Played')
     goldEarned=st.number_input('gold Earned')
     totalMinionsKilled=st.number_input('totalMinionsKilled')
     totalDamageDealt=st.number_input('totalDamageDealt')
     champLevel=st.number_input('champLevel')
     
with mid:
     st.markdown('#### Map Resources Control')      
     dragonKills=st.number_input('dragonKills')    
     damageDealtToBuildings=st.number_input('damageDealtToBuildings')
     turretsLost= st.number_input('turret lost')
     turretTakedowns=st.number_input('turret Takedowns')
     inhibitorsLost=st.number_input('inhibitors Lost')
     inhibitorTakedowns=st.number_input('inhibitor Takedowns')

data = {"turretsLost":turretsLost,
        "timePlayed":timePlayed,
        "goldEarned":goldEarned,
        "totalMinionsKilled":totalMinionsKilled,
        "inhibitorTakedowns":inhibitorTakedowns,
        "inhibitorsLost":inhibitorsLost,
        "turretTakedowns":inhibitorsLost,
        "kda":kda,
        "champLevel":champLevel,
        "totalDamageDealt":totalDamageDealt,
        "dragonKills":dragonKills,
        "damageDealtToBuildings":damageDealtToBuildings
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

