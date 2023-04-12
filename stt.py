import streamlit as st
import pandas as pd
import pickle
import shap
from streamlit_shap import st_shap
from PIL import Image


st.set_page_config(page_title="LoL Game Strategy Estimator",layout="centered") 



image = Image.open('leageu.jpg')
st.image(image)
st.title('🎮 Game Strategy Estimator')
st.info("Trained the model(97% accuracy) with 50K games, predict your game strategy and win the game! ")
left,mid=st.columns(2)
with left:
     st.success("Player Performance")
     
     kda=st.number_input('KDA',max_value=12,step=1)
     timePlayed=st.number_input('time Played(Minutes)',max_value=60,step=1)
     goldEarned=st.number_input('gold Earned(Minutes)',step=1)
     totalMinionsKilled=st.number_input('totalMinionsKilled(Minutes)',step=1)
     totalDamageDealt=st.number_input('totalDamageDealt(Minutes)',step=1)
     champExperience=st.number_input('champExperience(Minutes)',step=1)
     
with mid:
     st.success('Map Resources Control')      
     dragonKills=st.number_input('dragonKills',max_value=4,step=1)    
     damageDealtToBuildings=st.number_input('damageDealtToBuildings',step=1)
     turretsLost= st.number_input('turret lost',max_value=8,step=1)
     turretTakedowns=st.number_input('turret Takedowns',max_value=8,step=1)
     inhibitorsLost=st.number_input('inhibitors Lost',max_value=3,step=1)
     inhibitorTakedowns=st.number_input('inhibitor Takedowns',max_value=3,step=1)

data = {"turretsLost":turretsLost,
        "timePlayed":timePlayed,
        "goldEarned":goldEarned,
        "totalMinionsKilled":totalMinionsKilled,
        "inhibitorTakedowns":inhibitorTakedowns,
        "inhibitorsLost":inhibitorsLost,
        "turretTakedowns":inhibitorsLost,
        "kda":kda,
        "champExperience":champExperience,
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

