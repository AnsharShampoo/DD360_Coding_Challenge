import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor



with open('rbmodel.pkl', 'rb') as li:
    rb_reg = pickle.load(li)

def main():
    st.title("DD360 Challenge: Predictor de precio por metro cuadrado")
    st.subheader("Por favor carga un DataSet (CSV):")
    raw = st.file_uploader(" ")
    st.text("Recuerda que tu DataSet debe contener las siguientes features mínimas:")
    st.markdown("- lat (latitúd)")
    st.markdown("- lon (longitúd)")
    st.markdown("- bathrooms (cantidad de baños)")
    st.markdown("- parking_lots (cantidad de lugares de estacionamiento)")
    st.markdown("- num_bedrooms (cantidad de habitaciones)")
    if st.button("Calcular"):
        dataset = pd.read_csv(raw)
        x_in = dataset[['lat','lon', 'bathrooms', 'parking_lots', 'num_bedrooms']]
        ans = rb_reg.predict(x_in)
        ans = pd.DataFrame(ans)
        st.write(ans, sorted=False)
        archive = pd.DataFrame.to_csv(ans)
        st.download_button("Descarga los resultados en un archivo CSV",archive,'resultados.csv', 'text/csv')

if __name__ == '__main__':
    main()