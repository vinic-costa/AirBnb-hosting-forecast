from joblib import load
import pandas as pd
import pydeck as pdk
import streamlit as st


BASE_DADOS_HOSPEDAGEM = "dados/dados_filtrados_2023_12.parquet"
BASE_BAIRROS = "dados/bairros_e_regioes.csv"
MODELO = "modelos/modelo_apresentacao.pkl"


@st.cache_data
def carregar_bairros():
    return pd.read_csv(BASE_BAIRROS)


@st.cache_data
def carregar_dados():
    return pd.read_parquet(BASE_DADOS_HOSPEDAGEM)


@st.cache_resource
def carregar_modelo():
    return load(MODELO)


dados_bairros = carregar_bairros()
dados_hospedagem = carregar_dados()
modelo = carregar_modelo()

bairros_listados = list(
    dados_hospedagem["neighbourhood_cleansed"].value_counts().nlargest(20).index
)

TIPOS_HOSPEDAGEM = {
    "Casa/apto inteiro": "Entire home/apt",
    "Quarto privado": "Private room",
    "Quarto de hotel": "Hotel room",
    "Quarto compartilhado": "Shared room",
}

COMODIDADES = [
    "Ar condicionado",
    "Café da manhã",
    "Máquina de lavar louça",
    "Secadora",
    "Elevador",
    "Freezer",
    "Academia",
    "Secador de cabelo",
    "Aquecedor",
    "Água quente",
    "Ferro de passar",
    "Cozinha",
    "Microondas",
    "Piscina",
    "TV",
    "WiFi",
]


widget_id = (id for id in range(1, 100))

aba1, aba2 = st.tabs(["Explore", "Simule"])

with aba1:
    min_preco, max_preco = st.slider(
        "Faixa de preço", 0, 2000, (0, 2000), key=next(widget_id)
    )
    selecionar_bairro_vis = st.selectbox(
        "Bairro", sorted(bairros_listados), 0, key=next(widget_id)
    )
    selecionar_tipo_hospedagem_vis = st.selectbox(
        "Tipo de quarto", sorted(TIPOS_HOSPEDAGEM.keys()), key=next(widget_id)
    )

    tipo_hospedagem_vis = TIPOS_HOSPEDAGEM[selecionar_tipo_hospedagem_vis]

    dados_filtrados = dados_hospedagem.query(
        " ".join(
            (
                "price >= @min_preco",
                "and price <= @max_preco",
                "and neighbourhood_cleansed == @selecionar_bairro_vis",
                "and room_type == @tipo_hospedagem_vis",
            )
        )
    )

    centralizar_mapa = dados_bairros.query("neighbourhood == @selecionar_bairro_vis")[
        ["latitude", "longitude"]
    ].values[0]

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=centralizar_mapa[0],
                longitude=centralizar_mapa[1],
                zoom=11,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=dados_filtrados[["longitude", "latitude"]],
                    get_position=["longitude", "latitude"],
                    get_color=[255, 140, 0, 140],
                    get_radius=100,
                    elevation_scale=10,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                    auto_highlight=True,
                    coverage=1,
                ),
            ],
        )
    )

with aba2:
    selecionar_bairro_previsao = st.selectbox(
        "Bairro", sorted(bairros_listados), 0, key=next(widget_id)
    )

    selecionar_tipo_hospedagem_previsao = st.selectbox(
        "Tipo de quarto", sorted(TIPOS_HOSPEDAGEM.keys()), key=next(widget_id)
    )

    tipo_hospedagem_previsao = TIPOS_HOSPEDAGEM[selecionar_tipo_hospedagem_previsao]

    selecionar_quantidade_pessoas_previsao = st.slider(
        "Número de pessoas", 1, 6, 1, key=next(widget_id)
    )

    selecionar_min_noites_previsao = st.slider("Noites", 1, 30, 1, key=next(widget_id))

    selecionar_comodidades_previsao = st.multiselect(
        "Comodidades", sorted(COMODIDADES), key=next(widget_id)
    )

    latitude = dados_bairros.query("neighbourhood == @selecionar_bairro_previsao")[
        "latitude"
    ].values

    longitude = dados_bairros.query("neighbourhood == @selecionar_bairro_previsao")[
        "longitude"
    ].values

    entrada_modelo = {
        "latitude": latitude,
        "longitude": longitude,
        "room_type": tipo_hospedagem_previsao,
        "accommodates": selecionar_quantidade_pessoas_previsao,
        "minimum_nights": selecionar_min_noites_previsao,
        "n_amenities": len(selecionar_comodidades_previsao),
        "neighbourhood_group": dados_bairros.query(
            "neighbourhood == @selecionar_bairro_previsao"
        )["neighbourhood_group"].values,
    }

    dados_entrada_modelo = pd.DataFrame(entrada_modelo)

    preco_previsao = st.button("Previsão de preço", key=next(widget_id))

    if preco_previsao:

        previsto = modelo.predict(dados_entrada_modelo.iloc[:, :])

        st.title(f"R$ {previsto[0]:.2f} por noite")
