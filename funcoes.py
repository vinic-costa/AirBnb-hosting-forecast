import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer


def pre_processamento(X, y):
    variaveis_numericas = X.select_dtypes(include=["int64", "float64"]).columns
    variaveis_categoricas = X.select_dtypes(include=["object"]).columns

    transformacao_numericas = Pipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="median")),
            ("escala", RobustScaler()),
        ]
    )

    transformacao_categoricas = Pipeline(
        steps=[
            (
                "imputacao",
                SimpleImputer(strategy="most_frequent", fill_value="missing"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformacoes = ColumnTransformer(
        transformers=[
            ("numericas", transformacao_numericas, variaveis_numericas),
            ("categoricas", transformacao_categoricas, variaveis_categoricas),
        ]
    )

    return transformacoes


def executa_pipeline(modelo, X, y):

    transformacoes = pre_processamento(X, y)

    divisoes = KFold(n_splits=5, shuffle=True, random_state=42)

    pipeline_modelo = Pipeline(
        steps=[("transformacoes", transformacoes), ("regressor", modelo)]
    )

    metricas = cross_validate(
        pipeline_modelo,
        X,
        y,
        cv=divisoes,
        scoring=(
            "r2",
            "neg_mean_absolute_error",
        ),
        n_jobs=-1,  # usar todo o processamento da máquina
        verbose=0,
    )

    return pipeline_modelo, metricas


def organiza_resultados(resultados):

    for chave, valor in resultados.items():
        resultados[chave]["time_seconds"] = (
            resultados[chave]["fit_time"] + resultados[chave]["score_time"]
        )

    df_resultados = (
        pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "model"})
    )

    df_resultados_expandido = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    try:
        df_resultados_expandido = df_resultados_expandido.apply(pd.to_numeric)
    except ValueError:
        pass

    return df_resultados_expandido
