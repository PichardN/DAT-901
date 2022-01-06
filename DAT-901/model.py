import pandas as pd
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from typing import List
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Embedding,
    Flatten,
    Input,
    Multiply,
)
#preprocessing
from sklearn.preprocessing import OrdinalEncoder

import streamlit as st


# Define K as the number of item we want, and N_EPOCHS as the number of epochs we'll do
TOP_K = 5
N_EPOCHS = 1

# Create an array of every entry the pivot table has
def wide_to_long(wide, ratings):
    def _get_ratings(arr, rating):
        idx = np.where(arr == rating)
        return np.vstack(
            (idx[0], idx[1], np.ones(idx[0].size, dtype="int8") * rating)
        ).T
    long_arrays = []
    for r in ratings:
        long_arrays.append(_get_ratings(wide, r))
    return np.vstack(long_arrays)

# Define the neural collabirative filter
def create_ncf(
    nb_users: int,
    nb_items: int,
    latent_dim_mf: int = 4,
    latent_dim_mlp: int = 32,
    reg_mf: int = 0,
    reg_mlp: int = 0.01,
    dense_layers: List[int] = [8, 4],
    reg_layers: List[int] = [0.01, 0.01],
    activation_dense: str = "relu",
) -> keras.Model:

    # input layer
    user = Input(shape=(), dtype="int64", name="CLI_ID")
    item = Input(shape=(), dtype="int64", name="LIBELLE")

    # embedding layers
    mf_user_embedding = Embedding(
        input_dim=nb_users,
        output_dim=latent_dim_mf,
        name="mf_user_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mf),
        input_length=1,
    )
    mf_item_embedding = Embedding(
        input_dim=nb_items,
        output_dim=latent_dim_mf,
        name="mf_item_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mf),
        input_length=1,
    )

    mlp_user_embedding = Embedding(
        input_dim=nb_users,
        output_dim=latent_dim_mlp,
        name="mlp_user_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mlp),
        input_length=1,
    )
    mlp_item_embedding = Embedding(
        input_dim=nb_items,
        output_dim=latent_dim_mlp,
        name="mlp_item_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mlp),
        input_length=1,
    )

    # Matrix Factorisation vector
    mf_user_latent = Flatten()(mf_user_embedding(user))
    mf_item_latent = Flatten()(mf_item_embedding(item))
    mf_cat_latent = Multiply()([mf_user_latent, mf_item_latent])

    # Multi-Layer Perceptron vector
    mlp_user_latent = Flatten()(mlp_user_embedding(user))
    mlp_item_latent = Flatten()(mlp_item_embedding(item))
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

    mlp_vector = mlp_cat_latent

    # Build dense layers for model
    for i in range(len(dense_layers)):
        layer = Dense(
            dense_layers[i],
            activity_regularizer=l2(reg_layers[i]),
            activation=activation_dense,
            name="layer%d" % i,
        )
        mlp_vector = layer(mlp_vector)

    predict_layer = Concatenate()([mf_cat_latent, mlp_vector])

    result = Dense(
        1, activation="sigmoid", kernel_initializer="lecun_uniform", name="interaction"
    )

    output = result(predict_layer)

    model = Model(
        inputs=[user, item],
        outputs=[output],
    )

    return model

def make_tf_dataset(df, targets, val_split = 0.1):
    batch_size = 512
    seed = 0

    n_val = round(df.shape[0] * val_split)
    if seed:
        # shuffle all the rows
        x = df.sample(frac=1, random_state=seed).to_dict("series")
    else:
        x = df.to_dict("series")
    y = dict()
    for t in targets:
        y[t] = x.pop(t)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds_val = ds.take(n_val).batch(batch_size)
    ds_train = ds.skip(n_val).batch(batch_size)
    return ds_train, ds_val

#////////////////////MAIN////////////////////////

def clean_data(df_clean):
    #drop TICKET_ID
    features = ["MOIS_VENTE", "PRIX_NET", "FAMILLE", "UNIVERS", "MAILLE", "LIBELLE", "CLI_ID"]
    #create ordinal encoder to transform categoricals features
    ordinal_encoder = OrdinalEncoder()
    cat_cols = ["FAMILLE", "UNIVERS", "MAILLE"]
    df_clean[cat_cols] = ordinal_encoder.fit_transform(df_clean[cat_cols])
    return df_clean[features]

def main(df):
    df_train = clean_data(df)
    df_collab = df_train.drop(columns=['MOIS_VENTE', 'PRIX_NET','FAMILLE' ,'UNIVERS' ,'MAILLE'])
    qte_libelle = pd.DataFrame(df_collab.value_counts()).reset_index()
    qte_libelle.columns = ['LIBELLE', 'CLI_ID', 'COUNT']

    pivot_weight_df = qte_libelle.pivot_table(index = 'CLI_ID', columns='LIBELLE', values = "COUNT", fill_value = 0)
    weight_pivot = pivot_weight_df.to_numpy()
    long = wide_to_long(weight_pivot, np.unique(weight_pivot))
    df_weight = pd.DataFrame(long, columns=["CLI_ID", "LIBELLE", "interaction"])
    #We try to reduce the interaction weight
    df_weight_red = pd.DataFrame(df_weight, columns=["CLI_ID", "LIBELLE", "interaction"])
    df_weight_red["interaction"] = np.sqrt(df_weight["interaction"])

    n_users, n_items = weight_pivot.shape

    ncf_model = create_ncf(n_users, n_items)
    ncf_model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    ncf_model._name = "neural_collaborative_filtering"
    ds_train, ds_val = make_tf_dataset(df_weight_red, ["interaction"])
    train_hist = ncf_model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=N_EPOCHS,
    )

    long_test = wide_to_long(weight_pivot, np.unique(weight_pivot))
    df_test = pd.DataFrame(long_test, columns=["CLI_ID", "LIBELLE", "interaction"])
    ds_test, _ = make_tf_dataset(df_test, ["interaction"], val_split = 0)
    ncf_predictions = ncf_model.predict(ds_test)
    df_test["ncf_predictions"] = ncf_predictions

    df_test_clean = df_test.drop(columns = ["interaction"])
    table_test = pd.pivot_table(data = df_test_clean, index = "CLI_ID", columns = "LIBELLE", values = "ncf_predictions")
    table_test.columns = pivot_weight_df.columns
    table_test.index = pivot_weight_df.index
    recommended_items = table_test.idxmax(axis="columns")
    st.dataframe(recommended_items)
    return recommended_items
