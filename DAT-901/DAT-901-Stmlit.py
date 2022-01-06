from re import A, L
from typing import List
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns

#Import model function
import model


st.set_page_config(
    page_title="KaDo",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)
st.set_option('deprecation.showPyplotGlobalUse', False)

#Const
filename = "C:/Users/Nicolas/Documents/EPITECH/DAT-901/KaDo_less.csv"
path_image = "C:/Users/Nicolas/Documents/EPITECH/DAT-901/cadeau.png"
months = ["January", "February", "March", "April", "May", "June", "Jully", "August", "September", "October", "November", "December"]
sidebar = "---"

#Init var
df = pd.DataFrame()

#Image
def inverse_png(path):
    image = Image.open(path)
    r,g,b,a = image.split()
    rgb_image = Image.merge('RGB', (r,g,b))
    inverted_image = ImageOps.invert(rgb_image)
    r2,g2,b2 = inverted_image.split()
    final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))
    return final_transparent_image

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

@st.cache(suppress_st_warning=True)
def reco_model(df):
    recommended_list = model.main(df)
    return recommended_list

st.sidebar.title("Navigation")
placeholder = st.sidebar.empty()
nav = placeholder.radio("", ("Home", "Contact"))

placeholder_home = st.empty()


#------------------------ PAGE HOME--------------------

if nav == "Home" and df.empty:
    with placeholder_home.container():
        st.title("Home")
        st.markdown(sidebar)
        st.subheader("Voulez-vous utiliser uploader un dataset ou utiliser celui par défaut ?")
        choice_df = st.radio("", ("Uploader", "Défaut"))
        if choice_df == "Uploader":
            file = st.file_uploader("Si vous voulez changer les données sur lesquels l'app se base", type=['csv'])
            if file:
                extension = file.name.split('.')[1]
                if extension.upper() == 'CSV':
                    df = pd.read_csv(file, encoding="utf8")
                    st.success("Dataset uploadé")
                    nav = placeholder.radio("", ("Introduction", "Données", "Recommendation", "What's next", "À propos"))
                    placeholder_home.empty()
                else:
                    st.error("Uploader un fichier csv s'il vous plait")
        else:
            df = pd.read_csv(filename, encoding="utf8")
            st.success("Dataset uploadé")
            nav = placeholder.radio("", ("Introduction", "Données", "Recommendation", "What's next", "À propos"))
            placeholder_home.empty()

st.sidebar.markdown(sidebar)

#------------------------ PAGE INTRODUCTION--------------------

if nav == "Introduction":
    #TITLE
    #st.title('Système de recommendation - KaDo')
    #st.markdown(sidebar)

    #INTRO
    st.title("Introduction")
    st.text('''
    L'objectif est d'explorer, d'analyser et d'avoir une comprehension profonde du dataset et des enjeux qu'il implique.
    Pour se faire nous utiliserons des outils variés pour transformer et trouver des indicateurs parlant.
    ''')
    home_col1, home_col2, home_col3 = st.columns([3,1,1])
    home_col1.text('''
    Les données sont composées de 8 features :

        - TICKET_ID     ID unique de la commande
        - MOIS_VENTE    mois de la vente
        - PRIX_NET      prix net de l'item
        - FAMILLE       famille de l'item
        - UNIVERS       univers de l'item
        - MAILLE        maille de l'item
        - LIBELLE       nom unique de l'item
        - CLI_ID        ID unique du client'''
    )

    #METRICS GENERAL
    home_col2.metric("TICKET_ID", f'{df["TICKET_ID"].nunique():,} tickets')
    home_col2.metric("CLI_ID", f'{df["CLI_ID"].nunique():,} clients')
    home_col2.metric("LIBELLE", f'{df["LIBELLE"].nunique():,} libellés')
    home_col3.metric("FAMILLE", "%i familles" % df["FAMILLE"].nunique())
    home_col3.metric("UNIVERS", "%i univers" % df["UNIVERS"].nunique())
    home_col3.metric("MAILLE", "%i mailles" % df["MAILLE"].nunique())

    #DATASET SAMPLE
    st.title("📖 Dataset")
    st.text("Extrait des 5 premières entrées")
    st.dataframe(df.head())

#------------------------ PAGE DONNEES--------------------
if nav == "Données":
    sub_nav = st.sidebar.radio("", ("Prix", "Famille", "Clients", "Commandes"))

    #Page Données>Prix
    if sub_nav == "Prix":
        st.title("Prix")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max", "%0.2f €" % max(df.PRIX_NET))
        col2.metric("Min", "%0.2f €" % min(df.PRIX_NET))
        col3.metric("Median", "%0.2f €" % df.PRIX_NET.median())
        col4.metric("Moyenne", "%0.2f €" % df.PRIX_NET.mean())
        st.markdown(sidebar)

        st.title("Répartition des prix par famille, maille ou univers")
        df_famille_prix = df[["FAMILLE", "PRIX_NET", "MAILLE", "UNIVERS"]].sort_values(by=['FAMILLE', 'PRIX_NET'])

        check_famille = st.checkbox("Famille")
        check_maille = st.checkbox("Maille")
        check_univers = st.checkbox("Univers")

        famille_selected = []
        maille_selected = []
        univers_selected = []
        data = []

        selected = None

        if check_famille:
            if selected is None:
                selected = "FAMILLE"
            famille_selected = st.multiselect("Selectionner les familles que vous voulez visualiser", df["FAMILLE"].unique())
            data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected)]
        if check_maille:
            selected = "MAILLE"
            if famille_selected:
                maille_selected = st.multiselect(
                    "Selectionner les mailles que vous voulez visualiser",
                    df["MAILLE"].loc[df["FAMILLE"].isin(famille_selected)].unique()
                )
                data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected) & df_famille_prix["MAILLE"].isin(maille_selected)]
            else:
                maille_selected = st.multiselect(
                    "Selectionner les mailles que vous voulez visualiser",
                    df["MAILLE"].unique()
                )
                data = df_famille_prix.loc[df_famille_prix["MAILLE"].isin(maille_selected)]
        if check_univers:
            selected = "UNIVERS"
            if famille_selected:
                if maille_selected:
                    univers_selected = st.multiselect(
                        "Selectionner les univers que vous voulez visualiser",
                        df["UNIVERS"].loc[df["FAMILLE"].isin(famille_selected) & df["MAILLE"].isin(maille_selected)].unique()
                    )
                    data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected) & df_famille_prix["MAILLE"].isin(maille_selected) & df_famille_prix["UNIVERS"].isin(univers_selected)]
                else:
                    univers_selected = st.multiselect(
                        "Selectionner les univers que vous voulez visualiser",
                        df["UNIVERS"].loc[df["FAMILLE"].isin(famille_selected)].unique()
                    )
                    data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected) & df_famille_prix["UNIVERS"].isin(univers_selected)]
            else:
                if maille_selected:
                    univers_selected = st.multiselect(
                        "Selectionner les univers que vous voulez visualiser",
                        df["UNIVERS"].loc[df["MAILLE"].isin(maille_selected)].unique()
                    )
                    data = df_famille_prix.loc[df_famille_prix["MAILLE"].isin(maille_selected) & df_famille_prix["UNIVERS"].isin(univers_selected)]
                else:
                    univers_selected = st.multiselect(
                        "Selectionner les univers que vous voulez visualiser",
                        df["UNIVERS"].unique()
                    )
                    data = df_famille_prix.loc[df_famille_prix["UNIVERS"].isin(univers_selected)]

        if famille_selected or maille_selected or univers_selected:
            fig_fam = plt.clf()
            ax = sns.displot(
                data = data,
                col = selected,
                x = "PRIX_NET",
                common_norm = False,
                kde=True, fill=True
            )
            plt.xlim(min(data.PRIX_NET), max(data.PRIX_NET))
            ax.set(xlabel='Prix (€)', ylabel='Quantité')
            st.pyplot(fig_fam)

    #Page Données>Famille
    if sub_nav == "Famille":
        st.title("Famille")
        st.subheader("Répartition des familles de produits vendus sur l'année")

        col1, col2, col3 =  st.columns(3)

        #Column 1
        df_familles = df["FAMILLE"].value_counts()
        fig1 = plt.figure(figsize=(10, 10))
        plt.pie(
            df_familles,
            labels = df_familles.index,
            autopct='%1.1f%%',
            explode=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        )
        plt.xticks(rotation=45, ha="right")
        #Display
        col1.pyplot(fig1)

        #Column 2 / 3
        col2.metric("Nombre de produits vendus sur l'année", f'{df.shape[0]:,}')
        for fam in range(len(df_familles)):
            if fam < len(df_familles)/2 - 1:
                col2.metric(df_familles.index[fam], f'{df_familles[fam]:,} produits')
            else:
                col3.metric(df_familles.index[fam], f'{df_familles[fam]:,} produits')

        #PLOT 3 CHOISIR MOIS
        input_month = st.selectbox(
            'Choisir un mois',
            (months)
        )

        fig2 = plt.figure(figsize = (15, 5))
        if(st.button('Confirmer mois')):
            month = months.index(input_month)
            df_month = df.loc[df["MOIS_VENTE"] == month + 1]["FAMILLE"].value_counts().sort_index()
            plt.bar(
                df_month.index,
                height = df_month.values
            )
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Famille")
            plt.ylabel("Nombre d'achats")
            st.subheader(months[month])
            st.pyplot(fig2)

        #PLOT 4 CHOISIR FAMILLE
        input_famille = st.selectbox(
            'Choisir une famille',
            (df["FAMILLE"].unique())
        )

        fig3 = plt.figure(figsize = (15, 5))
        if(st.button('Confirmer famille')):
            st.text(input_famille)
            df_famille =df.loc[df["FAMILLE"] == input_famille]["MOIS_VENTE"].value_counts().sort_index()
            df_famille.index = months
            plt.bar(
                df_famille.index,
                height = df_famille.values,
                width=0.99
            )
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Mois")
            plt.ylabel("Nombre d'achats")
            st.subheader(input_famille.title())
            st.pyplot(fig3)

    #Page Données>Clients
    if sub_nav == "Clients":
        client_id = st.text_input("Entrer identifiant client")
        if(st.button('Confirmer')):
            df_client = df.loc[df["CLI_ID"] == int(client_id.title())]
            st.dataframe(df_client)

    #Page Données>Commandes
    if sub_nav == "Commandes":
        st.title("Commandes")

        col1, col2 = st.columns(2)

        #COL 1 GRAPH
        df_command_month = df.groupby("MOIS_VENTE").agg({"TICKET_ID": "nunique"}).reset_index()
        df_command_month = df_command_month.drop(["MOIS_VENTE"], axis=1)
        df_command_month.index = months
        fig = plt.figure(figsize=(10, 4))
        plt.bar(
            df_command_month.index,
            height = df_command_month.TICKET_ID
        )
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Mois")
        plt.ylabel("Nombre d'achats")
        #Display
        col1.subheader("Répartition du nombre de commandes sur l'année")
        col1.pyplot(fig)

        #COL2 GRAPH
        df_mean_command = df.groupby(["CLI_ID", "TICKET_ID"], as_index=False)["PRIX_NET"].sum().groupby("CLI_ID")["PRIX_NET"].mean()
        df_num_command = df.groupby(["CLI_ID"])["TICKET_ID"].count()
        df_command = pd.concat([df_mean_command, df_num_command], axis = 1)
        df_command = df_command.rename(columns = {'PRIX_NET': 'MEAN', 'TICKET_ID': 'COUNT'})
        fig_com = plt.clf()
        ax = sns.regplot(x = df_command["MEAN"], y =  df_command["COUNT"], fit_reg = False)
        ax.set(
            xlabel = "Prix moyen par commande (en €)",
            ylabel = "Nombre de commandes par client"
        )
        col2.subheader("Répartition du nombre de commandes sur l'année")
        col2.pyplot(fig_com)


        command_id = st.text_input("Entrer identifiant commande")
        if(st.button('Confirmer')):
            df_command = df.loc[df["TICKET_ID"] == int(command_id.title())]
            st.dataframe(df_command)

        st.sidebar.title("Data selection")


#------------------------ PAGE RECOMMENDER MODEL--------------------
if nav == "Recommendation":
    st.title("Système de recommendation")
    st.subheader("Basé sur les comportements des autres clients")
    client_id = st.text_input("Entrer identifiant client")
    st.text("Vous pouvez choisir quels critère mettre en avant dans le choix de recommendation")
    data_choice = st.radio("Choississez", ("Aucun", "Prix", "Quantité"))
    if data_choice == "Prix":
        df_rec = df
        #TODO loc df values where cat prix du client
    elif data_choice == "Quantité":
        df_rec = df
        #TODO loc df values where cat qté du client
    else:
        df_rec = df
    recommended_list = reco_model(df_rec)
    if st.button(f'Obtenir la recommendation pour le client'):
        item_recommended = recommended_list.loc[recommended_list.index == int(client_id.title())]
        st.subheader(f"""Pour le client {client_id} l'item recommendé est :
            {item_recommended.values[0]}""")

#------------------------ PAGE WHAT'S NEXT--------------------
if nav == "What's next":
    st.title("What's next")
    st.write("NLP recognition")
    st.write("CI with AWS Bucket and ")
    st.write("Un dashboard plus complet et plus modulable")

#------------------------ PAGE CONTACT--------------------
if nav == "À propos":
    st.title("À propos")
    st.subheader("L'équipe")
    col1, col2, col3 = st.columns(3)

    image = Image.open("C:/Users/Nicolas/Documents/EPITECH/DAT-901/image.jpg")
    col1.image(image)
    col1.markdown("<h2 style='text-align: center'>BOUACEM Yannis</h1>", unsafe_allow_html=True)

    image1 = Image.open("C:/Users/Nicolas/Documents/EPITECH/DAT-901/image1.jpg")
    col2.image(image1)
    col2.markdown("<h2 style='text-align: center'>AMMAR Sana</h1>", unsafe_allow_html=True)

    image2 = Image.open("C:/Users/Nicolas/Documents/EPITECH/DAT-901/image2.jpg")
    col3.image(image2)
    col3.markdown("<h2 style='text-align: center'>PICHARD Nicolas</h1>", unsafe_allow_html=True)

    st.markdown(sidebar)

    st.subheader("Les librairies")
    col_lib1, col_lib2, col_lib3 = st.columns(3)
    libs = [
        "streamlit",
        "seaborn",
        "pandas",
        "numpy",
        "keras",
        "tensorflow",
        "scikit",
        "matplotlib"
    ]
    path = "C:/Users/Nicolas/Documents/EPITECH/DAT-901/"
    for lib in range(len(libs)):
        if lib < len(libs)*2/3:
            if lib < len(libs)/3:
                image = Image.open(path + libs[lib] + ".png")
                image.thumbnail((150, 250),Image.ANTIALIAS)
                col_lib1.image(image)
            else:
                image = Image.open(path + libs[lib] + ".png")
                image.thumbnail((150, 250),Image.ANTIALIAS)
                col_lib2.image(image)
        else:
            image = Image.open(path + libs[lib] + ".png")
            image.thumbnail((150, 250),Image.ANTIALIAS)
            col_lib3.image(image)


st.sidebar.image(inverse_png(path_image))


