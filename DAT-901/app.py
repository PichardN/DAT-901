from typing import List
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Import model function
import model

st.set_page_config(
    page_title="KaDo",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.set_option('deprecation.showPyplotGlobalUse', False)

#Const
FILEPATH = os.getcwd()
PATH_IMAGE = (f"{FILEPATH}/asset/image/")
FILENAME = "KaDo_less.csv"
MONTHS = ["January", "February", "March", "April", "May", "June", "Jully", "August", "September", "October", "November", "December"]
SIDEBAR = "---"

#Init var
df = pd.DataFrame()

@st.cache(suppress_st_warning=True)
def reco_model(df):
    recommended_list = model.main(df)
    return recommended_list

#------------------------ SIDEBAR NAV -----------------
st.sidebar.title("Navigation")
placeholder = st.sidebar.empty()
nav = placeholder.radio("", ("Home", "Contact"))
placeholder_home = st.empty()

#------------------------ PAGE HOME--------------------

if nav == "Home" and df.empty:
    with placeholder_home.container():
        st.title("Home")
        st.markdown(SIDEBAR)
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
            df = pd.read_csv(f"{FILEPATH}/data/{FILENAME}", encoding="utf8")
            st.success("Dataset uploadé")
            nav = placeholder.radio("", ("Introduction", "Données", "Recommendation", "À propos"))
            placeholder_home.empty()

st.sidebar.markdown(SIDEBAR)



#------------------------ PAGE INTRODUCTION--------------------

if nav == "Introduction":
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
    sub_nav = st.sidebar.radio("", ("Prix", "Famille", "Client", "Commande"))
    st.sidebar.markdown(SIDEBAR)
    #Page Données>Prix
    if sub_nav == "Prix":
        st.title("Prix")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max", "%0.2f €" % max(df.PRIX_NET))
        col2.metric("Min", "%0.2f €" % min(df.PRIX_NET))
        col3.metric("Median", "%0.2f €" % df.PRIX_NET.median())
        col4.metric("Moyenne", "%0.2f €" % df.PRIX_NET.mean())
        st.markdown(SIDEBAR)

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
            if famille_selected:
                data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected)]
        if check_maille:
            selected = "MAILLE"
            if famille_selected:
                maille_selected = st.multiselect(
                    "Selectionner les mailles que vous voulez visualiser",
                    df["MAILLE"].loc[df["FAMILLE"].isin(famille_selected)].unique()
                )
                if maille_selected:
                    data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected) & df_famille_prix["MAILLE"].isin(maille_selected)]
            else:
                maille_selected = st.multiselect(
                    "Selectionner les mailles que vous voulez visualiser",
                    df["MAILLE"].unique()
                )
                if maille_selected:
                    data = df_famille_prix.loc[df_famille_prix["MAILLE"].isin(maille_selected)]
        if check_univers:
            selected = "UNIVERS"
            if famille_selected:
                if maille_selected:
                    univers_selected = st.multiselect(
                        "Selectionner les univers que vous voulez visualiser",
                        df["UNIVERS"].loc[df["FAMILLE"].isin(famille_selected) & df["MAILLE"].isin(maille_selected)].unique()
                    )
                    if univers_selected:
                        data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected) & df_famille_prix["MAILLE"].isin(maille_selected) & df_famille_prix["UNIVERS"].isin(univers_selected)]
                else:
                    univers_selected = st.multiselect(
                        "Selectionner les univers que vous voulez visualiser",
                        df["UNIVERS"].loc[df["FAMILLE"].isin(famille_selected)].unique()
                    )
                    if univers_selected:
                        data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected) & df_famille_prix["UNIVERS"].isin(univers_selected)]
            else:
                if maille_selected:
                    univers_selected = st.multiselect(
                        "Selectionner les univers que vous voulez visualiser",
                        df["UNIVERS"].loc[df["MAILLE"].isin(maille_selected)].unique()
                    )
                    if maille_selected:
                        data = df_famille_prix.loc[df_famille_prix["MAILLE"].isin(maille_selected) & df_famille_prix["UNIVERS"].isin(univers_selected)]
                else:
                    univers_selected = st.multiselect(
                        "Selectionner les univers que vous voulez visualiser",
                        df["UNIVERS"].unique()
                    )
                    if univers_selected:
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
            (MONTHS)
        )

        fig2 = plt.figure(figsize = (15, 5))
        if(st.button('Confirmer mois')):
            month = MONTHS.index(input_month)
            df_month = df.loc[df["MOIS_VENTE"] == month + 1]["FAMILLE"].value_counts().sort_index()
            plt.bar(
                df_month.index,
                height = df_month.values
            )
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Famille")
            plt.ylabel("Nombre d'achats")
            st.subheader(MONTHS[month])
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
            df_famille.index = MONTHS
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
    if sub_nav == "Client":
        st.title("Client")
        client_id = st.text_input("Entrer identifiant client")
        if(st.button('Confirmer')):
            df_client = df.loc[df["CLI_ID"] == int(client_id.title())]
            st.dataframe(df_client)

    #Page Données>Commandes
    if sub_nav == "Commande":
        st.title("Commande")

        col1, col2 = st.columns(2)

        #COL 1 GRAPH
        df_command_month = df.groupby("MOIS_VENTE").agg({"TICKET_ID": "nunique"}).reset_index()
        df_command_month = df_command_month.drop(["MOIS_VENTE"], axis=1)
        df_command_month.index = MONTHS
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

#------------------------ PAGE RECOMMENDER MODEL--------------------
if nav == "Recommendation":
    st.title("Système de recommendation")
    st.subheader("Basé sur les comportements des autres clients")
    client_id = st.text_input("Entrer identifiant client")
    st.text("Vous pouvez choisir quels critère mettre en avant dans le choix de recommendation")
    col1, col2 = st.columns(2)
    data_choice = col1.radio("Choississez", ("Aucun", "Prix"))
    num = col2.number_input("Nombre de recommendations", step=1, min_value=1, max_value=10, format="%i")
    if st.button(f'Obtenir la recommendation pour le client') and client_id:
        df_rec = model.choose_data(data_choice, df, client_id)
        table = reco_model(df_rec)
        n_larg = pd.DataFrame({n: table.T[col].nlargest(num).index.tolist()
                        for n, col in enumerate(table.T)}).T
        n_larg.index = table.index
        client_rec = n_larg.loc[int(client_id)]
        affinity_item = table[client_rec].loc[table.index == int(client_id)]

        st.header(f"Pour le client {client_id} les items recommendés sont :")
        res_col1, res_col2 = st.columns(2)
        res_col1.subheader("Produit")
        res_col1.caption("")
        res_col2.subheader("Affinité")
        res_col2.caption("' The bigger the better '")
        for item in affinity_item:
            res_col1.text(item)
            res_col2.text("%.2f" % (affinity_item[item].values[0] * 100))

#------------------------ PAGE CONTACT--------------------
if nav == "À propos":
    st.title("À propos")
    st.subheader("L'équipe")
    col1, col2, col3 = st.columns(3)

    image = Image.open(f"{PATH_IMAGE}image.jpg")
    col1.image(image)
    col1.markdown("<h2 style='text-align: center'>BOUACEM Yannis</h1>", unsafe_allow_html=True)

    image1 = Image.open(f"{PATH_IMAGE}image1.jpg")
    col2.image(image1)
    col2.markdown("<h2 style='text-align: center'>AMMAR Sana</h1>", unsafe_allow_html=True)

    image2 = Image.open(f"{PATH_IMAGE}image2.jpg")

    col3.image(image2)
    col3.markdown("<h2 style='text-align: center'>PICHARD Nicolas</h1>", unsafe_allow_html=True)

    st.markdown(SIDEBAR)

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
    for lib in range(len(libs)):
        if lib < len(libs)*2/3:
            if lib < len(libs)/3:
                image = Image.open(PATH_IMAGE + libs[lib] + ".png")
                image.thumbnail((150, 250),Image.ANTIALIAS)
                col_lib1.image(image)
            else:
                image = Image.open(PATH_IMAGE + libs[lib] + ".png")
                image.thumbnail((150, 250),Image.ANTIALIAS)
                col_lib2.image(image)
        else:
            image = Image.open(PATH_IMAGE + libs[lib] + ".png")
            image.thumbnail((150, 250),Image.ANTIALIAS)
            col_lib3.image(image)

st.sidebar.image(Image.open(f"{PATH_IMAGE}gift.png"))