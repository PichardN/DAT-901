import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="KaDo",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)
#DataFrame
filename = "/Users/ammar/DAT-901/DAT-901/KaDo_less.csv"
path_image = "/Users/ammar//DAT-901/DAT-901/cadeau.png"
months = ["January", "February", "March", "April", "May", "June", "Jully", "August", "September", "October", "November", "December"]

df = pd.read_csv(filename, encoding="utf8")

#Image
def inverse_png(path):
    image = Image.open(path)
    r,g,b,a = image.split()
    rgb_image = Image.merge('RGB', (r,g,b))
    inverted_image = ImageOps.invert(rgb_image)
    r2,g2,b2 = inverted_image.split()
    final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))
    return final_transparent_image

#inverted_image = ImageOps.invert(image.convert('RGB'))

sidebar = "---"

st.sidebar.title("Navigation")
#La definition du barre de navigation Les 4 premiers 
nav = st.sidebar.radio("", ("Home", "Données", "Model de recommendation", "What's next"))
st.sidebar.markdown(sidebar)

if nav == "Home":
    #TITLE
    #////////////////// Pour Afficher st. .... //////////////////  
    #st.title('Système de recommendation - KaDo')
    #st.markdown(sidebar)

    #INTRO
    st.title("Introduction")
    st.text('''
    L'objectif est d'explorer, d'analyser et d'avoir une comprehension profonde du dataset et des enjeux qu'il implique.
    Pour se faire nous utiliserons des outils variés pour transformer et trouver des indicateurs parlant.
    ''')
    home_col1, home_col2, home_col3= st.columns([3,1,1])
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

if nav == "Données":
    sub_nav = st.sidebar.radio("", ("General", "Prix", "Famille", "Clients", "Commandes"))
    if sub_nav == "General":
        st.title("GENERAL")
    if sub_nav == "Prix":
        st.title("Prix")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max", "%0.2f €" % max(df.PRIX_NET))
        col2.metric("Min", "%0.2f €" % min(df.PRIX_NET))
        col3.metric("Median", "%0.2f €" % df.PRIX_NET.median())
        col4.metric("Moyenne", "%0.2f €" % df.PRIX_NET.mean())
        st.markdown(sidebar)

        st.title("Répartition des prix par filtre")
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
            famille_selected = st.multiselect("Select the FAMILLE you want to compare", df["FAMILLE"].unique())
            data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected)]
        if check_maille:
            selected = "MAILLE"
            if famille_selected:
                maille_selected = st.multiselect(
                    "Select the MAILLE you want to compare",
                    df["MAILLE"].loc[df["FAMILLE"].isin(famille_selected)].unique()
                )
                data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected) & df_famille_prix["MAILLE"].isin(maille_selected)]
            else:
                maille_selected = st.multiselect(
                    "Select the MAILLE you want to compare",
                    df["MAILLE"].unique()
                )
                data = df_famille_prix.loc[df_famille_prix["MAILLE"].isin(maille_selected)]
        if check_univers:
            selected = "UNIVERS"
            if famille_selected:
                if maille_selected:
                    univers_selected = st.multiselect(
                        "Select the UNIVERS you want to compare",
                        df["UNIVERS"].loc[df["FAMILLE"].isin(famille_selected) & df["MAILLE"].isin(maille_selected)].unique()
                    )
                    data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected) & df_famille_prix["MAILLE"].isin(maille_selected) & df_famille_prix["UNIVERS"].isin(univers_selected)]
                else:
                    univers_selected = st.multiselect(
                        "Select the UNIVERS you want to compare",
                        df["UNIVERS"].loc[df["FAMILLE"].isin(famille_selected)].unique()
                    )
                    data = df_famille_prix.loc[df_famille_prix["FAMILLE"].isin(famille_selected) & df_famille_prix["UNIVERS"].isin(univers_selected)]
            else:
                if maille_selected:
                    univers_selected = st.multiselect(
                        "Select the UNIVERS you want to compare",
                        df["UNIVERS"].loc[df["MAILLE"].isin(maille_selected)].unique()
                    )
                    data = df_famille_prix.loc[df_famille_prix["MAILLE"].isin(maille_selected) & df_famille_prix["UNIVERS"].isin(univers_selected)]
                else:
                    univers_selected = st.multiselect(
                        "Select the UNIVERS you want to compare",
                        df["UNIVERS"].unique()
                    )
                    data = df_famille_prix.loc[df_famille_prix["UNIVERS"].isin(univers_selected)]

        #selected = np.concatenate((famille_selected, maille_selected, univers_selected))

        #selectedaa = st.radio("Choisir catégorie", {"FAMILLE", "MAILLE", "UNIVERS"})

        #type_selected = st.multiselect("Select the FAMILLE you want to compare", df[selected].unique(),df[selected].unique()[0] )
        if famille_selected or maille_selected or univers_selected:
            fig_fam = sns.displot(
                data = data,
                col = selected,
                x = "PRIX_NET",
                common_norm = False,
                kde=False, fill=True
            )
            st.pyplot(fig_fam)


    if sub_nav == "Famille":
        st.title("Repartition")
        col1, col2 =  st.columns(2)

        #PLOT 1
        months = ["January", "February", "March", "April", "May", "June", "Jully", "August", "September", "October", "November", "December"]
        df_month = df["MOIS_VENTE"].value_counts().sort_index()
        df_month.index = months
        fig = plt.figure(figsize=(10, 7))
        plt.bar(
            df_month.index,
            height = df_month.values
        )
        plt.xticks(rotation=45, ha="left")
        plt.xlabel("Mois")
        plt.ylabel("Nombre d'achats")
        col1.subheader("Nombre d'achat par mois")
        col1.pyplot(fig)

        #PLOT 2
        df_familles = df["FAMILLE"].value_counts()
        #////////////////////////// AFfiher la DataFrame /////////////////////////
        st.dataframe(df_familles)
        fig1 = plt.figure(figsize=(10, 4))     # La taille (Le contneur) du graph 
          
        labels = df_familles.index
        sizes = df_familles.values
        explode = (0, 0.1, 0, 0, 0, 0.1, 0.1)
        fig1, ax1 = plt.subplots()
        pp=ax1.pie(sizes, explode=explode,  autopct='%1.1f%%',
        shadow=True, startangle=90)
        ax1.axis('equal')
        plt.legend(pp[0],labels, bbox_to_anchor=(1,0), loc="lower right", 
                          bbox_transform=plt.gcf().transFigure)

          #plt.bar(        # Creation du graph   ( remplacer par un pie plot)
          #  df_familles.index,      # Les valeurs a droite du graph (Hygiene , ... )
          #  height = df_familles.values      # Les Valeurs des graph a droite 
        plt.xticks(rotation=45, ha="right")
       # plt.xlabel("Famille")
       # plt.ylabel("Nombre d'achats")
        col2.subheader("Pourcentage d'achat par famille")
        #////////////////////////// Afficher le graph //////////////////////////
        col2.pyplot(fig1)

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

    if sub_nav == "Clients":
        client_id = st.text_input("Entrer identifiant client")
        if(st.button('Confirmer')):
            df_client = df.loc[df["CLI_ID"] == int(client_id.title())]
            st.dataframe(df_client)


    if sub_nav == "Commandes":
        st.title("Metrics")
        col1, col2, col3, col4 = st.columns(4)
        #TODO create groupby for command and calculte average price of command et repartition sur l'année
        col1.metric("Max", "%0.2f €" % max(df.PRIX_NET))
        col2.metric("Min", "%0.2f €" % min(df.PRIX_NET))
        col3.metric("Median", "%0.2f €" % df.PRIX_NET.median())
        col4.metric("Moyenne", "%0.2f €" % df.PRIX_NET.mean())
    st.sidebar.title("Data selection")

if nav == "Recommender model":
    st.title("MODEL")

if nav == "What's next":
    st.title("WHAT'S NEXT")


st.sidebar.image(inverse_png(path_image))
