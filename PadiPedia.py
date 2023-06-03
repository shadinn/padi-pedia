from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.style import set_palette
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from streamlit_folium import st_folium
import pickle


# Load Data
df = pd.read_excel(r'dataclusters2.xlsx')
dt = pd.read_excel(r'dataclusters2.xlsx')

# Layout 
st.set_page_config(
    page_title = 'PadiPedia',
    page_icon = '✅',
    layout = 'wide'
)

#Sidebar
st.sidebar.title("PadiPedia")
menu = st.sidebar.radio("Navigation Menu",["Dashboard","Clustering","Prediction"])

if menu == "Dashboard":
    #Tab    
    tab1, tab2 = st.tabs(["Main","Data"])

    with tab1:
        # maindashboard title
        st.title("Dashboard Potensi Padi Nasional 2018 - 2022")

        # top-level filters 
        filter1,filter2 = st.columns(2)
        with filter1:
            provinsi_filter = st.selectbox("Provinsi", pd.unique(df['Provinsi']))
        with filter2:
            tahun_filter = st.selectbox("Tahun", pd.unique(dt['Tahun']))

        # creating a single-element container.
        placeholder = st.empty()

        # dataframe filter 
        #df = df[df['Provinsi']==provinsi_filter]
        #dt = dt[dt['Tahun']==tahun_filter]

        # KPIs 
        df = df[df['Provinsi']==provinsi_filter]
        produksi = np.sum(df['Produksi'])     
        luaspanen = np.sum(df['Luas Panen'])
        produktivitas = np.sum(df['Produktivitas'])

        with placeholder.container():
            # create three columns
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric(label="Produksi", value=round(produksi,2))
            kpi2.metric(label="Luas Panen", value=round(luaspanen,2))
            kpi3.metric(label="Produktivitas", value=round(produktivitas,2))

        # 2 Columns
        kolom_bar, kolom_maps = st.columns(2)
    
        with kolom_bar:
        
            # Bar Chart\
            df = df[df['Provinsi']==provinsi_filter]
            fig = px.bar(df, x="Tahun", y=["Produksi","Luas Panen","Produktivitas"], barmode="group", width= 525, height=400)
            st.plotly_chart(fig)
        
        with kolom_maps:
            # Map
            map = folium.Map(location=[-0.789275, 113.921327], zoom_start=4, scrollWheelZoom=False, tiles='CartoDB positron')
            
            choropleth = folium.Choropleth(
                geo_data= r'indonesia-prov.geojson',
                data=dt,
                columns=('Provinsi', 'clusters'),
                key_on='feature.properties.Propinsi',
                line_opacity=0.8,
                highlight=True
            )

            choropleth.geojson.add_to(map)
            dt = dt[dt['Tahun']==tahun_filter]
            dt_indexed = dt.set_index('Kode') 
            for feature in choropleth.geojson.data['features']:
                kodewil = feature['properties']['kode']
                feature['properties']['ID'] = 'Produksi : ' + str('{:,}'.format(dt_indexed.loc[kodewil,'Produksi'] )if kodewil in list(dt_indexed.index) else 'n/a')
                feature['properties']['kode'] = 'Luas Panen: ' + str('{:,}'.format(dt_indexed.loc[kodewil,'Luas Panen'] )if kodewil in list(dt_indexed.index) else 'n/a')
                feature['properties']['SUMBER'] = 'Produktivitas: ' + str('{:,}'.format(dt_indexed.loc[kodewil,'Produktivitas'] )if kodewil in list(dt_indexed.index) else 'n/a')

            choropleth.geojson.add_child(
                folium.features.GeoJsonTooltip(['Propinsi','ID','kode','SUMBER'], labels=False)
            )
            st_map = st_folium(map, width=500, height=320)


        # 3 kolom pie chart
        kolom_produksi, kolom_luaspanen, kolom_produktivitas = st.columns(3)
    
        with kolom_produksi:
            dt = dt[dt['Tahun']==tahun_filter]
            grouped_produksi = dt.groupby('Provinsi')['Produksi'].sum().reset_index()
            sorted_produksi = grouped_produksi.sort_values('Produksi', ascending=False)
            top5_produksi = sorted_produksi.head(5)
            piechart_produksi = px.pie(names=top5_produksi['Provinsi'], values=top5_produksi['Produksi'], hole=0.5)
            st.subheader('Top 5 Produksi')
            kolom_produksi.plotly_chart(piechart_produksi, use_container_width=True)
    
        with kolom_luaspanen:
            dt = dt[dt['Tahun']==tahun_filter]
            grouped_luaspanen = dt.groupby('Provinsi')['Luas Panen'].sum().reset_index()
            sorted_luaspanen = grouped_luaspanen.sort_values('Luas Panen', ascending=False)
            top5_luaspanen = sorted_luaspanen.head(5)
            piechart_luaspanen = px.pie(names=top5_luaspanen['Provinsi'], values=top5_luaspanen['Luas Panen'], hole=0.5)
            st.subheader('Top 5 Luas Panen')
            kolom_luaspanen.plotly_chart(piechart_luaspanen, use_container_width=True)
    
        with kolom_produktivitas:
            dt = dt[dt['Tahun']==tahun_filter]
            grouped_produktivitas = dt.groupby('Provinsi')['Produktivitas'].sum().reset_index()
            sorted_produktivitas = grouped_produktivitas.sort_values('Produktivitas', ascending=False)
            top5_produktivitas = sorted_produktivitas.head(5)
            piechart_produktivitas = px.pie(names=top5_produktivitas['Provinsi'], values=top5_produktivitas['Produktivitas'], hole=0.5)
            st.subheader('Top 5 Produktivitas')
            kolom_produktivitas.plotly_chart(piechart_produktivitas, use_container_width=True)




    with tab2 :
        # Tittle
        st.title("Data Padi 2018 - 2022")
        # Table
        dt = dt[dt['Tahun']==tahun_filter]
        datatable = dt[['Provinsi', 'Produksi', 'Luas Panen', 'Produktivitas']]
        datatable = dt.groupby(['Provinsi'])['Produksi', 'Luas Panen', 'Produktivitas'].agg('sum').reset_index()
        st.table(datatable)# will display the table

if menu == 'Clustering':
    dp = pd.read_excel(r'datapadi.xlsx')
    st.title('Klustering Potensi Padi Provinsi')
    
    # Pilihan metode
    tab1, tab2 = st.tabs(["K-Means","Analisis"])

    with tab1:

        #scaling dataset
        df_cluster = dp[['Produksi', 'Luas Panen', 'Produktivitas']]
        X = pd.DataFrame(StandardScaler().fit_transform(df_cluster))

        #PCA
        X = np.asarray(X)
        pca = PCA(n_components=3, random_state=24)
        X = pca.fit_transform(X)

        #Kmeans
        # --- Figures Settings ---
        color_palette=['#FFCC00', '#54318C']
        set_palette(color_palette)
        title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
        text_style=dict(fontweight='bold', fontfamily='serif')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Elbow Score ---
        elbow_score = KElbowVisualizer(KMeans(random_state=32, max_iter=500), k=(2, 10), ax=ax1)
        elbow_score.fit(X)
        elbow_score.finalize()
        elbow_score.ax.set_title('Distortion Score Elbow\n', **title)
        elbow_score.ax.tick_params(labelsize=7)
        for text in elbow_score.ax.legend_.texts:
            text.set_fontsize(9)
        for spine in elbow_score.ax.spines.values():
            spine.set_color('None')
        elbow_score.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderpad=2, frameon=False, fontsize=8)
        elbow_score.ax.grid(axis='y', alpha=0.5, color='#9B9A9C', linestyle='dotted')
        elbow_score.ax.grid(axis='x', alpha=0)
        elbow_score.ax.set_xlabel('\nK Values', fontsize=9, **text_style)
        elbow_score.ax.set_ylabel('Distortion Scores\n', fontsize=9, **text_style)

        # --- Elbow Score (Calinski-Harabasz Index) ---
        elbow_score_ch = KElbowVisualizer(KMeans(random_state=32, max_iter=500), k=(2, 10), metric='calinski_harabasz', timings=False, ax=ax2)
        elbow_score_ch.fit(X)
        elbow_score_ch.finalize()
        elbow_score_ch.ax.set_title('Calinski-Harabasz Score Elbow\n', **title)
        elbow_score_ch.ax.tick_params(labelsize=7)
        for text in elbow_score_ch.ax.legend_.texts:
            text.set_fontsize(9)
        for spine in elbow_score_ch.ax.spines.values():
            spine.set_color('None')
        elbow_score_ch.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderpad=2, frameon=False, fontsize=8)

        elbow_score_ch.ax.grid(axis='y', alpha=0.5, color='#9B9A9C', linestyle='dotted')
        elbow_score_ch.ax.grid(axis='x', alpha=0)
        elbow_score_ch.ax.set_xlabel('\nK Values', fontsize=9, **text_style)
        elbow_score_ch.ax.set_ylabel('Calinski-Harabasz Score\n', fontsize=9, **text_style)
        
        plt.tight_layout()
        plt.show()
        st.subheader('Perhitungan Nilai K')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())

        # --- Implementing K-Means ---
        kmeans = KMeans(n_clusters=3, random_state=32, max_iter=500)
        y_kmeans = kmeans.fit_predict(X)
    
        # --- Figures Settings ---
        cluster_colors=['#FFBB00', '#3C096C', '#9D4EDD']
        labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids']
        title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
        text_style=dict(fontweight='bold', fontfamily='serif')
        scatter_style=dict(linewidth=0.65, edgecolor='#100C07', alpha=0.85)
        legend_style=dict(borderpad=2, frameon=False, fontsize=8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Silhouette Plots ---
        s_viz = SilhouetteVisualizer(kmeans, ax=ax1, colors=cluster_colors)
        s_viz.fit(X)
        s_viz.finalize()
        s_viz.ax.set_title('Silhouette Plots of Clusters\n', **title)
        s_viz.ax.tick_params(labelsize=7)
        for text in s_viz.ax.legend_.texts:
            text.set_fontsize(9)
        for spine in s_viz.ax.spines.values():
            spine.set_color('None')
        s_viz.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), **legend_style)
        s_viz.ax.grid(axis='x', alpha=0.5, color='#9B9A9C', linestyle='dotted')
        s_viz.ax.grid(axis='y', alpha=0)
        s_viz.ax.set_xlabel('\nCoefficient Values', fontsize=9, **text_style)
        s_viz.ax.set_ylabel('Cluster Labels\n', fontsize=9, **text_style)

        # --- Clusters Distribution ---
        y_kmeans_labels = list(set(y_kmeans.tolist()))
        for i in y_kmeans_labels:
            ax2.scatter(X[y_kmeans==i, 0], X[y_kmeans == i, 1], s=50, c=cluster_colors[i], **scatter_style)
        ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=65, c='#0353A4', label='Centroids', **scatter_style)
        for spine in ax2.spines.values():
            spine.set_color('None')
        ax2.set_title('Scatter Plot Clusters Distributions\n', **title)
        ax2.legend(labels, bbox_to_anchor=(0.95, -0.05), ncol=5, **legend_style)
        ax2.grid(axis='both', alpha=0.5, color='#9B9A9C', linestyle='dotted')
        ax2.tick_params(left=False, right=False , labelleft=False , labelbottom=False, bottom=False)
        ax2.spines['bottom'].set_visible(True)
        ax2.spines['bottom'].set_color('#CAC9CD')

        # --- Suptitle & WM ---
        st.subheader('Hasil Clustering')
        plt.tight_layout()
        plt.show()
        st.pyplot(plt.show())

        # --- Create Accuracy Comparison Table ---
        def evaluate_clustering(X, y):
            db_index = round(davies_bouldin_score(X, y), 3)
            s_score = round(silhouette_score(X, y), 3)
            ch_index = round(calinski_harabasz_score(X, y), 3)
            return db_index, s_score, ch_index

        db_kmeans, ss_kmeans, ch_kmeans = evaluate_clustering(X, y_kmeans)

        compare = pd.DataFrame({'Model': ['K-Means'],
                                'Davies-Bouldin Index': [db_kmeans],
                                'Silhouette Score': [ss_kmeans],
                                'Calinski-Harabasz Index': [ch_kmeans]})
        st.subheader('Model Accuracy Evaluation')
        st.write(compare)

        #Cluster Profiling
        st.subheader('Cluster Profiling')
        # --- Add K-Means Prediction to Data Frame ----
        df_cluster['cluster_result'] = y_kmeans+1
        df_cluster['cluster_result'] = 'Cluster '+df_cluster['cluster_result'].astype(str)

        # --- Calculationg Overall Mean from Current Data Frame ---
        df_profile_overall = pd.DataFrame() 
        df_profile_overall['Overall'] = df_cluster.describe().loc[['mean']].T

        # --- Summarize Mean of Each Clusters --- 
        df_cluster_summary = df_cluster.groupby('cluster_result').describe().T.reset_index().rename(columns={'level_0': 'Column Name', 'level_1': 'Metrics'})
        df_cluster_summary = df_cluster_summary[df_cluster_summary['Metrics'] == 'mean'].set_index('Column Name')

        # --- Combining Both Data Frame ---
        df_profile = df_cluster_summary.join(df_profile_overall).reset_index()
        df_profile.style.background_gradient(cmap='YlOrBr').hide_index()
        st.write(df_profile)

        # Hasil Cluster
        st.subheader('Hasil Klustering')
        dp['clusters_results'] = y_kmeans
        dp = dp[['Provinsi','clusters_results']]
        clus1, clus2, clus3 = st.columns(3)
        with clus1:
            st.write('Cluster 1')
            dp1 = dp[dp['clusters_results']==0]
            dp1 = dp1['Provinsi'].drop_duplicates()
            st.write(dp1)

        with clus2:
            st.write('Cluster 2')
            dp2 = dp[dp['clusters_results']==1]
            dp2 = dp2['Provinsi'].drop_duplicates()
            st.write(dp2)

        with clus3:
            st.write('Cluster 3')
            dp3 = dp[dp['clusters_results']==2]
            dp3 = dp3['Provinsi'].drop_duplicates()
            st.write(dp3)
    with tab2:
        st.subheader('KLuster 1')
        st.subheader('Kluster 2')
        st.subheader('Kluster 3')

if menu == "Prediction":

    # Load Data
    with open('modeldt.pkl','rb') as file:
        model = pickle.load(file)

    st.title("Prediksi Potensi Padi Provinsi")
    input_provinsi = st.selectbox("Provinsi", pd.unique(df['Provinsi']))
    input_produksi = st.number_input("Produksi")
    input_luaspanen = st.number_input("Luas Panen")
    input_produktivitas = st.number_input("Produktivitas")
    X = np.array([[input_produksi, input_luaspanen, input_produktivitas]])
    X = X.astype(float)
    prediction = model.predict(X)[0]
    
    if st.button("Prediksi"):
        if prediction== 0:
            st.warning("Berhasil memprediksi di kluster 1")
        elif prediction== 1:
            st.warning("Berhasil memprediksi di kluster 2")
        elif prediction== 2:
            st.warning("Berhasil memprediksi di kluster 3")
