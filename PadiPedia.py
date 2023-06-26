from sklearn.cluster import KMeans,  DBSCAN, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.style import set_palette
from sklearn.neighbors import NearestNeighbors
from streamlit_folium import st_folium
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from streamlit_folium import st_folium
import pickle
import scipy.cluster.hierarchy as shc
import plotly.graph_objects as go
import json


# Load Data
df = pd.read_excel(r'dataclusters2.xlsx')
dt = pd.read_excel(r'dataclusters2.xlsx')
geo_data= json.load(open("indonesia-prov.geojson"))

# --- Create Accuracy Comparison Table ---
def evaluate_clustering(X, y):
    db_index = round(davies_bouldin_score(X, y), 3)
    s_score = round(silhouette_score(X, y), 3)
    ch_index = round(calinski_harabasz_score(X, y), 3)
    return db_index, s_score, ch_index

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
        tahun_filter = st.selectbox("Tahun", pd.unique(dt['Tahun']))

        # creating a single-element container.
        placeholder = st.empty()

        # dataframe filter 
        dt = dt[dt['Tahun']==tahun_filter]
        
        # KPIs 
        produksi = np.sum(dt['Produksi'])     
        luaspanen = np.sum(dt['Luas Panen'])
        produktivitas = np.sum(dt['Produktivitas'])

        with placeholder.container():
            # create three columns
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric(label="Produksi", value=round(produksi,2))
            kpi2.metric(label="Luas Panen", value=round(luaspanen,2))
            kpi3.metric(label="Produktivitas", value=round(produktivitas,2))

        # 2 Columns
        kolom_bar, kolom_maps = st.columns(2)
    
        with kolom_bar:
        
            # Bar Chart
            fig = px.bar(df, x="Tahun", y=["Produksi","Luas Panen","Produktivitas"], barmode="group", width= 525, height=400)
            st.plotly_chart(fig)
        
        with kolom_maps:
            # Map
            map = folium.Map(location=[-0.789275, 113.921327], zoom_start=4, scrollWheelZoom=False, tiles='CartoDB positron')

            choropleth = folium.Choropleth(
                geo_data= geo_data,
                name = 'choropleth',
                data=dt,
                columns=['ID','clusters'],
                key_on= 'feature.properties.ID',
                fill_color='YlOrRd',
                line_opacity=0.8,
                highlight=True,
                legend_name= "Kluster"
            )
            choropleth.geojson.add_to(map)
            
            dt_indexed = dt.set_index('Kode') 
            for feature in choropleth.geojson.data['features']:
                kodewil = feature['properties']['kode']
                feature['properties']['ID2'] = 'Kluster : ' + str('{:,}'.format(dt_indexed.loc[kodewil,'clusters'] )if kodewil in list(dt_indexed.index) else 'n/a')
                feature['properties']['ID3'] = 'Produksi : ' + str('{:,}'.format(dt_indexed.loc[kodewil,'Produksi'] )if kodewil in list(dt_indexed.index) else 'n/a')
                feature['properties']['kode'] = 'Luas Panen: ' + str('{:,}'.format(dt_indexed.loc[kodewil,'Luas Panen'] )if kodewil in list(dt_indexed.index) else 'n/a')
                feature['properties']['SUMBER'] = 'Produktivitas: ' + str('{:,}'.format(dt_indexed.loc[kodewil,'Produktivitas'] )if kodewil in list(dt_indexed.index) else 'n/a')

            choropleth.geojson.add_child(
                folium.features.GeoJsonTooltip(['Propinsi','ID2','ID3','kode','SUMBER'], labels=False)
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


        st.subheader("Radar Chart Cluster")
        radar, hasil = st.columns(2)
        with radar :

            variables = ['Produksi', 'Luas Panen', 'Produktivitas']
            selected_data = dt[['clusters'] + variables]
            data = selected_data.groupby('clusters').mean()


            # Define the categories and variables
            categories = ['Produksi', 'Luas Panen', 'Produktivitas']
            data = data.reset_index()
            clusters = data['clusters'].tolist()
            
            # filters 
            kluster_filter = st.selectbox("Kluster", pd.unique(data['clusters']))

             # dataframe filter 
            data = data[data['clusters']==kluster_filter]
            
            # Create a trace for each cluster
            data_traces = []
            for cluster in clusters:
                values = data.loc[data['clusters'] == cluster, categories].values.flatten().tolist()
                values += values[:1]  # Repeat the first value to close the loop
                data_traces.append(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=cluster
                ))

            # Create the layout for the radar chart
            layout = go.Layout(
                polar=dict(
                    radialaxis=dict(visible=True)
                ),
                showlegend=False
            )

            # Create the figure with multiple traces
            fig = go.Figure(data=data_traces, layout=layout)

            # Display the radar chart
            st.plotly_chart(fig)
        st.markdown('<iframe src='https://flo.uri.sh/visualisation/14251593/embed' width="800px" height="400px" frameborder="0" scrolling="no"></iframe>', unsafe_allow_html=True)
        
      
    with tab2 :
        # Tittle
        st.title("Data Padi 2018 - 2022")
        # Table
        datatable = dt[['Provinsi', 'Produksi', 'Luas Panen', 'Produktivitas']]
        st.table(datatable)

if menu == 'Clustering':
    dp = pd.read_excel(r'datapadi.xlsx')
    st.title('Klustering Potensi Padi Provinsi')
    # dataframe filter 
    tahun_filter = st.selectbox("Tahun", pd.unique(dp['Tahun']))
    dp = dp[dp['Tahun']==tahun_filter]

    #scaling dataset
    df_cluster = dp[['Produksi', 'Luas Panen', 'Produktivitas']]
    X = np.log(df_cluster)
    X = np.asarray(X)

    # Pilihan metode
    tab1, tab2, tab3 = st.tabs(["K-Means","Hierarchical","DBSCAN"])

    with tab1:
        
        #Penentuan Nilai K
        elbow, calinski = st.columns(2)
        # --- Figures Settings ---
        color_palette=['#FFCC00', '#54318C']
        set_palette(color_palette)
        title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
        text_style=dict(fontweight='bold', fontfamily='serif')
        
        with elbow :
            elbowc = st.checkbox('Elbow Score')
            if elbowc :
                fig, ax = plt.subplots()
                elbow_score = KElbowVisualizer(KMeans(random_state=32, max_iter=500), k=(2, 10), ax=ax)
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

                plt.tight_layout()
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(plt.show())

        with calinski:
            calinski = st.checkbox('Calinski Score')
            if calinski :
                fig, ax = plt.subplots()
                elbow_score_ch = KElbowVisualizer(KMeans(random_state=32, max_iter=500), k=(2, 10), metric='calinski_harabasz', timings=False, ax=ax)
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
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(plt.show())


        # number kmeans
        numcol1,numcol2 = st.columns(2)
        with numcol1:
            number = st.slider('Tentukan Nilai K',2,5)

        # Model KMeans
        kmeans = KMeans(n_clusters=number, random_state=32, max_iter=500)
        y_kmeans = kmeans.fit_predict(X)

        silhouette, scatter = st.columns(2)
        # --- Figures Settings ---
        cluster_colors=['#FFBB00', '#3C096C', '#9D4EDD', '#FFE270','#100C07' ]
        labels = ['Cluster 1', 'Cluster 2', 'Cluster 3' ,'Clusters 4','Clusters 5', 'Centroids']
        title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
        text_style=dict(fontweight='bold', fontfamily='serif')
        scatter_style=dict(linewidth=0.65, edgecolor='#100C07', alpha=0.85)
        legend_style=dict(borderpad=2, frameon=False, fontsize=8)
        
        with silhouette :
            silhouette = st.checkbox('Silhouette Plot')
            if silhouette :
                fig, ax = plt.subplots()
                s_viz = SilhouetteVisualizer(kmeans, ax=ax, colors=cluster_colors)
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
                
                plt.tight_layout()
                plt.show()
                st.pyplot(plt.show())

        with scatter :
            scatter = st.checkbox('Scatter Plot')
            if scatter :
                fig, ax2 = plt.subplots()
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

                plt.tight_layout()
                plt.show()
                st.pyplot(plt.show())


        # Evaluasi Matrix
        db_kmeans, ss_kmeans, ch_kmeans = evaluate_clustering(X, y_kmeans)
       
        # create three columns
        st.subheader('Model Evaluation')
        km1, km2, km3 = st.columns(3)
        km1.metric(label="Davies-Bouldin Index", value=round(db_kmeans,2))
        km2.metric(label="Silhouette Score", value=round(ss_kmeans,2))
        km3.metric(label="Calinski-Harabasz Index", value=round(ch_kmeans,2))
        
        #Cluster Profiling
        st.subheader('Cluster Profiling')
        radarkmeans, profilkmeans = st.columns(2)
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
        st.write(df_profile)

        # Hasil Cluster
        st.subheader('Hasil Klustering')
        dp['clusters_results'] = y_kmeans
        dp = dp[['Provinsi','clusters_results']]
        clus1, clus2, clus3, clus4, clus5 = st.columns(5)
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
        with clus4:
            st.write('Cluster 4')
            dp4 = dp[dp['clusters_results']==3]
            dp4 = dp4['Provinsi'].drop_duplicates()
            st.write(dp4)
        with clus5:
            st.write('Cluster 5')
            dp5 = dp[dp['clusters_results']==4]
            dp5 = dp5['Provinsi'].drop_duplicates()
            st.write(dp5)

    with tab2:

        #Penentuan Nilai K
        ddgram, elbowh = st.columns(2)
        # --- Figures Settings ---
        color_palette=['#FFCC00', '#54318C']
        set_palette(color_palette)
        title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
        text_style=dict(fontweight='bold', fontfamily='serif')
        
        with ddgram :
            ddgram = st.checkbox('Dendogram')
            if ddgram :
                color_palette=['#472165', '#FFBB00', '#3C096C', '#9D4EDD', '#FFE270']
                set_palette(color_palette)
                text_style=dict(fontweight='bold', fontfamily='serif')
                ann=dict(textcoords='offset points', va='center', ha='center', fontfamily='serif', style='italic')
                title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
                bbox=dict(boxstyle='round', pad=0.3, color='#FFDA47', alpha=0.6)
                
                fig, ax1= plt.subplots()
                dend= shc.dendrogram(shc.linkage(X, method='ward', metric='euclidean'))
                plt.axhline(y=115, color='#3E3B39', linestyle='--')
                plt.xlabel('\nData Points', fontsize=9, **text_style)
                plt.ylabel('Euclidean Distances\n', fontsize=9, **text_style)
                plt.annotate('Horizontal Cut Line', xy=(15000, 130), xytext=(1, 1), fontsize=8, bbox=bbox, **ann)
                plt.tick_params(labelbottom=False)
                for spine in ax1.spines.values():
                    spine.set_color('None')
                plt.grid(axis='both', alpha=0)
                plt.tick_params(labelsize=7)
                plt.title('Dendrograms\n', **title)
                plt.tight_layout()
                st.pyplot(plt.show())
        with elbowh :
            elbowdd = st.checkbox('Elbow')
            if elbowdd :
                fig, ax2= plt.subplots()
                elbow_score_ch = KElbowVisualizer(AgglomerativeClustering(), metric='calinski_harabasz', timings=False, ax=ax2)
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
                st.pyplot(plt.show())
        # penentuan model

        # number kmeans
        numcol3,numcol4 = st.columns(2)
        with numcol3:
            numberdd = st.slider('Nilai K',2,5)

        # model
        agg_cluster = AgglomerativeClustering(n_clusters=numberdd, affinity='euclidean', linkage='ward')
        y_agg_cluster = agg_cluster.fit_predict(X)

        # plot scatter
        numcol5,numcol6 = st.columns(2)
        with numcol5:

            scatterh = st.checkbox('Scatter Hierichal')
            if scatterh :
                cluster_colors=['#FFBB00', '#3C096C', '#9D4EDD', '#FFE270','#100C07' ]
                labels = ['Cluster 1', 'Cluster 2', 'Cluster 3' ,'Clusters 4','Clusters 5', 'Centroids']
                scatter_style=dict(linewidth=0.65, edgecolor='#100C07', alpha=0.85)
                title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')

                fig, ax = plt.subplots()
                y_agg_labels = list(set(y_agg_cluster.tolist()))
                for i in y_agg_labels:
                    ax.scatter(X[y_agg_cluster==i, 0], X[y_agg_cluster == i, 1], s=50, c=cluster_colors[i], label=labels[i], **scatter_style)
                for spine in ax.spines.values():
                    spine.set_color('None')
                ax.legend(labels, bbox_to_anchor=(0.95, -0.05), ncol=5)
                plt.tight_layout()
                plt.show()
                st.pyplot(plt.show())

        # Evaluasi Matrix
        db_agg, ss_agg, ch_agg = evaluate_clustering(X, y_agg_cluster)
       
        # create three columns
        st.subheader('Model Evaluation')
        agg1, agg2, agg3 = st.columns(3)
        agg1.metric(label="Davies-Bouldin Index", value=round(db_agg,2))
        agg2.metric(label="Silhouette Score", value=round(ss_agg,2))
        agg3.metric(label="Calinski-Harabasz Index", value=round(ch_agg,2))
       
        #Cluster Profiling
        st.subheader('Cluster Profiling')
        # --- Add K-Means Prediction to Data Frame ----
        df_cluster['cluster_result'] = y_agg_cluster+1
        df_cluster['cluster_result'] = 'Cluster '+df_cluster['cluster_result'].astype(str)

        # --- Calculationg Overall Mean from Current Data Frame ---
        df_profile_overall = pd.DataFrame() 
        df_profile_overall['Overall'] = df_cluster.describe().loc[['mean']].T

        # --- Summarize Mean of Each Clusters --- 
        df_cluster_summary = df_cluster.groupby('cluster_result').describe().T.reset_index().rename(columns={'level_0': 'Column Name', 'level_1': 'Metrics'})
        df_cluster_summary = df_cluster_summary[df_cluster_summary['Metrics'] == 'mean'].set_index('Column Name')

        # --- Combining Both Data Frame ---
        df_profile = df_cluster_summary.join(df_profile_overall).reset_index()
        st.write(df_profile)

        # Hasil Cluster
        st.subheader('Hasil Klustering')
        dp['clusters_results'] = y_agg_cluster
        dp = dp[['Provinsi','clusters_results']]
        clus1, clus2, clus3, clus4, clus5 = st.columns(5)
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
        with clus4:
            st.write('Cluster 4')
            dp4 = dp[dp['clusters_results']==3]
            dp4 = dp4['Provinsi'].drop_duplicates()
            st.write(dp4)
        with clus5:
            st.write('Cluster 5')
            dp5 = dp[dp['clusters_results']==4]
            dp5 = dp5['Provinsi'].drop_duplicates()
            st.write(dp5)

    with tab3:
        # model
        dbscan = DBSCAN(eps=1.25, min_samples=2)
        y_dbscan = dbscan.fit_predict(X)

        #Penentuan Nilai epsilon
        epsilon = st.checkbox('Epsilon')
        if epsilon :
        # Menghitung jarak ke tetangga terdekat
            def compute_nearest_neighbors(data, k):
                neigh = NearestNeighbors(n_neighbors=k)
                neigh.fit(data)
                distances, indices = neigh.kneighbors(data)
                return distances[:, -1]  # Mengambil jarak ke tetangga terdekat

            # Mengurutkan jarak secara menaik
            def sort_distances(distances):
                return np.sort(distances)

            # Plot jarak ke tetangga terdekat
            def plot_distances(distances):
                plt.plot(distances)
                plt.xlabel('Data Index')
                plt.ylabel('Distance to Nearest Neighbor')
                plt.title('Nearest Neighbor Distance Plot')
                plt.show()

            k = 4

            distances = compute_nearest_neighbors(X, k)
            sorted_distances = sort_distances(distances)
            st.pyplot(plot_distances(sorted_distances))

        scatterdbscan = st.checkbox('Scatter Plot DBSCAN')
        if scatterdbscan:
            # Figure Setting
            cluster_colors=['#FFBB00', '#3C096C', '#9D4EDD']
            labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'outliers']
            scatter_style=dict(linewidth=0.65, edgecolor='#100C07', alpha=0.85)
            title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
            legend_style=dict(borderpad=2, frameon=False, fontsize=6)
            
            # Label Setting
            # --- Percentage labels ---
            unique, counts = np.unique(y_dbscan, return_counts=True)
            dbscan_count = dict(zip(unique, counts))
            total = sum(dbscan_count.values())
            dbscan_label = {key: round(value/total*100, 2) for key, value in dbscan_count.items() if key != -1}

            fig, ax = plt.subplots()
            y_dbscan_labels = list(set(y_dbscan.tolist()))
            for i in np.arange(0, 2, 1):
                ax.scatter(X[y_dbscan==i, 0], X[y_dbscan == i, 1], s=50, c=cluster_colors[i], label=labels[i], **scatter_style)
            ax.scatter(X[y_dbscan==-1, 0], X[y_dbscan == -1, 1], s=15, c=cluster_colors[2], label=labels[2], **scatter_style)
            for spine in ax.spines.values():
                spine.set_color('None')
            plt.legend([f"Cluster {i+1} - ({k}%)" for i, k in dbscan_label.items()], bbox_to_anchor=(0.75, -0.01), ncol=3, **legend_style)
            plt.tight_layout()
            plt.show()
            st.pyplot(plt.show())

        # Evaluasi Matrix
        db_dbscan, ss_dbscan, ch_dbscan = evaluate_clustering(X, y_dbscan)
       
        # create three columns
        st.subheader('Model Evaluation')
        dbs1, dbs2, dbs3 = st.columns(3)
        dbs1.metric(label="Davies-Bouldin Index", value=round(db_dbscan,2))
        dbs2.metric(label="Silhouette Score", value=round(ss_dbscan,2))
        dbs3.metric(label="Calinski-Harabasz Index", value=round(ch_dbscan,2))
       
        #Cluster Profiling
        st.subheader('Cluster Profiling')
        # --- Add K-Means Prediction to Data Frame ----
        df_cluster['cluster_result'] = y_dbscan+1
        df_cluster['cluster_result'] = 'Cluster '+df_cluster['cluster_result'].astype(str)

        # --- Calculationg Overall Mean from Current Data Frame ---
        df_profile_overall = pd.DataFrame() 
        df_profile_overall['Overall'] = df_cluster.describe().loc[['mean']].T

        # --- Summarize Mean of Each Clusters --- 
        df_cluster_summary = df_cluster.groupby('cluster_result').describe().T.reset_index().rename(columns={'level_0': 'Column Name', 'level_1': 'Metrics'})
        df_cluster_summary = df_cluster_summary[df_cluster_summary['Metrics'] == 'mean'].set_index('Column Name')

        # --- Combining Both Data Frame ---
        df_profile = df_cluster_summary.join(df_profile_overall).reset_index()
        st.write(df_profile)

        # Hasil Cluster
        st.subheader('Hasil Klustering')
        dp['clusters_results'] = y_dbscan
        dp = dp[['Provinsi','clusters_results']]
        clus1, clus2 = st.columns(2)
        with clus1:
            st.write('Cluster 1')
            dp1 = dp[dp['clusters_results']==0]
            dp1 = dp1['Provinsi'].drop_duplicates()
            st.write(dp1)

        with clus2:
            st.write('Outlier')
            dp2 = dp[dp['clusters_results']==-1]
            dp2 = dp2['Provinsi'].drop_duplicates()
            st.write(dp2)


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
