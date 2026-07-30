import json
import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.title("A5 Part B")

# LOAD DATA:
# loading data + getting lon/lat coords + getting rid of nan values
def load_data():
  gdf=gpd.read_file("dataset/business-licences.geojson")
  gdf["lon"]=gdf.geometry.x
  gdf["lat"]=gdf.geometry.y
  gdf=gdf.dropna(subset=["localarea"])
  return gdf
data = load_data()
# counting businesses in each local area
area_counts_all = data["localarea"].value_counts()

# SIDEBAR DETAILS:
# threshhold slider for minimum business count per area
st.sidebar.header("Controls ")
min_count = st.sidebar.slider(
  "minimum business count per area",
  min_value=0,
  max_value=int(area_counts_all.max()),
  value=500,
  step=50,
)
# slider for number of clusters
k = st.sidebar.slider("Num of clusters (K)", min_value=2, max_value=10, value=4)



# B1 — Choose your granularity and engineer an area-level feature vector
keep_areas=area_counts_all[area_counts_all>= min_count].index
filtered=data[data["localarea"].isin(keep_areas)]
# showing how many areas pass the filter 
st.sidebar.markdown(
  f"**{len(keep_areas)}/{len(area_counts_all)}** areas kept at this threshold"
)
# making sure there are less areas than amount of clusters (ex. cant make 5 clusters from 3 areas)
if len(keep_areas) < k:
  st.error(
  f"Only {len(keep_areas)} areas pass the minimum business count threshold, "
  f"which is fewer than K={k}, lower the threshold or lower K")
  st.stop()
# creating a table that shows the percentage of each type of business in each of the areas
composition = (
  pd.crosstab(filtered["localarea"], filtered["businesstype"], normalize="index")* 100)
# count businesses again and find avg lon/lat for each area to get centroids
area_counts=filtered["localarea"].value_counts().rename("business_count")
centroids =(
  filtered.dropna(subset=["lat", "lon"])
  .groupby("localarea")[["lat", "lon"]]
  .mean())
# join the composition, counts, and centroids into one dataframe for display
area_info = composition.join(area_counts).join(centroids)
# section title and captioning the composition matrix
st.subheader("area vs business type composition matrix") 
st.caption(
  f"{composition.shape[0]} areas vs {composition.shape[1]} business types "
  "each row sums to 100%")
st.dataframe(composition.round(1))



# B2 — Interactive K-means in Streamlit
X = composition.values
# create k means cluster model
kmeans=KMeans(n_clusters=k, random_state=42, n_init=10)
clusters=kmeans.fit_predict(X)
area_info=area_info.copy()
area_info["cluster"]=clusters.astype(str)
#  creating PCA model
pca=PCA(n_components=2, random_state=42)
coords=pca.fit_transform(X)
area_info["pc1"]=coords[:, 0]
area_info["pc2"]=coords[:, 1]
#  calculating the variance explained by each principal component
var_explained=pca.explained_variance_ratio_
#  showing summary and scatter plot of PCA result
st.subheader("PCA scatter of areas colored by cluster")
st.caption(
  f"PC1 explains {var_explained[0]*100:.1f}% and PC2 explains "
  f"{var_explained[1]*100:.1f}% of variance in the composition matrix."
)
fig_pca = px.scatter(
  area_info.reset_index(),
  x= "pc1",
  y="pc2",
  color="cluster",
  size="business_count",
  hover_name="localarea",
  text="localarea",
  labels={"pc1": "PC1", "pc2": "PC2"},
)
# adjusting labels and displaying
fig_pca.update_traces(textposition="top center")
st.plotly_chart(fig_pca, use_container_width=True)



# B3 — Geographic visualization
st.subheader("geographic display of clusters")
# dropping any rows without lon or lat + any with missing geographic info or missing vals
geo_df=area_info.dropna(subset=["lat", "lon"]).reset_index()
missing_geo = area_info[area_info[["lat", "lon"]].isna().any(axis=1)]
if not missing_geo.empty:
  st.caption(
    "no plottable centroid for: "
    + ", ".join(missing_geo.index.tolist()) )
fig_map=px.scatter_map(
  geo_df,
  lat="lat",
  lon="lon",
  color="cluster",
  size="business_count",
  hover_name="localarea",
  zoom=10,
  height=500,)
# gettin rid of map margins and changing style for better visiblity
fig_map.update_layout(map_style= "carto-positron", margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig_map, use_container_width=True)



# B4 — Cluster membership and interpretation
st.subheader("cluster membership")
#  looping through each cluster, sorting the areas by business count, and displaying the top 5 business types by average share in that cluster
for c in sorted(area_info["cluster"].unique(), key=int):
  members = area_info[area_info["cluster"]==c].sort_values(
    "business_count", ascending= False
  )
  top_types =composition.loc[members.index].mean().sort_values(ascending=False).head(5)
# displaying the cluster info in an expander
  with st.expander(
    f"Cluster {c}-{len(members)} areas: "
    + ", ".join(members.index.tolist())
):
    st.write("areas (sorted by business count):")
    st.dataframe(members[["business_count"]])
    st.write("top 5 business types by average share in this cluster:")
    st.dataframe(top_types.round(1).rename("avg % of licences"))
 
st.subheader("1) Part B Analysis")
st.markdown(
  """
  After testing out different K values, I found that K=6 had the most groupings without 
  separating the clusters into too many groups of 1. There are however, 
  still multiple groups of one here regardless.

  **The two big clusters** 
  Both of the big clusters (the 11 areas and 8 areas) are dominated by the long term rental, 
  healthcare, and some retail/contractor activity in the mix. The split between them isnt super clean.
  Cluster 1 leans more toward short term rentals, but they look pretty similar otherwise.
  One thing that does stand out is that Kerrisdale ends up being grouped with Mount Pleasant and West End.
  This is kinda surprising since Kerrisdale is a quiet, wealthier, west side neighborhool while the other 
  two are more denser and reter heavy urban neighborhoods.

  **Fairview and healthcare**
  An interesting detail I noticed was that Fairview dominates in the health care realm. 
  A quarter of its lisences are health care professionals and services which is way more than 
  anywhere else. This alot of sense, since VGH and the Broadway medical corridor are in Fairview 
  so there are naturally alot of clinics and private practices


  **Downtown and Strathacona as their own clusters** 
  This makes sense for downtown since its the commercial core of Vancouver. Its lisence is 
  heavy on legal services, health care, and long term rental. There are more short term rentals than 
  residential areas which is expected of the most commercial part of the city. <br>
  Strathacona showfood and non-food manufacturing in its top 5 which is very interesting. This 
  aligns with the history of the actual area, since its more older and industrial. Its near the rail 
  yards and old warehouse district

  **Other thoughts** 
  The clustering does a good job at isolating the handfuls of neighborhoods with a specific business mix
  . Most of the citys neighborhoods fall into one of two long term, rental heavy groups. 
  I'd say the main thing that was surprising to me was Kerisdale being grouped in with more denser and central 
  neighborhoods (since I would have expected ut to be grouped with quieter west-side ones ex. Dunbar, Shaughnessy)
  """
)