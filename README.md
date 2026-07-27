# IAT 461 — Assignment 3: Unsupervised Learning — Vancouver Business Licences Explorer

## Overview

In this assignment you will apply unsupervised learning — clustering, PCA, and an interactive Streamlit app — to a real, messy, large open dataset: the City of Vancouver's business licence records. There is no scaffold notebook handing you clean data or pre-built functions. You will acquire and clean the real dataset yourself, engineer features, run clustering and PCA, and build an interactive app to explore neighborhood-level results.

A separate **Streamlit Build Guide** will cover technical hints for the app (caching, plotting large datasets, widget layout, etc.). This document covers the data science tasks and questions.

### Learning goals

- Acquire, clean, and prepare a large real-world geospatial dataset for machine learning
- Engineer features from raw categorical, numeric, temporal, and spatial fields
- Apply K-means and DBSCAN clustering and compare how they behave on spatial data
- Apply PCA to project a higher-dimensional feature space into an interpretable 2D view
- Aggregate row-level data up to a coarser unit of analysis (business → neighborhood/zipcode)
- Build an interactive Streamlit app where clustering choices update live
- Interpret whether a statistical grouping is actually meaningful in the real world

---

## Data and tools

### Source

**City of Vancouver Open Data Portal — Business Licences**
https://opendata.vancouver.ca/explore/dataset/business-licences/

**Download the dataset in GeoJSON format** (not the CSV export) — part of the point of this assignment is getting comfortable reading and parsing GeoJSON directly, a format you'll encounter often with real-world spatial data. It contains ~200,000 business licence records with fields including business type/subtype, status, address, local area, postal code, number of employees, fee paid, issue/expiry dates, and geographic coordinates.

You are working with the real, full dataset (~130+ MB as GeoJSON) — expect missing values and fields that look useful but aren't. Deciding how to handle that is part of the assignment.

### Environment

- Python, Jupyter Notebook for Part A; Streamlit for Part B
- No scaffold notebook — structure your own notebook(s)/app

### Expected libraries

- `pandas`, `numpy`
- `geopandas` **or** plain `json`/`pandas` to read the GeoJSON — either is fine
- `matplotlib` or `plotly` — static visualizations in Part A
- `scikit-learn` — `KMeans`, `DBSCAN`, `PCA`, `StandardScaler`
- `streamlit`, `plotly` (`scatter_mapbox`/`scattergeo`) — Part B's interactive app

---

## Assignment structure

Two parts. Part A works at the level of individual business licences and is a notebook-based exploration. Part B aggregates up to neighborhood/zipcode level and is where you build your Streamlit app.

---

### Part A — Business-Level Clustering (notebook)

#### A1 — Data acquisition and cleaning

- Load the GeoJSON and inspect structure, types, and missingness yourself
- Decide how to handle records with no geographic coordinates, and missing `numberofemployees`/`feepaid` — justify your choice
- Filter to a sensible subset (e.g., `status == "Issued"`) and justify
- `businesstype` has ~94 categories, many sparse — decide how to consolidate (e.g., keep the top N, group the rest as "Other")
- **Optional refinement to try:** combine `businesstype` and `businesssubtype` into a single, more fine-grained industry label where a subtype exists (falling back to `businesstype` alone where it doesn't). This gives you a sharper industry signal for some categories — worth comparing against `businesstype` alone.
- Keep your sample size manageable for both plotting and later Streamlit responsiveness — the Build Guide covers this

> **Hints:**
> - Load the GeoJSON with `geopandas.read_file()` (gives you a ready-to-use `geometry` column), or with `json.load()` + `pd.json_normalize(data["features"])` if you'd rather stick with plain pandas — either is fine.
> - The fields you actually need live under each feature's `properties` — if `pd.json_normalize` doesn't flatten them automatically, pass `record_path`/prefix arguments, or normalize `[f["properties"] for f in data["features"]]` directly.
> - `geo_point_2d`/`geometry` is what gives you usable latitude/longitude — a business without it can't be placed on a map or used in A2's location clustering, so those rows are natural candidates to drop (but say so explicitly).
> - A fast way to decide what to do with a column: `df["column"].isna().mean()`. Run it on `numberofemployees`, `feepaid`, `businesssubtype`, and `postalcode` before deciding.
> - For consolidating `businesstype`, `df["businesstype"].value_counts()` shows you the long tail directly — a defensible approach is keeping however many categories cover ~80–90% of records and bucketing the rest as "Other."
> - `status` has values beyond `Issued` (pending, cancelled, gone out of business, etc.) — `df["status"].value_counts()` shows what you're including or excluding.
> - You don't need to hand-fix every messy field (inconsistent `province` values, duplicate `licencenumber` from revisions, etc.) — a reasonable, clearly justified pass on the columns your features actually use is enough.

> **Markdown required:** Document each cleaning decision and why you made it.

#### A2 — Location-only clustering: K-means vs. DBSCAN

Using **only latitude/longitude**, cluster the businesses two ways:

- **K-means:** use the elbow method to choose K, then fit
- **DBSCAN:** choose `eps`/`min_samples` reasonably (some trial and error expected)
- Produce a **static plot** (matplotlib) of the points colored by cluster assignment, one for each method

> **Markdown required:**
> 1. How do the K-means clusters map onto actual Vancouver geography — do they correspond to anything you'd recognize (downtown core, west side, etc.)?
> 2. How do the DBSCAN clusters compare? Where do the two methods agree or disagree, and why — think about what each algorithm assumes about cluster shape and density, and what DBSCAN's "noise" points represent.

#### A3 — Feature-based clustering: Size, Industry, Lifecycle

Engineer and scale features from:

- **Size** — number of employees, fee paid
- **Industry** — your consolidated business type (or type+subtype) categories, encoded
- **Lifecycle** — licence duration (`expireddate − issueddate`), month issued (cyclically encoded), licence revision number
- (Feel free to include other features you think are useful)

Run a simple K-means clustering on the combined, scaled feature set, then apply PCA to project to 2D and visualize the clusters.

> **Markdown required:** What do the resulting clusters seem to represent? Does this differ from the purely geographic clustering in A2?

> **Optional/suggested (not required):** Profile each cluster using z-scores of the input features (as shown in lecture) to name each cluster — a nice way to deepen your interpretation, but not graded as a separate requirement.

---

### Part B — Neighborhood/Zipcode Similarity Explorer (Streamlit app)

Shift the unit of analysis to *areas*: which neighborhoods/zipcodes look similar based on the mix of business types present? Build this as an interactive Streamlit app.

#### B1 — Choose your granularity and engineer an area-level feature vector

The key shift here: you're moving from **row = business** (Part A) to **row = area**. That means you need to engineer a new feature vector for each area, built up from the businesses inside it.

**Step 1 — pick your unit of analysis.** Choose **one**:
- **Postal FSA** (first 3 characters of `postalcode`) — more zipcode-like, ~54% coverage
- **`localarea`** (25 named neighborhoods) — cleaner, near-complete coverage

**Step 2 — engineer the features.** For each area, turn "what business types are present here" into a numeric feature vector: one feature (column) per business type, where the value is the **percentage of that area's businesses belonging to that type**. This is essentially a frequency-encoding of `businesstype`, aggregated up to the area level and row-normalized so an area with 50 businesses and one with 5,000 are still comparable on the same 0–100% scale.

**Step 3 — clean up thin areas.** Areas with very few businesses produce noisy percentages (one business can swing a whole feature by a large margin). Consider a minimum business-count threshold before including an area, and justify your cutoff.

> **Hints:**
> - `pd.crosstab(df["area_column"], df["businesstype"], normalize="index") * 100` does steps 1–2 in a single line — it directly gives you a row-normalized area × business-type matrix. `pd.pivot_table` with `aggfunc="count"` followed by dividing each row by its sum works too if you want more control.
> - The result of that crosstab **is** your feature matrix for clustering — each row is one "sample" (an area), each column is one "feature" (a business type's share of that area).
> - To apply the count threshold from Step 3, compute `df["area_column"].value_counts()` first, keep only areas above your cutoff, then build the crosstab from the filtered data (or build the crosstab first and drop rows where the row's original business count is too low — either order works).
> - You do not need to scale this feature matrix the way you scaled Size/Industry/Lifecycle in A3 — the values are already on a comparable 0–100% scale, but it's fine to standardize anyway if you prefer being consistent with your A3 pipeline.

#### B2 — Interactive K-means in Streamlit

In your app, add a slider/control for **K**, and re-run K-means on the composition matrix live as the user changes it. Show a PCA scatter of areas colored by cluster.

#### B3 — Geographic visualization

Plot each area as a point at its centroid lat/lon, colored by cluster, sized by business count. Should update live with the K selection.

#### B4 — Cluster membership and interpretation

Show which specific neighborhoods/zipcodes fall into each cluster (a list or on-map highlight), and interpret:

> **Markdown/writeup required:** Does the grouping seem meaningful given what you know (or can look up) about these areas? Any surprising groupings? Explain using the actual cluster membership, not just the general shape of the plot.

> **Optional/suggested:** cluster profiling by dominant business type per area-cluster, as in A3, deepens this but is not required.

#### B5 — Suggested reflection questions (optional, not required)

- Do business-similar areas also tend to be geographically close, or does similarity cut across geography?
- How does this compare to your Part A "Industry" clustering — consistent story, or different?
- Pick one area you didn't expect to see grouped with another — what do they have in common?

---

## Deliverables and submission

1. **Part A notebook** — cleaning, A2 and A3 clustering/visualizations, and required markdown responses
2. **Part B Streamlit app** (`app.py` + supporting files) — must run locally with the interactive K control working
3. **A short report (PDF or notebook export)** with the required write-ups from A1, A2, A3, and B4 (B5 optional)

You do not need to deploy the app; a local, runnable app plus your writeup is sufficient.

---

## Marking rubric (100 points)

| Task | What we're looking for | Points |
|---|---|---|
| **A1** | Justified cleaning decisions (coordinates, missing values, status filter, business type consolidation) | 15 |
| **A2** | Correct K-means (elbow-selected K) and DBSCAN implementation; static visualizations; thoughtful geographic/algorithmic comparison | 20 |
| **A3** | Correct feature engineering, scaling, K-means, and PCA; sound interpretation | 15 |
| **B1** | Correctly constructed, justified composition matrix | 10 |
| **B2** | Working interactive Streamlit K-means (live K control, live PCA scatter) | 15 |
| **B3** | Working, readable geographic visualization, updates live with K | 10 |
| **B4** | Clear cluster membership display and meaningful interpretation | 10 |
| **App quality** | App runs, is responsive, and updates correctly end-to-end | 5 |
| **Bonus** | Optional cluster profiling (A3 and/or B4) and/or B5 reflections | up to 5 |

---

## Academic integrity and AI use

Same policy as A1/A2:

- Individual assignment. Discuss general concepts with peers; all code and analysis must be your own.
- AI tools are permitted for specific steps (debugging, syntax help, understanding a library) with full disclosure of the tool and exact prompt used, marked with `#BEGIN`/`#END` comments as in A1.
- AI tools are **not** permitted to solve entire tasks, write your interpretation/writeup responses, or generate your cleaning-decision justifications for you.