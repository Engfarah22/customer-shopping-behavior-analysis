import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="ShopSense",
    page_icon="✦",
    layout="wide"
)

# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f6f4ef;
}

.hero {
    padding: 35px 10px 20px 10px;
}

.eyebrow {
    font-size: 13px;
    letter-spacing: 3px;
    color: #777;
    font-weight: bold;
}

.hero-title {
    font-size: 55px;
    font-weight: 800;
    color: #171717;
    margin: 5px 0;
}

.hero-text {
    font-size: 18px;
    color: #555;
}

.section-title {
    font-size: 28px;
    font-weight: 750;
    color: #171717;
    margin-top: 25px;
}

.metric-box {
    background-color: white;
    border: 1px solid #e5e1d8;
    border-radius: 18px;
    padding: 20px;
}

.metric-label {
    color: #777;
    font-size: 13px;
    text-transform: uppercase;
}

.metric-value {
    color: #171717;
    font-size: 30px;
    font-weight: 800;
}

.insight {
    background-color: #171717;
    color: white;
    padding: 20px;
    border-radius: 18px;
}

.insight-title {
    font-size: 12px;
    text-transform: uppercase;
    opacity: 0.6;
}

.insight-main {
    font-size: 19px;
    font-weight: bold;
    margin-top: 6px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    try:
        df = pd.read_csv("shopping_behavior_updated (1).csv")
    except:
        df = pd.read_csv("shopping_behavior_updated.csv")

    return df


df = load_data()

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["Gender_Binary"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

cluster_features = [
    "Age",
    "Gender_Binary",
    "Purchase Amount (USD)",
    "Review Rating",
    "Previous Purchases"
]

# ============================================================
# K-MEANS
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[cluster_features]
)

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Customer Segment"] = kmeans.fit_predict(
    X_scaled
)

# ============================================================
# SEGMENT PROFILE
# ============================================================

segment_profile = (
    df.groupby("Customer Segment")[cluster_features]
    .mean()
)

segment_size = (
    df["Customer Segment"]
    .value_counts()
    .sort_index()
)

segment_profile["Customer Count"] = segment_size

segment_profile["Customer %"] = (
    segment_size / len(df) * 100
)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="eyebrow">
CUSTOMER INTELLIGENCE PLATFORM
</div>

<div class="hero-title">
ShopSense
</div>

<div class="hero-text">
Understand your customers through shopping behavior,
customer segmentation, and machine learning.
</div>

</div>
""", unsafe_allow_html=True)

# ============================================================
# NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Customer Map",
    "Behavior Studio",
    "Segment Predictor"
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">Customer Landscape</div>',
        unsafe_allow_html=True
    )

    total_customers = len(df)

    avg_purchase = df[
        "Purchase Amount (USD)"
    ].mean()

    avg_rating = df[
        "Review Rating"
    ].mean()

    avg_previous = df[
        "Previous Purchases"
    ].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(f"""
        <div class="metric-box">

        <div class="metric-label">
        Customers
        </div>

        <div class="metric-value">
        {total_customers:,}
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-box">

        <div class="metric-label">
        Average Purchase
        </div>

        <div class="metric-value">
        ${avg_purchase:.2f}
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-box">

        <div class="metric-label">
        Average Rating
        </div>

        <div class="metric-value">
        {avg_rating:.2f}
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown(f"""
        <div class="metric-box">

        <div class="metric-label">
        Previous Purchases
        </div>

        <div class="metric-value">
        {avg_previous:.1f}
        </div>

        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">What stands out?</div>',
        unsafe_allow_html=True
    )

    largest_segment = int(
        segment_size.idxmax()
    )

    loyal_segment = int(
        segment_profile[
            "Previous Purchases"
        ].idxmax()
    )

    rating_segment = int(
        segment_profile[
            "Review Rating"
        ].idxmax()
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(f"""
        <div class="insight">

        <div class="insight-title">
        Largest Segment
        </div>

        <div class="insight-main">
        Segment {largest_segment}
        · {segment_size[largest_segment]:,} customers
        </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="insight">

        <div class="insight-title">
        Most Previous Purchases
        </div>

        <div class="insight-main">
        Segment {loyal_segment}
        · {segment_profile.loc[loyal_segment, "Previous Purchases"]:.1f}
        </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class="insight">

        <div class="insight-title">
        Highest Rating
        </div>

        <div class="insight-main">
        Segment {rating_segment}
        · {segment_profile.loc[rating_segment, "Review Rating"]:.2f}
        </div>

        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Segment Distribution</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    sns.barplot(
        x=segment_size.index,
        y=segment_size.values,
        ax=ax
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers per Segment")

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

# ============================================================
# TAB 2 — CUSTOMER MAP
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">Customer Map</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore how customer segments differ "
        "in age, spending, and purchase history."
    )

    selected_segments = st.multiselect(
        "Select segments",
        sorted(
            df["Customer Segment"].unique()
        ),
        default=sorted(
            df["Customer Segment"].unique()
        )
    )

    filtered_df = df[
        df["Customer Segment"].isin(
            selected_segments
        )
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    sns.scatterplot(
        data=filtered_df,
        x="Age",
        y="Purchase Amount (USD)",
        hue="Customer Segment",
        size="Previous Purchases",
        sizes=(30, 250),
        alpha=0.7,
        ax=ax
    )

    ax.set_title(
        "Age vs Purchase Amount"
    )

    ax.set_xlabel("Age")
    ax.set_ylabel(
        "Purchase Amount (USD)"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.markdown(
        '<div class="section-title">Segment Profiles</div>',
        unsafe_allow_html=True
    )

    profile_table = segment_profile.copy()

    profile_table["Gender"] = (
        profile_table["Gender_Binary"]
        .map({
            1: "Male",
            0: "Female"
        })
    )

    profile_table = profile_table[
        [
            "Gender",
            "Age",
            "Purchase Amount (USD)",
            "Review Rating",
            "Previous Purchases",
            "Customer Count",
            "Customer %"
        ]
    ]

    st.dataframe(
        profile_table.round(2),
        use_container_width=True
    )

# ============================================================
# TAB 3 — BEHAVIOR STUDIO
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">Behavior Studio</div>',
        unsafe_allow_html=True
    )

    chart = st.selectbox(
        "Choose an analysis",
        [
            "Age Distribution",
            "Purchase Amount Distribution",
            "Gender Distribution",
            "Product Categories",
            "Top Purchased Items",
            "Subscription Status",
            "Discount Applied",
            "Purchase Frequency"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    if chart == "Age Distribution":

        sns.histplot(
            df["Age"],
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Customers")

    elif chart == "Purchase Amount Distribution":

        sns.histplot(
            df["Purchase Amount (USD)"],
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_xlabel(
            "Purchase Amount (USD)"
        )

        ax.set_ylabel("Customers")

    elif chart == "Gender Distribution":

        sns.countplot(
            data=df,
            x="Gender",
            ax=ax
        )

        ax.set_ylabel("Customers")

    elif chart == "Product Categories":

        order = (
            df["Category"]
            .value_counts()
            .index
        )

        sns.countplot(
            data=df,
            y="Category",
            order=order,
            ax=ax
        )

        ax.set_xlabel("Customers")

    elif chart == "Top Purchased Items":

        top_items = (
            df["Item Purchased"]
            .value_counts()
            .head(15)
        )

        sns.barplot(
            x=top_items.values,
            y=top_items.index,
            ax=ax
        )

        ax.set_xlabel("Purchases")
        ax.set_ylabel("Item")

    elif chart == "Subscription Status":

        sns.countplot(
            data=df,
            x="Subscription Status",
            ax=ax
        )

        ax.set_ylabel("Customers")

    elif chart == "Discount Applied":

        sns.countplot(
            data=df,
            x="Discount Applied",
            ax=ax
        )

        ax.set_ylabel("Customers")

    elif chart == "Purchase Frequency":

        order = (
            df["Frequency of Purchases"]
            .value_counts()
            .index
        )

        sns.countplot(
            data=df,
            y="Frequency of Purchases",
            order=order,
            ax=ax
        )

        ax.set_xlabel("Customers")

    ax.set_title(chart)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

# ============================================================
# TAB 4 — SEGMENT PREDICTOR
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">Segment Predictor</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter a customer's information to predict "
        "which customer segment they belong to."
    )

    left, right = st.columns(2)

    with left:

        age = st.slider(
            "Age",
            18,
            70,
            30
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        purchase_amount = st.number_input(
            "Purchase Amount (USD)",
            min_value=1.0,
            max_value=100.0,
            value=60.0
        )

    with right:

        review_rating = st.slider(
            "Review Rating",
            1.0,
            5.0,
            4.0,
            0.1
        )

        previous_purchases = st.number_input(
            "Previous Purchases",
            min_value=0,
            max_value=100,
            value=20
        )

        predict = st.button(
            "Analyze Customer",
            use_container_width=True
        )

    if predict:

        gender_binary = (
            1 if gender == "Male"
            else 0
        )

        new_customer = pd.DataFrame({
            "Age": [age],
            "Gender_Binary": [gender_binary],
            "Purchase Amount (USD)": [
                purchase_amount
            ],
            "Review Rating": [
                review_rating
            ],
            "Previous Purchases": [
                previous_purchases
            ]
        })

        new_scaled = scaler.transform(
            new_customer[
                cluster_features
            ]
        )

        predicted_segment = int(
            kmeans.predict(
                new_scaled
            )[0]
        )

        profile = segment_profile.loc[
            predicted_segment
        ]

        st.success(
            f"Predicted Segment: {predicted_segment}"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Segment Customers",
                f"{int(profile['Customer Count']):,}"
            )

        with c2:

            st.metric(
                "Average Purchase",
                f"${profile['Purchase Amount (USD)']:.2f}"
            )

        with c3:

            st.metric(
                "Average Rating",
                f"{profile['Review Rating']:.2f}"
            )

        with c4:

            st.metric(
                "Previous Purchases",
                f"{profile['Previous Purchases']:.1f}"
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ShopSense | Customer Shopping Behaviour Analysis"
)