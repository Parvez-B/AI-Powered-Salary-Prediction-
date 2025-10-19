# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import Pipeline

# # ---------------- Page Config ----------------
# st.set_page_config(page_title="💼 Salary Prediction App", layout="wide")

# # ---------------- Custom CSS for Styling ----------------
# st.markdown("""
#     <style>
#         /* Background gradient */
#         .main {
#             background: linear-gradient(135deg, #667eea, #764ba2);
#             color: white;
#         }

#         /* Title Styling */
#         .big-title {
#             font-size: 42px;
#             text-align: center;
#             font-weight: bold;
#             color: #f8f9fa;
#             margin-bottom: 20px;
#         }

#         .subtitle {
#             font-size: 20px;
#             text-align: center;
#             color: #e0e0e0;
#             margin-bottom: 40px;
#         }

#         /* Chart Section */
#         .chart-card {
#             background: rgba(255, 255, 255, 0.1);
#             padding: 25px;
#             border-radius: 15px;
#             margin-bottom: 30px;
#             box-shadow: 0px 8px 20px rgba(0,0,0,0.25);
#         }
#     </style>
# """, unsafe_allow_html=True)

# # ---------------- Page Header ----------------
# st.markdown("<h1 class='big-title'>📊 Insights & Graphs</h1>", unsafe_allow_html=True)
# st.markdown("<p class='subtitle'>Explore trends, education impacts, and salary distributions</p>", unsafe_allow_html=True)

# # ---------------- Example Dataset ----------------
# np.random.seed(42)
# df = pd.DataFrame({
#     "Education": np.random.choice(["High School", "Bachelors", "Masters", "PhD"], 300),
#     "Experience": np.random.randint(0, 20, 300),
#     "Age": np.random.randint(21, 60, 300),
#     "Skills": np.random.choice(["Python", "Java", "SQL", "ML", "Cloud"], 300),
#     "Salary": np.random.randint(20000, 120000, 300)
# })

# # ---------------- Salary Distribution ----------------
# st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
# st.subheader("💵 Salary Distribution")
# fig, ax = plt.subplots(figsize=(8,4))
# sns.histplot(df["Salary"], bins=20, kde=True, ax=ax, color="#36D7B7")
# ax.set_xlabel("Salary")
# ax.set_ylabel("Count")
# st.pyplot(fig)
# st.markdown("</div>", unsafe_allow_html=True)

# # ---------------- Boxplot by Education ----------------
# st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
# st.subheader("🎓 Salary by Education Level")
# fig, ax = plt.subplots(figsize=(8,4))
# sns.boxplot(x="Education", y="Salary", data=df, ax=ax, palette="Set2")
# ax.set_xlabel("Education Level")
# ax.set_ylabel("Salary")
# st.pyplot(fig)
# st.markdown("</div>", unsafe_allow_html=True)

# # ---------------- Average Salary by Education ----------------
# st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
# st.subheader("📈 Average Salary by Education Level")
# avg_salary = df.groupby("Education")["Salary"].mean().reset_index()
# fig, ax = plt.subplots(figsize=(8,4))
# sns.barplot(x="Education", y="Salary", data=avg_salary, ax=ax, palette="viridis")
# ax.set_ylabel("Average Salary")
# st.pyplot(fig)
# st.markdown("</div>", unsafe_allow_html=True)

# # ---------------- Correlation Heatmap ----------------
# st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
# st.subheader("🔗 Correlation Heatmap (Experience, Age, Salary)")
# corr = df[["Experience", "Age", "Salary"]].corr()
# fig, ax = plt.subplots(figsize=(6,4))
# sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
# st.pyplot(fig)
# st.markdown("</div>", unsafe_allow_html=True)

# # ---------------- Skill-wise Salary Analysis ----------------
# st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
# st.subheader("🛠️ Average Salary by Skill")
# avg_skill_salary = df.groupby("Skills")["Salary"].mean().reset_index()
# fig, ax = plt.subplots(figsize=(8,4))
# sns.barplot(x="Skills", y="Salary", data=avg_skill_salary, ax=ax, palette="magma")
# ax.set_ylabel("Average Salary")
# st.pyplot(fig)
# st.markdown("</div>", unsafe_allow_html=True)


# Streamlit app: Salary Data Insights Dashboard
# Save this file as salary_insights_dashboard.py and run: streamlit run salary_insights_dashboard.py

# Streamlit app: Salary Data Insights Dashboard
# Save this file as salary_insights_dashboard.py and run: streamlit run salary_insights_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Salary Data Insights", layout="wide")

@st.cache_data
def load_default_data():
    # Load the built-in dataset
    path = 'salaries.csv'
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading built-in dataset: {e}")
        return None

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

st.title('Salary Data Insights Dashboard')

st.markdown('You can explore the built-in **salaries.csv** dataset or upload your own CSV files to compare insights visually.')

# Sidebar for file upload
st.sidebar.header('Upload CSV files (optional)')
file1 = st.sidebar.file_uploader('Upload first dataset', type=['csv'], key='file1')
file2 = st.sidebar.file_uploader('Upload second dataset (optional)', type=['csv'], key='file2')

# Load datasets — default first, then user uploads
df_default = load_default_data()
if df_default is not None:
    dfs = {'Built-in Dataset': df_default}
else:
    dfs = {}

if file1:
    dfs['Uploaded Dataset 1'] = load_data(file1)
if file2:
    dfs['Uploaded Dataset 2'] = load_data(file2)

if len(dfs) == 0:
    st.error('No datasets available. Please ensure the built-in dataset exists or upload a CSV.')
    st.stop()

# Tabs for datasets
tabs = st.tabs(list(dfs.keys()))

for i, (name, df) in enumerate(dfs.items()):
    with tabs[i]:
        st.header(f'{name} Overview')
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(50))

        st.subheader('Basic Statistics')
        st.write(df.describe(include='all'))

        st.subheader('Missing Values')
        missing = df.isnull().sum().sort_values(ascending=False)
        if missing.sum() > 0:
            st.bar_chart(missing[missing > 0])
        else:
            st.info('No missing values found.')

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if numeric_cols:
            st.subheader('Numeric Distributions')
            num_col = st.selectbox(f'Select numeric column to visualize ({name})', numeric_cols, key=f'num_{i}')

            fig, ax = plt.subplots()
            sns.histplot(df[num_col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {num_col}')
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df[num_col].dropna(), ax=ax2)
            ax2.set_title(f'Boxplot of {num_col}')
            st.pyplot(fig2)

            if len(numeric_cols) > 1:
                st.subheader('Correlation Heatmap')
                fig3, ax3 = plt.subplots(figsize=(8,6))
                sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax3)
                st.pyplot(fig3)

        if cat_cols:
            st.subheader('Categorical Column Analysis')
            cat_col = st.selectbox(f'Select categorical column to visualize ({name})', cat_cols, key=f'cat_{i}')
            fig4, ax4 = plt.subplots()
            df[cat_col].value_counts().head(20).plot(kind='bar', ax=ax4)
            ax4.set_title(f'Value Counts of {cat_col}')
            st.pyplot(fig4)

        st.subheader('Download Basic Insights')
        insights = {
            'rows': [df.shape[0]],
            'columns': [df.shape[1]],
            'numeric_columns': [', '.join(numeric_cols)],
            'categorical_columns': [', '.join(cat_cols)],
        }
        insights_df = pd.DataFrame(insights)
        st.dataframe(insights_df.T)

        csv = insights_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f'Download {name} Insights CSV',
            data=csv,
            file_name=f'{name.lower().replace(" ", "_")}_insights.csv',
            mime='text/csv'
        )

st.success('Graphs and insights generated successfully!')