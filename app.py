import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, silhouette_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import io
import re
import unicodedata

st.set_page_config(
    page_title="AI Literacy and Ethical Awareness",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: none; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 52px; padding: 8px 20px;
        background-color: #ffffff; border: 2px solid #1e3a8a;
        border-radius: 6px; color: #1e3a8a; font-weight: 600; font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a !important; color: #ffffff !important;
        border-color: #1e3a8a !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stApp { background-color: #f8fafc; }
    h1, h2, h3, h4 { color: #1e3a8a; }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-left: 4px solid #1e3a8a; border-radius: 8px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #475569; }
    div[data-testid="stMetricValue"] { color: #1e3a8a; }
    section[data-testid="stSidebar"] { background-color: #f1f5f9; }
    </style>
    """,
    unsafe_allow_html=True,
)

KNOWLEDGE_COLS = [
    'KNOWLEDGE [I can define Artificial Intelligence in my own words.]',
    'KNOWLEDGE [I am familiar with common AI tools and applications]',
    'KNOWLEDGE [I know how AI is used in education.]',
    'KNOWLEDGE [I understand the benefits of AI technologies.]',
    'KNOWLEDGE [I know the limitations of AI systems.]',
    'KNOWLEDGE [I understand how AI processes information.]',
    'KNOWLEDGE [I know the difference between AI and traditional software]',
    'KNOWLEDGE [I know that AI can generate inaccurate information.]',
    'KNOWLEDGE [I understand the basic concepts behind AI systems.]',
    'KNOWLEDGE [I know that AI is widely used in industries today.]',
]

SKILLS_COLS = [
    'SKILLS AND TECHNICAL COMPETENCY [I can use AI tools effectively for academic tasks.]',
    'SKILLS AND TECHNICAL COMPETENCY [I can evaluate AI-generated outputs critically.]',
    'SKILLS AND TECHNICAL COMPETENCY [I can use AI for research purposes.]',
    'SKILLS AND TECHNICAL COMPETENCY [I can learn new AI technologies independently.]',
    'SKILLS AND TECHNICAL COMPETENCY [I can choose appropriate AI tools for tasks.]',
    'SKILLS AND TECHNICAL COMPETENCY [I can apply AI tools to solve problems.]',
    'SKILLS AND TECHNICAL COMPETENCY [I can edit AI-generated content effectively.]',
    'SKILLS AND TECHNICAL COMPETENCY [I can integrate AI tools into learning activities.]',
    'SKILLS AND TECHNICAL COMPETENCY [I can use AI tools to improve productivity.]',
    'SKILLS AND TECHNICAL COMPETENCY [I can create prompts for AI tools properly.]',
]

AWARENESS_COLS = [
    'AWARENESS [I am aware of the growing use of AI globally.]',
    'AWARENESS [I am aware of the importance of critical thinking when using AI.]',
    'AWARENESS [I am aware that overreliance on AI may affect learning.]',
    'AWARENESS [I am aware that AI systems can contain bias.]',
    'AWARENESS [I am aware of the importance of responsible AI usage.]',
    'AWARENESS [I am aware of AI policies and guidelines in education.]',
    'AWARENESS [I am aware that AI can spread misinformation.]',
    'AWARENESS [I am aware of ethical concerns regarding AI usage.]',
    'AWARENESS [I am aware of the possible dangers of AI misuse.]',
    'AWARENESS [I am aware that AI can influence education systems.]',
]

ETHICS_COLS = [
    'ETHICAL AWARENESS QUESTIONS [I understand that AI-generated content should always be reviewed before academic or professional submission processes.]',
    'ETHICAL AWARENESS QUESTIONS [I know using AI irresponsibly may result in misinformation, plagiarism, and inaccurate academic work submissions.]',
    'ETHICAL AWARENESS QUESTIONS [I understand students should avoid depending entirely on AI-generated answers during academic learning and assessment activities.]',
    'ETHICAL AWARENESS QUESTIONS [I know AI systems store personal information (e.g., email, contact, picture, etc.)]',
    'ETHICAL AWARENESS QUESTIONS [I understand the importance of protecting personal data when using AI.]',
    'ETHICAL AWARENESS QUESTIONS [I know artificial intelligence systems can produce biased results.]',
    'ETHICAL AWARENESS QUESTIONS [I know responsible AI use includes verifying facts from reliable sources before accepting generated outputs completely.]',
    'ETHICAL AWARENESS QUESTIONS [I understand the ethical risks of AI dependency.]',
    'ETHICAL AWARENESS QUESTIONS [I understand plagiarism may occur when students directly copy AI-generated content without proper modification or citation practices.]',
    'ETHICAL AWARENESS QUESTIONS [I know AI systems cannot always distinguish between accurate information and misleading or false online content sources.]',
    'ETHICAL AWARENESS QUESTIONS [I know excessive reliance on AI tools may negatively affect creativity, critical thinking, and independent learning skills.]',
    'ETHICAL AWARENESS QUESTIONS [I avoid using AI for cheating purposes.]',
    'ETHICAL AWARENESS QUESTIONS [I know developers and users share responsibility for ensuring ethical and safe implementation of AI technologies everywhere.]',
    'ETHICAL AWARENESS QUESTIONS [I know ethical awareness is important when using AI technologies in education, communication, and research-related activities.]',
]

COLLEGE_COL = 'DEMOGRAPHIC PROFILE (COLLEGE)'
SEX_COL = 'DEMOGRAPHIC PROFILE (SEX)'
YEAR_COL = 'DEMOGRAPHIC PROFILE (YEAR LEVEL)'
AGE_COL = 'DEMOGRAPHIC PROFILE (AGE)'

VALID_COLLEGES = [
    'College of Teachers Education',
    'College of Engineering and Industrial Technology',
    'College of Computing and Information Sciences',
    'College of Agriculture',
    'College of Arts and Sciences',
]

CAT_COLORS = {'High': '#059669', 'Moderate': '#d97706', 'Low': '#dc2626'}
CAT_ORDER = ['High', 'Moderate', 'Low']


def clean_text(value):
    if pd.isna(value):
        return value
    text = str(value)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('C'))
    text = re.sub(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def categorize(score):
    if pd.isna(score):
        return np.nan
    if score >= 4.21:
        return 'High'
    if score >= 3.41:
        return 'Moderate'
    return 'Low'


def load_uploaded(uploaded_file):
    if uploaded_file.name.lower().endswith('.csv'):
        return pd.read_csv(uploaded_file)
    xls = pd.ExcelFile(uploaded_file)
    for sheet in xls.sheet_names:
        preview = xls.parse(sheet, nrows=5)
        if len(preview.columns) > 10:
            return xls.parse(sheet)
    return xls.parse(xls.sheet_names[0])


def generate_example(n=80, seed=42):
    rng = np.random.RandomState(seed)
    colleges = ['College of Teachers Education',
                'College of Engineering and Industrial Technology',
                'College of Computing and Information Sciences',
                'College of Agriculture',
                'College of Arts and Sciences']
    sexes = ['Male', 'Female']
    years = ['1st Year', '2nd Year', '3rd Year', '4th Year']
    freq = ['Daily', '3–5 times a week', 'Once a week', '1–3 times a month', 'Rarely']
    dur = ['Less than 10 minutes', '10–30 minutes', '30–60 minutes', 'More than 1 hour']
    purposes = ['Assignments', 'Research', 'Studying/Reviewing lessons',
                'Coding/Programming', 'Personal use']
    tools = ['ChatGPT', 'Google Gemini', 'Quillbot', 'Grammarly', 'Perplexity AI']
    devices = ['Mobile Phone', 'Laptop', 'Tablet']
    places = ['Home', 'School', 'Anywhere (with mobile data/WiFi)']
    programs = ['BSEd', 'BSIS', 'BSIT', 'BIT-ET', 'BAT', 'BEED', 'BSABE']

    data = {}
    data[AGE_COL] = rng.randint(17, 27, n)
    data[SEX_COL] = rng.choice(sexes, n)
    data[COLLEGE_COL] = rng.choice(colleges, n)
    data['DEMOGRAPHIC PROFILE (COURSE/PROGRAM)'] = rng.choice(programs, n)
    data[YEAR_COL] = rng.choice(years, n)
    data['How often do you use AI tools?'] = rng.choice(freq, n)
    data['How long do you typically use AI per session?'] = rng.choice(dur, n)
    data['What do you use AI for?'] = [', '.join(rng.choice(purposes, rng.randint(1, 3), replace=False)) for _ in range(n)]
    data['Which AI tools do you use?'] = [', '.join(rng.choice(tools, rng.randint(1, 3), replace=False)) for _ in range(n)]
    data['What device do you commonly used to access AI tools?'] = [', '.join(rng.choice(devices, rng.randint(1, 2), replace=False)) for _ in range(n)]
    data['Where do you usually access AI tools?'] = [', '.join(rng.choice(places, rng.randint(1, 2), replace=False)) for _ in range(n)]

    latent_lit = rng.normal(3.8, 0.5, n).clip(2.6, 5.0)
    latent_eth = (latent_lit * 0.7 + 1.15 + rng.normal(0, 0.32, n)).clip(2.5, 5.0)
    for cols, latent in [(KNOWLEDGE_COLS, latent_lit),
                         (SKILLS_COLS, latent_lit),
                         (AWARENESS_COLS, latent_lit),
                         (ETHICS_COLS, latent_eth)]:
        for col in cols:
            raw = (latent + rng.normal(0, 0.3, n)).clip(1, 5)
            data[col] = np.round(raw).astype(int)
    return pd.DataFrame(data)


def prepare(df):
    df = df.copy()
    df.columns = [clean_text(c) for c in df.columns]
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]
    df = df.drop_duplicates()

    for col in [AGE_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in [COLLEGE_COL, SEX_COL, YEAR_COL]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    if COLLEGE_COL in df.columns:
        df = df[df[COLLEGE_COL].isin(VALID_COLLEGES)]

    for col in (KNOWLEDGE_COLS + SKILLS_COLS + AWARENESS_COLS + ETHICS_COLS):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    required = [c for c in (KNOWLEDGE_COLS + SKILLS_COLS + AWARENESS_COLS + ETHICS_COLS) if c in df.columns]
    if required:
        threshold = len(required) * 0.25
        df = df[df[required].isna().sum(axis=1) <= threshold]
        df = df.dropna(subset=required, how='all')

    present_k = [c for c in KNOWLEDGE_COLS if c in df.columns]
    present_s = [c for c in SKILLS_COLS if c in df.columns]
    present_a = [c for c in AWARENESS_COLS if c in df.columns]
    present_e = [c for c in ETHICS_COLS if c in df.columns]

    if present_k:
        df['Knowledge'] = df[present_k].mean(axis=1)
    if present_s:
        df['Skills'] = df[present_s].mean(axis=1)
    if present_a:
        df['Awareness'] = df[present_a].mean(axis=1)
    if present_e:
        df['Ethics'] = df[present_e].mean(axis=1)

    dimension_cols = [c for c in ['Knowledge', 'Skills', 'Awareness', 'Ethics'] if c in df.columns]
    if len(dimension_cols) < 4:
        return df

    df = df.dropna(subset=dimension_cols)
    if df.empty:
        return df

    df['AI_Literacy'] = df[['Knowledge', 'Skills', 'Awareness']].mean(axis=1)
    for dim in ['Knowledge', 'Skills', 'Awareness', 'Ethics', 'AI_Literacy']:
        df[f'{dim}_Class'] = df[dim].apply(categorize)
    return df.reset_index(drop=True)


def build_transactions(df):
    transactions = []
    for _, row in df.iterrows():
        items = {
            f"Knowledge_{row['Knowledge_Class']}",
            f"Skills_{row['Skills_Class']}",
            f"Awareness_{row['Awareness_Class']}",
            f"Ethics_{row['Ethics_Class']}",
        }
        transactions.append(items)
    return transactions


def manual_apriori(transactions, min_support=0.1, min_confidence=0.5, min_lift=1.0):
    n = len(transactions)
    if n == 0:
        return pd.DataFrame()
    all_items = sorted({item for txn in transactions for item in txn})

    frequent = {}
    for item in all_items:
        support = sum(1 for txn in transactions if item in txn) / n
        if support >= min_support:
            frequent[frozenset([item])] = support

    k = 2
    while True:
        previous = [fs for fs in frequent if len(fs) == k - 1]
        candidates = set()
        for i, left in enumerate(previous):
            for right in previous[i + 1:]:
                union = left | right
                if len(union) != k or union in candidates:
                    continue
                if all(frozenset(sub) in frequent for sub in combinations(union, k - 1)):
                    candidates.add(union)
        if not candidates:
            break
        new_level = {}
        for cand in candidates:
            support = sum(1 for txn in transactions if cand.issubset(txn)) / n
            if support >= min_support:
                new_level[cand] = support
        if not new_level:
            break
        frequent.update(new_level)
        k += 1

    rules = []
    for itemset, support in frequent.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for ant_tuple in combinations(itemset, r):
                antecedent = frozenset(ant_tuple)
                consequent = itemset - antecedent
                if antecedent not in frequent or consequent not in frequent:
                    continue
                confidence = support / frequent[antecedent]
                if confidence < min_confidence:
                    continue
                lift = confidence / frequent[consequent]
                if lift < min_lift:
                    continue
                rules.append({
                    'Antecedent': ', '.join(sorted(antecedent)),
                    'Consequent': ', '.join(sorted(consequent)),
                    'Support': support,
                    'Confidence': confidence,
                    'Lift': lift,
                })
    return pd.DataFrame(rules)


def support_band(v):
    if v < 0.10: return 'Very Rare Pattern'
    if v < 0.20: return 'Low Frequency Pattern'
    if v < 0.40: return 'Moderately Frequent Pattern'
    if v < 0.60: return 'Frequent Pattern'
    return 'Very Frequent Pattern'


def confidence_band(v):
    if v < 0.50: return 'Weak Rule'
    if v < 0.70: return 'Moderate Rule'
    if v < 0.85: return 'Strong Rule'
    return 'Very Strong Rule'


def lift_band(v):
    if v < 1.00: return 'Negative Association'
    if v == 1.00: return 'No Association'
    if v < 1.50: return 'Weak Positive Association'
    if v < 2.00: return 'Moderate Positive Association'
    return 'Strong Positive Association'


def corr_band(r):
    a = abs(r)
    if a < 0.20: return 'Very Weak'
    if a < 0.40: return 'Weak'
    if a < 0.60: return 'Moderate'
    if a < 0.80: return 'Strong'
    return 'Very Strong'


def render_literacy(df):
    st.markdown("### AI Literacy Dimension Averages")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Knowledge", f"{df['Knowledge'].mean():.3f}")
    c2.metric("Skills", f"{df['Skills'].mean():.3f}")
    c3.metric("Awareness", f"{df['Awareness'].mean():.3f}")
    c4.metric("Overall AI Literacy", f"{df['AI_Literacy'].mean():.3f}")

    means = pd.DataFrame({
        'Dimension': ['Knowledge', 'Skills', 'Awareness', 'Overall AI Literacy'],
        'Mean': [df['Knowledge'].mean(), df['Skills'].mean(),
                 df['Awareness'].mean(), df['AI_Literacy'].mean()],
    })
    means['Category'] = means['Mean'].apply(categorize)

    fig = px.bar(
        means, x='Dimension', y='Mean', color='Category',
        color_discrete_map=CAT_COLORS, text=means['Mean'].round(3),
        category_orders={'Category': CAT_ORDER},
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis_range=[1, 5], height=380, showlegend=True,
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        yaxis_title='Mean Score', xaxis_title='',
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    class_counts = pd.DataFrame({
        'Dimension': [],
        'Category': [],
        'Count': [],
    })
    rows = []
    for dim in ['Knowledge', 'Skills', 'Awareness']:
        counts = df[f'{dim}_Class'].value_counts().reindex(CAT_ORDER, fill_value=0)
        for cat in CAT_ORDER:
            rows.append({'Dimension': dim, 'Category': cat, 'Count': int(counts[cat])})
    class_counts = pd.DataFrame(rows)

    fig2 = px.bar(
        class_counts, x='Dimension', y='Count', color='Category',
        color_discrete_map=CAT_COLORS, barmode='stack',
        category_orders={'Category': CAT_ORDER},
    )
    fig2.update_layout(
        height=340, paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        yaxis_title='Respondent Count', xaxis_title='',
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

    top_dim = means.iloc[:-1]['Mean'].idxmax()
    top_name = means.iloc[top_dim]['Dimension']
    low_name = means.iloc[:-1]['Mean'].idxmin()
    low_name = means.iloc[low_name]['Dimension']
    st.markdown(
        f"Respondents show the strongest literacy in **{top_name}** and the weakest in "
        f"**{low_name}**. Overall AI Literacy falls in the "
        f"**{categorize(df['AI_Literacy'].mean())}** band based on the paper's score interpretation."
    )


def render_ethics(df):
    st.markdown("### Ethical Awareness Averages")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ethics Mean", f"{df['Ethics'].mean():.3f}")
    c2.metric("Ethics Category", categorize(df['Ethics'].mean()))
    c3.metric("Respondents", f"{len(df)}")

    present = [c for c in ETHICS_COLS if c in df.columns]
    if present:
        item_means = df[present].mean().sort_values()
        labels = [c.split('[')[-1].rstrip(']').strip() for c in item_means.index]
        short_labels = [l[:70] + ('...' if len(l) > 70 else '') for l in labels]
        fig = px.bar(
            x=item_means.values, y=short_labels, orientation='h',
            labels={'x': 'Mean Score', 'y': ''},
            color=item_means.values,
            color_continuous_scale=['#dc2626', '#d97706', '#059669'],
            range_color=[1, 5],
        )
        fig.update_layout(
            height=max(420, len(short_labels) * 26),
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    counts = df['Ethics_Class'].value_counts().reindex(CAT_ORDER, fill_value=0)
    fig2 = px.bar(
        x=counts.index, y=counts.values, color=counts.index,
        color_discrete_map=CAT_COLORS, text=counts.values,
        category_orders={'x': CAT_ORDER},
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(
        height=320, showlegend=False, paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff', xaxis_title='Ethics Category',
        yaxis_title='Respondent Count', margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

    pct_high = (df['Ethics_Class'] == 'High').mean() * 100
    st.markdown(
        f"Overall Ethical Awareness is **{categorize(df['Ethics'].mean())}** with "
        f"**{pct_high:.1f}%** of respondents in the High band."
    )


def render_association(df):
    st.markdown("### Association Rules Predicting Ethical Awareness")
    transactions = build_transactions(df)
    rules = manual_apriori(transactions, min_support=0.10,
                           min_confidence=0.50, min_lift=1.00)
    if rules.empty:
        st.info("No association rules satisfied the support, confidence, and lift thresholds.")
        return

    rules = rules[rules['Consequent'].str.contains('Ethics_')].copy()
    if rules.empty:
        st.info("No rules with an Ethics consequent were generated under the current thresholds.")
        return

    rules['Support Band'] = rules['Support'].apply(support_band)
    rules['Confidence Band'] = rules['Confidence'].apply(confidence_band)
    rules['Lift Band'] = rules['Lift'].apply(lift_band)
    rules = rules.sort_values(['Lift', 'Confidence'], ascending=False).reset_index(drop=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rules Generated", len(rules))
    c2.metric("Max Support", f"{rules['Support'].max() * 100:.1f}%")
    c3.metric("Max Confidence", f"{rules['Confidence'].max() * 100:.1f}%")
    c4.metric("Max Lift", f"{rules['Lift'].max():.2f}")

    display = rules.copy()
    display['Support'] = (display['Support'] * 100).round(2).astype(str) + '%'
    display['Confidence'] = (display['Confidence'] * 100).round(2).astype(str) + '%'
    display['Lift'] = display['Lift'].round(2)
    st.dataframe(
        display[['Antecedent', 'Consequent', 'Support', 'Support Band',
                 'Confidence', 'Confidence Band', 'Lift', 'Lift Band']],
        use_container_width=True, hide_index=True,
    )

    scatter_df = rules.copy()
    scatter_df['SupportPct'] = scatter_df['Support'] * 100
    scatter_df['ConfidencePct'] = scatter_df['Confidence'] * 100
    fig = px.scatter(
        scatter_df, x='SupportPct', y='ConfidencePct',
        size='Lift', color='Lift', hover_name='Antecedent',
        color_continuous_scale='Blues',
        labels={'SupportPct': 'Support (%)',
                'ConfidencePct': 'Confidence (%)',
                'Lift': 'Lift'},
    )
    fig.update_layout(
        height=380, paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    top = rules.iloc[0]
    st.markdown(
        f"The strongest rule is **{top['Antecedent']} → {top['Consequent']}** with "
        f"support {top['Support'] * 100:.1f}% ({support_band(top['Support'])}), "
        f"confidence {top['Confidence'] * 100:.1f}% ({confidence_band(top['Confidence'])}), "
        f"and lift {top['Lift']:.2f} ({lift_band(top['Lift'])})."
    )


CLUSTER_LABELS = [
    'AI-Proficient and Ethically Responsible Students',
    'Developing AI Users',
    'At-Risk AI Users',
    'Low AI Literacy Users',
]


def render_clustering(df):
    st.markdown("### K-Means Clusters on AI Literacy and Ethical Awareness")
    features = ['Knowledge', 'Skills', 'Awareness', 'Ethics']
    X = df[features].copy()
    if len(X) < 4:
        st.info("Not enough respondents to run K-Means with four clusters.")
        return
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = model.fit_predict(Xs)
    try:
        sil = silhouette_score(Xs, labels)
    except ValueError:
        sil = float('nan')

    df = df.copy()
    df['Cluster'] = labels
    centroids = df.groupby('Cluster')[features].mean()
    order = centroids.mean(axis=1).sort_values(ascending=False).index.tolist()
    label_map = {cluster: CLUSTER_LABELS[i] for i, cluster in enumerate(order)}
    df['Cluster Label'] = df['Cluster'].map(label_map)

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", 4)
    c2.metric("Silhouette Score", f"{sil:.3f}" if not np.isnan(sil) else "n/a")
    c3.metric("Largest Cluster Share",
              f"{(df['Cluster'].value_counts().max() / len(df)) * 100:.1f}%")

    centroids['Cluster Label'] = [label_map[c] for c in centroids.index]
    centroids = centroids.reset_index()

    fig = go.Figure()
    palette = ['#1e3a8a', '#0891b2', '#f59e0b', '#dc2626']
    for i, feat in enumerate(features):
        fig.add_trace(go.Bar(
            name=feat, x=centroids['Cluster Label'],
            y=centroids[feat], marker_color=palette[i],
        ))
    fig.update_layout(
        barmode='group', height=400,
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        yaxis_title='Mean Score', xaxis_title='',
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    sizes = df['Cluster Label'].value_counts().reset_index()
    sizes.columns = ['Cluster Label', 'Count']
    fig2 = px.pie(
        sizes, values='Count', names='Cluster Label', hole=0.45,
        color_discrete_sequence=['#1e3a8a', '#0891b2', '#f59e0b', '#dc2626'],
    )
    fig2.update_layout(
        height=380, paper_bgcolor='#ffffff',
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

    largest = sizes.iloc[0]
    st.markdown(
        f"The largest cluster is **{largest['Cluster Label']}** with "
        f"**{largest['Count']}** respondents. "
        f"Silhouette of {sil:.3f} indicates "
        f"{'well-separated clusters' if sil > 0.5 else 'moderate cluster separation' if sil > 0.25 else 'overlapping cluster boundaries'}."
    )


def render_classification(df):
    st.markdown("### Decision Tree Classification of Ethical Awareness")
    features = ['Knowledge', 'Skills', 'Awareness']
    X = df[features].copy()
    y = df['Ethics_Class'].copy()
    if y.nunique() < 2 or len(y) < 20:
        st.info("Not enough variation or respondents to train and cross-validate the classifier.")
        return

    model = DecisionTreeClassifier(
        criterion='entropy', max_depth=4,
        min_samples_leaf=3, random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    model.fit(X, y)
    pred = model.predict(X)

    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, average='weighted', zero_division=0)
    rec = recall_score(y, pred, average='weighted', zero_division=0)
    f1 = f1_score(y, pred, average='weighted', zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1-Score", f"{f1:.3f}")

    c5, c6 = st.columns(2)
    c5.metric("5-Fold CV Mean", f"{scores.mean():.3f}")
    c6.metric("5-Fold CV Std", f"{scores.std():.3f}")

    labels = [l for l in CAT_ORDER if l in set(y)]
    cm = confusion_matrix(y, pred, labels=labels)
    fig = px.imshow(
        cm, x=labels, y=labels, text_auto=True,
        color_continuous_scale='Blues', aspect='auto',
        labels={'x': 'Predicted', 'y': 'Actual'},
    )
    fig.update_layout(
        height=360, paper_bgcolor='#ffffff',
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    rules_text = export_text(model, feature_names=features)
    st.code(rules_text, language='text')

    st.markdown(
        f"The entropy-based tree reaches a cross-validated accuracy of "
        f"**{scores.mean():.3f}** and an in-sample F1-Score of **{f1:.3f}** "
        f"when predicting Ethical Awareness from Knowledge, Skills, and Awareness."
    )


def render_correlation(df):
    st.markdown("### Spearman Correlation with Ethical Awareness")
    pairs = [
        ('Knowledge', 'Ethics'),
        ('Skills', 'Ethics'),
        ('Awareness', 'Ethics'),
        ('AI_Literacy', 'Ethics'),
    ]
    rows = []
    for x_col, y_col in pairs:
        sub = df[[x_col, y_col]].dropna()
        if len(sub) < 3:
            continue
        r, p = stats.spearmanr(sub[x_col], sub[y_col])
        direction = 'Positive' if r >= 0 else 'Negative'
        rows.append({
            'Pair': f'{x_col} ↔ {y_col}',
            'r': r,
            'p-value': p,
            'n': len(sub),
            'Strength': corr_band(r),
            'Direction': direction,
            'Significant': 'Yes' if p < 0.05 else 'No',
        })
    if not rows:
        st.info("Not enough paired observations to compute Spearman correlations.")
        return
    table = pd.DataFrame(rows)

    fig = px.bar(
        table, x='Pair', y='r', color='r',
        color_continuous_scale='RdBu', range_color=[-1, 1], text=table['r'].round(3),
    )
    fig.update_traces(textposition='outside')
    fig.add_hline(y=0, line_dash='dash', line_color='#64748b')
    fig.update_layout(
        height=380, paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        yaxis_range=[-1, 1], yaxis_title='Spearman r',
        xaxis_title='', coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    display = table.copy()
    display['r'] = display['r'].round(3)
    display['p-value'] = display['p-value'].apply(lambda v: f"{v:.4f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    top_row = table.iloc[table['r'].abs().argmax()]
    st.markdown(
        f"The strongest association is **{top_row['Pair']}** with r = "
        f"**{top_row['r']:.3f}** ({top_row['Strength']} {top_row['Direction'].lower()} relationship) "
        f"and p = {top_row['p-value']:.4f}. "
        f"{'This relationship is statistically significant.' if top_row['p-value'] < 0.05 else 'This relationship is not statistically significant.'}"
    )


with st.sidebar:
    st.markdown("### Data Source")
    uploaded = st.file_uploader("Upload dataset", type=['csv', 'xlsx', 'xls'])
    use_example = st.checkbox("Use example dataset (if no upload)", value=False)

if uploaded is not None:
    raw = load_uploaded(uploaded)
elif use_example:
    raw = generate_example()
else:
    st.info("Upload a CSV or Excel file, or enable the example dataset option in the sidebar.")
    st.stop()

prepared = prepare(raw)
if prepared.empty or 'AI_Literacy' not in prepared.columns:
    st.info("The dataset could not be prepared. Confirm that the file contains the AI Literacy and Ethical Awareness items.")
    st.stop()

with st.sidebar:
    st.markdown("### Filter")
    if COLLEGE_COL in prepared.columns:
        college_options = sorted(prepared[COLLEGE_COL].dropna().unique())
        selected = st.multiselect("College", college_options, default=college_options)
        if selected:
            prepared = prepared[prepared[COLLEGE_COL].isin(selected)]

if prepared.empty:
    st.info("No respondents match the current filters.")
    st.stop()

st.markdown("# AI Literacy and Ethical Awareness Analysis")

top1, top2, top3, top4 = st.columns(4)
top1.metric("Respondents", len(prepared))
top2.metric("Colleges", prepared[COLLEGE_COL].nunique() if COLLEGE_COL in prepared.columns else 0)
top3.metric("Mean AI Literacy", f"{prepared['AI_Literacy'].mean():.3f}")
top4.metric("Mean Ethical Awareness", f"{prepared['Ethics'].mean():.3f}")

tab_literacy, tab_ethics, tab_arm, tab_cluster, tab_class, tab_corr = st.tabs([
    "AI Literacy Level",
    "Ethical Awareness Level",
    "Association Rules",
    "Clusters",
    "Classification",
    "Correlation Insights",
])

with tab_literacy:
    render_literacy(prepared)
with tab_ethics:
    render_ethics(prepared)
with tab_arm:
    render_association(prepared)
with tab_cluster:
    render_clustering(prepared)
with tab_class:
    render_classification(prepared)
with tab_corr:
    render_correlation(prepared)
