import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np


#弾性グラフ関数
def plot_elasticity(df, x_column, y_column, title, bin_size=10, bin_threshold=20):
    # binの作成
    bins = np.arange(0, 201, bin_size)
    df = df.copy()
    df['x_bin'] = pd.cut(df[x_column], bins=bins)

    # bin毎の確率を計算
    grouped = df.groupby('x_bin', observed=True)
    total_counts = grouped.size()
    call_or_raise_counts = grouped[y_column].apply(lambda x: ((x == 'call') | (x == 'raise')).sum())
    raise_counts = grouped[y_column].apply(lambda x: (x == 'raise').sum())

    probability_call_or_raise = (call_or_raise_counts / total_counts).reset_index()
    probability_call_or_raise.columns = ['x_bin', 'probability_call_or_raise']

    probability_raise = (raise_counts / total_counts).reset_index()
    probability_raise.columns = ['x_bin', 'probability_raise']

    # binの中央値を計算
    probability_call_or_raise['x_value'] = probability_call_or_raise['x_bin'].apply(lambda x: x.mid)
    probability_raise['x_value'] = probability_raise['x_bin'].apply(lambda x: x.mid)

    # 各ビンのサンプル数を計算
    bin_counts = total_counts.reset_index()
    bin_counts.columns = ['x_bin', 'count']
    bin_counts['x_value'] = bin_counts['x_bin'].apply(lambda x: x.mid)

    # 閾値未満のビンを除外
    valid_bins = bin_counts[bin_counts['count'] >= bin_threshold]['x_bin']
    probability_call_or_raise = probability_call_or_raise[probability_call_or_raise['x_bin'].isin(valid_bins)]
    probability_raise = probability_raise[probability_raise['x_bin'].isin(valid_bins)]
    bin_counts = bin_counts[bin_counts['x_bin'].isin(valid_bins)]

    # サブプロットの作成（2行1列）
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           row_heights=[0.7, 0.3])

    # 折れ線グラフの追加（call or raise）
    fig.add_trace(go.Scatter(x=probability_call_or_raise['x_value'], y=probability_call_or_raise['probability_call_or_raise'], 
                             mode='lines+markers', name='Call+Raise freq'), row=1, col=1)

    # 折れ線グラフの追加（raise only）
    fig.add_trace(go.Scatter(x=probability_raise['x_value'], y=probability_raise['probability_raise'], 
                             mode='lines+markers', name='Raise freq'), row=1, col=1)

    # MDF曲線を追加
    x_range = np.linspace(0, 200, 1000)
    y_curve = 1 / (1 + x_range / 100)
    fig.add_trace(go.Scatter(x=x_range, y=y_curve, mode='lines', name="MDF"),
                  row=1, col=1)

    # 棒グラフの追加
    fig.add_trace(go.Bar(x=bin_counts['x_value'], y=bin_counts['count'], name="Data Count"),
                  row=2, col=1)

    # レイアウトの設定
    fig.update_layout(
        title=title,
        yaxis_title='Frequency',
        yaxis_tickformat='.0%',
        xaxis2_title='Bet Size (%)',
        yaxis2_title='Data Count',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=800  # グラフの高さを調整
    )

    # Y軸の範囲を0-1に設定（上部のグラフ）
    fig.update_yaxes(range=[0, 1], row=1, col=1)

    return fig


def plot_action_distribution(df, column_name, title):
    # カテゴリ列を作成する関数
    def categorize(value):
        if pd.isna(value):
            return 'check'
        elif value >= 200:
            return '200%over'
        elif 0 <= value < 200:
            return f'{int(value // 10) * 10}%-{int((value // 10) + 1) * 10}%'

    # カテゴリ列を作成
    df['category'] = df[column_name].apply(categorize)

    # カテゴリごとの割合を計算
    category_counts = df['category'].value_counts()
    category_percentages = (category_counts / len(df)) * 100

    # カテゴリを適切な順序でソート
    def sort_key(category):
        if category == 'check':
            return -1
        elif category == '200%over':
            return 201
        else:
            return float(category.split('-')[0][:-1])

    sorted_categories = sorted(category_percentages.index, key=sort_key)
    sorted_categories.reverse()

    # グラフの作成
    fig = go.Figure(go.Bar(
        y=sorted_categories,
        x=[category_percentages[cat] for cat in sorted_categories],
        orientation='h',
        text=[f'{category_percentages[cat]:.1f}%' for cat in sorted_categories],
        textposition='outside',
        textfont=dict(size=12),
        marker=dict(line=dict(width=1))
    ))

    # レイアウトの設定
    fig.update_layout(
        title=dict(text=title),
        xaxis_title=dict(text="Frequency"),
        yaxis_title=dict(text='Action'),
        height=max(500, len(sorted_categories) * 30),
    )

    return fig


def plot_hand_distribution(df, category_column, title="Category Proportions"):
    # 空の値を除外し、カテゴリごとの割合を計算
    value_counts = df[category_column].dropna().value_counts()
    total = value_counts.sum()
    proportions = (value_counts / total * 100).sort_values(ascending=True)

    # プロットの作成
    fig = go.Figure(go.Bar(
        y=proportions.index,
        x=proportions.values,
        orientation='h',
        text=[f'{value:.1f}%' for value in proportions.values],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f'{index}: {value:.1f}%' for index, value in zip(proportions.index, proportions.values)]
    ))

    # レイアウトの設定
    fig.update_layout(
        title=title,
        xaxis_title="Frequency",
        yaxis_title="Hand rank",
        height=max(500, len(proportions) * 30),  # グラフの高さを動的に調整
        xaxis=dict(range=[0, 100])
    )

    return fig


# CSVファイルの読み込み関数
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df['Flop action 1'] == 'check']
    df = df[df['Flop action 2'] == 'bet']
    return df


# ここからページ作成
st.set_page_config(layout="wide")
st.title("root-check-betノードの分析画面")
file_name = {"Stars 200nl":"Stars 200nl mda.csv", "Stars 100nl":"Stars 100nl mda.csv", "Stars 50nl":"Stars 50nl mda.csv"}
selected_file = st.sidebar.selectbox("プールデータの選択", ["Stars 200nl", "Stars 100nl", "Stars 50nl"])
df = load_data(file_name[selected_file])
oop_rank = st.sidebar.selectbox("OOP Player Rank", ['All'] + list(df['OOP_player_rank'].unique()))
ip_rank = st.sidebar.selectbox("IP Player Rank", ['All'] + list(df['IP_player_rank'].unique()))

#サイドバー設定
with st.sidebar:
    with st.expander("Preflop設定"):
        pot_range = st.slider("Preflop終了時のpotsize(BB)", 2, 50,(4, 9))
        es_range = st.slider("Preflop終了時のES(BB)", 10, 500,(50, 150))
        aggressor = st.selectbox("Preflop Aggressor", ["All", "OOP", "IP"], index=2)
        oop_positions = st.multiselect("OOP Position", ['All'] + list(df['OOP_position'].unique()),default=['BB'])
        ip_positions = st.multiselect("IP Position", ['All'] + list(df['IP_position'].unique()),default=['All'])

    with st.expander("Community card設定"):
        high_card = st.slider("Flopハイカード", 2, 14,(2, 14))
        flop_type = st.selectbox("Flopのタイプ", ['All'] + list(df['Flop_type'].unique()))

    with st.expander("Postflop設定"):
        flop_bet_size = st.slider("Flop bet size(pot%)", 10, 200,(20, 60))
        turn_bet_size = st.slider("TurnCB size(pot%)", 10, 200,(50, 100))

# フィルタリングを適用
filtered_df = df[
    (df['Pot(BB)'] >= pot_range[0]) & (df['Pot(BB)'] <= pot_range[1]) &
    (df['ES(BB)'] >= es_range[0]) & (df['ES(BB)'] <= es_range[1]) &
    (df['Flop_high'] >= high_card[0]) & (df['Flop_high'] <= high_card[1])
]

if aggressor != 'All':
    filtered_df = filtered_df[filtered_df['Aggressor'] == aggressor]

if 'All' not in oop_positions:
    filtered_df = filtered_df[filtered_df['OOP_position'].isin(oop_positions)]

if 'All' not in ip_positions:
    filtered_df = filtered_df[filtered_df['IP_position'].isin(ip_positions)]

if oop_rank != 'All':
    filtered_df = filtered_df[filtered_df['OOP_player_rank'] == oop_rank]

if ip_rank != 'All':
    filtered_df = filtered_df[filtered_df['IP_player_rank'] == ip_rank]

if flop_type != 'All':
    filtered_df = filtered_df[filtered_df['Flop_type'] == flop_type]

filtered_df = filtered_df[(filtered_df["Flop size 2"] >= flop_bet_size[0]) & (filtered_df["Flop size 2"] <= flop_bet_size[1])]

#サンプルデータ数の表示
st.sidebar.write(f"このノードに到達したデータの数: {len(df)}")
st.sidebar.write(f"フィルタ後のデータ数: {len(filtered_df)}")
st.sidebar.write("Raise sizeは独自の定義をしています size = (raize + bet) / (pot + bet)")

#メインコンテンツ
with st.container():
    st.header("Raise検討用の情報")
    col1, col2 = st.columns(2)
    with col1: #raise弾性グラフ
        raise_df = filtered_df[filtered_df['Flop action 3'] == "raise"]
        fig1 = plot_elasticity(raise_df, "Flop raise size 3", "Flop action 4","Flop XRに対する弾性")
        st.plotly_chart(fig1, use_container_width=True)
    with col2: #raiseCB弾性グラフ
        raise_cb_df = raise_df[(raise_df["Flop action 4"]=="call") & (raise_df["Turn action 1"]=="bet")]
        fig2 = plot_elasticity(raise_cb_df, "Turn size 1", "Turn action 2","Turn raiseCBに対する弾性")
        st.plotly_chart(fig2, use_container_width=True)

with st.container():
    st.header("Call検討用の情報1")
    col3, col4 = st.columns(2)
    with col3:#TurnCBの頻度
        turn_df = filtered_df[(filtered_df['Flop action 3'] == "call") & (filtered_df['Turn action 1'] == "check")]
        fig3 = plot_action_distribution(turn_df,"Turn size 2","Turn CB頻度")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:#RiverBMCB弾性グラフ
        river_df = turn_df[(turn_df["Turn action 2"] == "check") & (turn_df["River action 1"] == "bet")]
        fig4 = plot_elasticity(river_df, "River size 1", "River action 2", "RiverBMCBに対する弾性")
        st.plotly_chart(fig4, use_container_width=True)

with st.container():
    st.header("Call検討用の情報2")
    col5, col6 = st.columns(2)
    with col5:#FlopCBの推定レンジ
        fig5 = plot_hand_distribution(filtered_df,"IP_Flop_hand_rank","Flop betの推定レンジ")
        st.plotly_chart(fig5, use_container_width=True)
    with col6:#TurnCBの推定レンジ
        turn_cb_df = turn_df[(turn_df["Turn action 2"] == "bet") & 
                             (turn_df["Turn size 2"] >= turn_bet_size[0]) & 
                             (turn_df["Turn size 2"] <= turn_bet_size[1])]
        fig6 = plot_hand_distribution(turn_cb_df,"IP_Turn_hand_rank","Turn CBの推定レンジ")
        st.plotly_chart(fig6, use_container_width=True)
