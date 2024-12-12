import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas import DataFrame

# TODO:
# - corr map: enable multiselection

# st.image(
#     "https://www.basketball-reference.com/req/202106291/images/headshots/hardeja01.jpg"
# )

st.write(
    """
<style>
    .stHeading a {
        text-decoration: none;
        color: inherit;
    }
    .stHeading a:hover {
        text-decoration: none;
        color: revert;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(year: int, measure: str) -> DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_{measure}.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    # Deletes repeating headers in content
    df = df.loc[df.Age != "Age", ~df.columns.isin(["Rk", "Awards"])]
    df = df.loc[df.Team.notna()]
    df = df.fillna(0)
    df = df.astype({"Age": np.int32, "G": np.int32, "GS": np.int32})
    return df


@st.cache_data
def convert_df_to_csv(df: DataFrame, index=True, encoding="utf-8"):
    return df.to_csv(index=index, encoding=encoding)


def main():
    measure_to_url = {
        "Per Game": "per_game",
        "Per 36 Minutes": "per_minute",
        "Per 100 Possessions": "per_poss",
    }

    st.sidebar.title("Enter User Input")
    # Sidebar: year
    selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2026))))
    stats_measures = list(measure_to_url.keys())
    selected_measure = st.sidebar.selectbox("Measure", stats_measures, index=0)

    # name_list = ["Stephen Curry", "Klay Thompson"]
    # store = {}
    # for name in name_list:
    #     store[name] = pd.DataFrame()
    # for year in range(2013, 2016):
    #     try:
    #         stats = load_data(year, measure_to_url[selected_measure])
    #     except Exception:
    #         st.error(f"Year {year}: data is not available.")
    #         break
    #     for name in name_list:
    #         target = stats[stats["Player"] == name]
    #         if not target.empty:
    #             target.insert(0, "Year", year)
    #             target = target.select_dtypes(include=[np.number]).drop(["Age"], axis=1)
    #             target.fillna(0, inplace=True)
    #             store[name] = pd.concat([store[name], target], ignore_index=True)

    # for name in store:
    #     player_stat = store[name]
    #     if player_stat.empty:
    #         st.error(f"No data of {name}.")
    #         break
    #     player_stat.set_index("Year", inplace=True)
    #     print(player_stat)
    #     compare_figure, ax = plt.subplots()
    #     ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    #     # for col in player_stat.columns:
    #     player_stat.plot(ax=ax, title=name)
    #     st.pyplot(compare_figure)
    try:
        playerstats = load_data(selected_year, measure_to_url[selected_measure])
    except Exception:
        st.error("The data is not available.")
        return

    st.sidebar.subheader("[Display Player Stats of Selected Team(s)](#selected-data)", divider=True)
    # Sidebar: team selection
    sorted_unique_team = sorted(playerstats.Team.unique())
    selected_team = st.sidebar.multiselect("Team", sorted_unique_team, sorted_unique_team)

    # Sidebar: position selection
    unique_pos = ["C", "PF", "SF", "PG", "SG"]
    selected_pos = st.sidebar.multiselect("Position", unique_pos, unique_pos)

    df_selected_team = playerstats.loc[(playerstats.Team.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

    # body
    st.title("NBA Player Stats Explorer")
    st.markdown("""
    * **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
    """)
    st.header("Display Player Stats of Selected Team(s)", anchor="selected-data", divider=True)
    st.write("{} rows and {} columns.".format(str(df_selected_team.shape[0]), str(df_selected_team.shape[1])))
    st.dataframe(df_selected_team)

    # Download NBA player stats data
    # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    # def filedownload(df):
    #     csv = df.to_csv(index=False)
    #     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    #     href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    #     return href

    # st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

    playerstats_csv = convert_df_to_csv(df_selected_team, index=False)

    st.download_button(
        label="Download data as csv",
        data=playerstats_csv,
        file_name="playerstats.csv",
        mime="text/csv",
        key="download-player-stats-as-csv",
    )
    # Heatmap
    st.sidebar.subheader("[Correlation Heatmap](#correlation-heatmap)", divider=True)

    # # Sidebar: variable selection
    # numeric_variables = sorted(playerstats..unique())
    # selected_players = st.sidebar.multiselect(
    #     "Player", sorted_unique_players, default=None
    # )
    st.subheader("Correlation Matrix Heatmap", anchor="correlation-heatmap", divider=True)
    # if st.button("Correlation Heatmap"):
    try:
        df_selected_team.to_csv("output.csv", index=False)
        df = pd.read_csv("output.csv")

        corr = df.corr(numeric_only=True)
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            _, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)  # noqa
        st.pyplot()
    except ValueError:
        st.error("Please select teams and positions to view the correlation heatmap.")

    # st.sidebar.subheader("[Compare Selected Player Stats](#player-stats)", divider=True)
    # # Sidebar: Player selection
    # sorted_unique_players = sorted(playerstats.Player.unique())
    # selected_players = st.sidebar.multiselect("Player", sorted_unique_players, default=None)
    #
    # # Sidebar: column selection
    # columns = list(playerstats.select_dtypes("number"))
    # selected_columns = st.sidebar.multiselect("Columns", columns, default=columns)
    #
    # selected_columns_with_player = ["Player"] + selected_columns
    #
    # # Filtering data
    # df_selected_players = playerstats.loc[playerstats.Player.isin(selected_players)]
    # df_selected_players = df_selected_players.loc[:, selected_columns_with_player]
    # st.subheader("Compare Selected Player Stats", anchor="player-stats", divider=True)
    # st.write("{} rows and {} columns.".format(str(df_selected_players.shape[0]), str(df_selected_players.shape[1])))
    # st.dataframe(df_selected_players)
    #
    # # grouped bar chart
    # width = 0.1
    # multiplier = 0
    # fig, ax = plt.subplots(layout="constrained")
    #
    # for attribute, measurement in df_selected_players.items():
    #     offset = width * multiplier
    #     rects = ax.bar(3 + offset, measurement, width, label=attribute)
    #     ax.bar_label(rects, padding=1)
    #     multiplier += 1
    #
    # # ax.set_ylabel()
    # st.pyplot(fig)


if __name__ == "__main__":
    main()
