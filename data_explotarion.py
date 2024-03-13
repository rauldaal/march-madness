import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# Replace 'data.csv' with your actual file path
df = pd.read_csv('data/MRegularSeasonDetailedResults.csv')

# Assuming 'Team', 'Season', and 'Points' columns exist

teams = df['WTeamID'].unique().tolist()

seasons = df['Season'].unique().tolist()


app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='team-dropdown',
        options=[{'label': team, 'value': team} for team in teams],
        value=teams[0]  # Default selection
    ),
    dcc.Dropdown(
        id='season-dropdown',
        options=[{'label': season, 'value': season} for season in seasons],
        value=seasons[0]  # Default selection
    ),
    dcc.Graph(id='points-graph')
])
@app.callback(
    Output(component_id='points-graph', component_property='figure'),
    Input(component_id='team-dropdown', component_property='value'),
    Input(component_id='season-dropdown', component_property='value'),
)
def update_graph(selected_team, selected_season):
    points = df[(df['Season'] == selected_season) & ((df['WTeamID'] == selected_team) | (df['LTeamID'] == selected_team))]["WScore"].tolist()
    data = [go.Scatter(x=list(range(len(points))), y=points, name=selected_team)]
    figure = go.Figure(data=data)
    figure.update_layout(title='Points Scored in winning game per season', xaxis_title='Game ID', yaxis_title='Points')
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)

