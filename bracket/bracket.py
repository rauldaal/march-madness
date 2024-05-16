import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .container import Container
from .edge import Edge
class Bracket:
    def __init__(self, dataFrame) -> None:
        self.padding = None
        self.dataFrame = dataFrame

        self.levels = self.dataFrame['level'].unique()
        self.num_levels = len(self.dataFrame['level'].unique())

        self.border_padding = 0.05
        self.box_height = None
        self.box_width = None

        self.assigned_colours = {}

        self.nodes = {}
        self.edges = {}

    def _get_teams_per_level(self):
        max_games_per_level = 0
        for level in self.levels:
            games_per_level = len(self.dataFrame[self.dataFrame['level'] == level])
            max_games_per_level = games_per_level if games_per_level > max_games_per_level else max_games_per_level
        return max_games_per_level

    def _set_padding(self):
        # Vertical padding
        max_games = self._get_teams_per_level()
        num_teams = max_games * 2
        space_per_team = (1 - self.border_padding * 2) / num_teams
        self.padding_vertical = space_per_team * 0.1
        self.box_height = space_per_team - self.padding_vertical * 2

        # Horizontal padding
        space_per_round = (1 - self.border_padding * 2) / self.num_levels
        self.padding_horizontal = space_per_round * 0.1
        self.box_width = space_per_round - self.padding_horizontal * 2

    def _calculate_box_position(self, id, level, column, max_on_level):
        iter = id * 2 + column

        # y = self.border_padding + (self.box_height + self.padding_vertical * 2) * iter + self.padding_vertical
        x = self.border_padding + self.box_width * level + (self.padding_horizontal * 2) * level + self.padding_horizontal

        # if iter < max_on_level:
        #     y = 0.5 + y
        # else:
        #     y = 1 - y - self.padding_vertical * 2 - self.box_height
        ######################
        if iter < max_on_level:
            y = 0.5 + (self.box_height + self.padding_vertical * 2) * iter + self.padding_vertical
            predicted = []
            iter_copy = 0
            while iter_copy < max_on_level:
                y_copy = 0.5 + (self.box_height + self.padding_vertical * 2) * iter_copy + self.padding_vertical
                predicted.append(y_copy)
                iter_copy += 1
            predicted.reverse()
            y = predicted[iter]
        else:
            y = 0.5 - (self.box_height + self.padding_vertical * 2) * (iter % max_on_level + 1) - self.padding_vertical
        # adjust order on seed type

        return x, y

    def _asign_colour(self, team):
        # Get N colours, assign them randomly
        if not self.assigned_colours.get(team):
            self.assigned_colours[team] = list(np.random.choice(range(256), size=3) / 255)
        return self.assigned_colours[team]

    def generate_graph(self, column_identifiers=[]):
        """
        column_identifiaers: list - Specific columns to use for node generation. Specify the columns from the dataframe where the team names are stored.
        """
        assert len(column_identifiers) == 2
        self._get_teams_per_level()
        self._set_padding()
        # Iterate games
        # Establecer padding y alturas y anchuras
        # Create rectangel for game
        for level in self.levels:
            for i, (idx, game) in enumerate(self.dataFrame[self.dataFrame['level'] == level].iterrows()):
                for z, column in enumerate(column_identifiers):
                    posx, posy = self._calculate_box_position(i, level, z, len(self.dataFrame[self.dataFrame['level'] == level]))
                    team = game[column]
                    box = Container(identifier=team, posx=posx, posy=posy, height=self.box_height, width=self.box_height, color=self._asign_colour(team))
                    if self.nodes.get(team) is None:
                        self.nodes[team] = [box]
                    else:
                        self.nodes[team].append(box)

        for team, nodes in self.nodes.items():
            if len(nodes) > 1:
                for i in range(1, len(nodes)):
                    edge = Edge(src=nodes[i - 1], dst=nodes[i])
                    if self.edges.get(team) is None:
                        self.edges[team] = [edge]
                    else:
                        self.edges[team].append(edge)

    def plot(self, ax):
        for node in self.nodes.values():
            for container in node:
                container.plot(ax)
        for edges in self.edges.values():
            for edge in edges:
                edge.plot(ax)
        ax.axis("off")
