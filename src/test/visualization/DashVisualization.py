import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from data_scripts.CreateEncodingData import create_element_for_each_block
from level.Level import Level
from test.TestEnvironment import TestEnvironment
from test.visualization.LevelVisualisation import create_plotly_data


class LevelVisualization:
    def __init__(self):
        self.test_environment = TestEnvironment('generated/single_structure')

        self.level_img_encoder = LevelImgEncoder()
        self.current_figure = None

        self.encoding = None
        self.level_selection = None
        self.test_level = None
        self.visual = None

        self._app = dash.Dash(__name__)
        self._app.layout = html.Div([
            dcc.Markdown("# Level Visualisation"),
            html.Div([
                html.Div([
                    dcc.Markdown("## Encoding"),
                    dcc.Checklist(
                        id = "encoding",
                        options = [
                            {"label": 'Multilayer', "value": 'multi_layer'},
                            {"label": 'Only One', "value": 'only_one'},
                            {"label": 'Zero Included', "value": 'zero_included'},
                            {"label": 'One Hot', "value": 'one_hot'},
                        ],
                        value = ['multi_layer']
                    )
                ], style = {'margin-right': '3em'}),
                html.Div([
                    dcc.Markdown("## Level Selection"),
                    dcc.Input(
                        id = "level_selection",
                        type = "number",
                        value = 0,
                        placeholder = "input type {}".format("number"),
                    )
                ], style = {'margin-right': '3em'}),
                html.Div([
                    dcc.Markdown("## Control"),
                    dcc.Checklist(
                        id = "test_level",
                        options = [
                            {"label": 'Test Encoding', "value": 'test_encoding'}
                        ],
                        value = []
                    )
                ], style = {'margin-right': '3em'}),
                html.Div([
                    dcc.Markdown("## Download"),
                    html.Button(
                        'Export',
                        id = "export_button",
                        n_clicks = 0
                    ),
                    html.Div(id = 'container-button-timestamp')
                ]),
                html.Div([
                    dcc.Markdown("## Visual"),
                    dcc.Checklist(
                        id = "visual",
                        options = [
                            {"label": 'Seperator', "value": 'seperator'}
                        ],
                        value = ['seperator']
                    )
                ]),
            ], style = {'display': 'flex'}),
            dcc.Graph(id = "voxel-chart", style = {'height': '90vh'}),
            html.Div(id = "output"),

        ])

        self.camera = {'up': {'x': 1, 'y': 0, 'z': 0}, 'center': {'x': -2.7755575615628914e-17, 'y': -6.945473922278155e-18, 'z': -2.220446049250313e-16}, 'eye': {'x': 0.3608546752098405, 'y': -0.6279236433934384, 'z': 1.6254188815527837}, 'projection': {'type': 'orthographic'}}
        self.aspectratio = {'x': 1.4244181662205107, 'y': 3.1002042441269952, 'z': 1.6757860779064833}

        self.defineExportCallBack(self._app)
        self.defineFigureStuff(self._app)
        self.relayout(self._app)

    def defineExportCallBack(self, _app):
        @_app.callback(
            Output('container-button-timestamp', 'children'),
            [
                Input('export_button', 'n_clicks')
            ]
        )
        def displayClick(btn1):

            if self.current_figure is not None:
                print('Export img')
                view = dict(
                    camera = self.camera,
                    aspectratio = self.aspectratio
                )
                self.current_figure.update_layout(scene = view)
                img_bytes = self.current_figure.to_image(format = "pdf", width = 1200, height = 600, scale = 2)
                f = open("exported_img.pdf", "wb")
                f.write(img_bytes)
                f.close()

            return html.Div("Exported")

    def defineFigureStuff(self, _app):
        @_app.callback(
            Output("voxel-chart", "figure"),
            [
                Input("encoding", "value"),
                Input("level_selection", "value"),
                Input('test_level', 'value'),
                Input('visual', 'value'),
            ]
        )
        def update_line_chart(encoding, level_selection, test_level, visual):

            changed = False
            if self.encoding != encoding:
                self.encoding = encoding
                changed = True
                print(self.encoding)

            if self.level_selection != level_selection:
                self.level_selection = level_selection
                changed = True
                print(self.level_selection)

            if self.test_level != test_level:
                self.test_level = test_level
                changed = True
                print(self.test_level)

            if self.visual != visual:
                self.visual = visual
                changed = True
                print(self.visual)

            if changed:
                if 'test_encoding' not in test_level:
                    level = self.test_environment.get_level(level_selection)
                else:
                    elements = create_element_for_each_block()[0]
                    level = Level.create_level_from_structure(elements)

                if 'only_one' in encoding:
                    level_img = self.level_img_encoder.create_one_element_img(
                        level.get_used_elements(),
                        air_layer = 'zero_included' in encoding,
                        multilayer = 'multi_layer' in encoding,
                        true_one_hot = 'one_hot' in encoding
                    )
                else:
                    level_img = self.level_img_encoder.create_calculated_img(level.get_used_elements())

                    if 'multi_layer' in encoding:
                        level_img = self.level_img_encoder.create_multi_dim_img_from_picture(
                            level_img, with_air_layer = 'zero_included' in encoding)

                level_img = np.flip(level_img, axis = 0)

                plotly_data = create_plotly_data(level_img, true_one_hot = 'one_hot' in encoding, seperator = 'seperator' in visual)
                self.current_figure = go.Figure(data = plotly_data)


            if self.current_figure is not None:
                view = dict(
                    camera = self.camera,
                    aspectratio = self.aspectratio
                )
                self.current_figure.update_layout(scene = view)

                self.current_figure.layout.uirevision = True
            return self.current_figure

    def relayout(self,_app):
        @_app.callback(
            Output("output", "children"),
            Input("voxel-chart", "relayoutData")
        )
        def show_data(data):
            # show camera settings like eye upon change
            return [str(data)]

    def start(self):
        print("starting")
        self._app.run_server(debug = True, port = 5052)


if __name__ == '__main__':
    dash_visualization = LevelVisualization()
    dash_visualization.start()
