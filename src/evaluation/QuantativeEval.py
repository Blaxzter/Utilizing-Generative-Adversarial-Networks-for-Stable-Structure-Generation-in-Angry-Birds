import base64
import pickle
from io import BytesIO
from itertools import chain

import dash
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from converter.gan_processing.DecodingFunctions import DecodingFunctions
from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from converter.to_img_converter.MultiLayerStackDecoder import MultiLayerStackDecoder
from evaluation.GridSearchDecode import GeneratedDataset
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment
from util.Config import Config

mpl.rcParams["savefig.format"] = 'pdf'

def fig_to_uri(in_fig, close_all = False, **save_args):
    """https://github.com/4QuantOSS/DashIntro/blob/master/notebooks/Tutorial.ipynb"""
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format = 'png', **save_args, bbox_inches = 0)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


class GridSearchEval:
    def __init__(self):
        self.test_environment = TestEnvironment('generated/single_structure')

        self.level_img_encoder = LevelImgEncoder()
        self.level_visualizer = LevelVisualizer(line_size = 1)

        self.config = Config.get_instance()
        self.grid_search_dataset = self.config.get_grid_search_file("generated_data_set")
        self.grid_search_output = self.config.get_grid_search_file("grid_search_output_new")

        self.multilayer_stack_decoder = MultiLayerStackDecoder()
        self.multilayer_stack_decoder.round_to_next_int = True
        self.multilayer_stack_decoder.custom_kernel_scale = True
        self.multilayer_stack_decoder.minus_one_border = True
        self.multilayer_stack_decoder.combine_layers = True
        self.multilayer_stack_decoder.negative_air_value = -1
        self.multilayer_stack_decoder.cutoff_point = 0.5
        self.multilayer_stack_decoder.display_decoding = False

        # self.multilayer_stack_decoder.visualizer = GanDecodingVisualization(plot_to_file = True)
        self.multilayer_stack_decoder.display_decoding = False
        self.decoder_functions = DecodingFunctions(threshold_callback = lambda: 0)

        with open(self.grid_search_output, 'rb') as f:
            self.data_output_list = pickle.load(f)

        with open(self.grid_search_dataset, 'rb') as f:
            self.generated_dataset: GeneratedDataset = pickle.load(f)

        self.fig = None
        self.sim_data = ['damage', 'is_stable', 'woodBlockDestroyed', 'iceBlockDestroyed', 'stoneBlockDestroyed', 'totalBlocksDestroyed']

        self.meta_data_options = [
            'min_x', 'max_x', 'min_y', 'max_y', 'height', 'width', 'block_amount',
            'platform_amount', 'pig_amount', 'special_block_amount', 'total', 'ice_blocks',
            'stone_blocks', 'wood_blocks'
        ]
        self.collected_data = self.sim_data + self.meta_data_options

        self.show_in_graph = {k: True for k in self.collected_data}
        self.show_in_graph['platform_amount'] = False
        self.show_in_graph['pig_amount'] = False
        self.show_in_graph['special_block_amount'] = False
        self.show_in_graph['woodBlockDestroyed'] = False
        self.show_in_graph['iceBlockDestroyed'] = False
        self.show_in_graph['stoneBlockDestroyed'] = False
        self.show_in_graph['ice_blocks'] = False
        self.show_in_graph['ice_blocks'] = False
        self.show_in_graph['stone_blocks'] = False
        self.show_in_graph['wood_blocks'] = False
        self.show_in_graph['min_x'] = False
        self.show_in_graph['max_x'] = False
        self.show_in_graph['min_y'] = False
        self.show_in_graph['max_y'] = False

        self.collected_data_labels = dict(
            damage = 'Damage',
            is_stable = 'Is Stable',
            woodBlockDestroyed = '# Wood Blocks Destroyed',
            iceBlockDestroyed = '# Ice Blocks Destroyed',
            stoneBlockDestroyed = '# Stone Blocks Destroyed',
            totalBlocksDestroyed = '# Total Blocks Destroyed',
            min_x = 'Min X',
            max_x = 'Max X',
            min_y = 'Min Y',
            max_y = 'Max Y',
            height = 'Height',
            width = 'Width',
            block_amount = '# Block',
            platform_amount = '# Platform',
            pig_amount = '# Pig',
            special_block_amount = '# Special Block',
            total = '# Total Elements',
            ice_blocks = '# ice block',
            stone_blocks = '# stone block',
            wood_blocks = '# wood block'
        )

        self.current_figure = None

        self._app = dash.Dash(__name__)
        self._app.layout = html.Div([
            dcc.Markdown("### Quality Visualisation"),
            html.Div([
                html.Div([
                    dcc.Markdown("#### Filter for"),
                    dcc.Checklist(
                        id = "optimized_value",
                        options = [
                            {"label": 'None', "value": 'none_selected'},
                            {"label": 'Smallest Damage', "value": 'smallest_damage'},
                            {"label": 'Stable', "value": 'stable'},
                            {"label": 'Unstable', "value": 'unstable'},
                            {"label": 'Height', "value": 'height'},
                            {"label": 'Width', "value": 'width'},
                            {"label": 'Block Amount', "value": 'block_amount'},
                            {"label": 'destroyed Blocks', "value": 'destroyed_blocks'}
                        ],
                        value = ['none_selected'],
                        inline = True
                   )], style = {'margin-right': '3em'}),
                html.Div([
                    dcc.Markdown("#### Level Selection"),
                    dcc.Input(
                        id = "level_selection",
                        type = "number",
                        value = 0,
                        placeholder = "input type {}".format("number"),
                    )
                ], style = {'margin-right': '3em'}),
                html.Div([
                    dcc.Markdown("#### Include in graph"),
                    dcc.Checklist(
                        id = "visualize_graph",
                        options = self.collected_data,
                        value = [k for k in self.collected_data if self.show_in_graph[k]]
                    ),
                ], style = {'margin-right': '3em'}),
                html.Div([
                    dcc.Markdown("#### Download"),
                    html.Button(
                        'Export',
                        id = "export_button",
                        n_clicks = 0
                    ),
                    html.Div(id = 'container-button-timestamp')
                ])
            ], style = {'display': 'flex'}),
            html.Div([
                dcc.Graph(id = "bar_chart"),
                html.Img(id = 'level_viz', src = ''),
            ], style = {'display': 'flex', 'width': '100%', 'justify-content': 'center'}),
            html.Div(id = "output"),
        ])

        self.defineExportCallBack(self._app)
        self.defineFigureStuff(self._app)

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
                img_bytes = self.current_figure.to_image(format = "pdf", width = 1200, height = 600, scale = 2)
                f = open("exported_img.pdf", "wb")
                f.write(img_bytes)
                f.close()

            if self.fig is not None:
                self.fig.savefig("Exported_level.pdf")

            return html.Div("Exported")

    def defineFigureStuff(self, _app):
        @_app.callback(
            Output("bar_chart", "figure"),
            Output('level_viz', component_property = 'src'),
            [
                Input("bool_parameters", "value"),
                Input("optimized_value", "value"),
                Input("negative_air_value", "value"),
                Input("cutoff_point", "value"),
                Input("level_selection", "value"),
                Input("biggest_difference", "value"),
                Input("compare_values", "value"),
                Input("compare_selection", "value"),
                Input("visualize_graph", "value"),
            ]
        )
        def update_line_chart(bool_parameters, optimized_value, negative_air_value, cutoff_point, level_selection, biggest_difference,
                              compare_values,
                              compare_selection, visualize_graph):

            if 'Compare Values' in compare_values:
                data = []

                for value in self.parameter_dict[compare_selection]['values']:
                    relevant_data = list(filter(
                        lambda parameter_data: parameter_data['parameter'][compare_selection] == value,
                        self.data_output_list
                    ))

                    collected_data_labels = []
                    y_data = []
                    for collected_data_key in self.collected_data:
                        if collected_data_key not in visualize_graph:
                            continue

                        collected_data_labels.append(self.collected_data_labels[collected_data_key])

                        data_source = 'level_metadata' if collected_data_key in self.meta_data_options else 'sim_data'
                        avg_value_list = list(
                            map(lambda relevant_element: relevant_element['collected'][collected_data_key], relevant_data)
                        )

                        cleared = [i for i in avg_value_list if i is not None and i != -1]
                        if type(cleared[0]) == bool:
                            y_data.append(len([True for value in avg_value_list if value is True]) / len(
                                [True for value in avg_value_list if value is False]) * 100)
                        else:
                            y_data.append(np.average(cleared))

                    data.append(
                        go.Bar(name = f'Value: {value}', x = collected_data_labels, y = y_data)
                    )

                self.current_figure = go.Figure(data = data)
                self.current_figure.update_layout(barmode = 'group')
            else:
                name = None

                if 'none_selected' not in optimized_value:
                    data = []

                    for c_optimized_value in optimized_value:
                        collected_data_labels = []
                        y_data = []

                        index = 0
                        if c_optimized_value == 'smallest_damage':
                            avg_value_list = list(map(lambda relevant_element: relevant_element['collected']['damage'], self.data_output_list))
                            index = np.argmin(avg_value_list)
                        elif c_optimized_value == 'stable':
                            avg_value_list = list(map(lambda relevant_element: relevant_element['collected']['is_stable'], self.data_output_list))
                            index = np.argmax(avg_value_list)
                        elif c_optimized_value == 'height':
                            avg_value_list = list(map(lambda relevant_element: relevant_element['collected']['height'], self.data_output_list))
                            index = np.argmax(avg_value_list)
                        elif c_optimized_value == 'width':
                            avg_value_list = list(map(lambda relevant_element: relevant_element['collected']['width'], self.data_output_list))
                            index = np.argmax(avg_value_list)
                        elif c_optimized_value == 'block_amount':
                            avg_value_list = list(map(lambda relevant_element: relevant_element['collected']['total'], self.data_output_list))
                            index = np.argmax(avg_value_list)
                        elif c_optimized_value == 'destroyed_blocks':
                            avg_value_list = list(map(lambda relevant_element: relevant_element['collected']['totalBlocksDestroyed'], self.data_output_list))
                            index = np.argmin(avg_value_list)

                        comp_para = self.data_output_list[index]['parameter']
                        self.selected_parameters = [k for k in self.parameter_dict.keys() if comp_para[k] == True]
                        self.negative_air_value = comp_para['negative_air_value']
                        self.cutoff_point = comp_para['cutoff_point']

                        print(c_optimized_value + ' '.join([f'{self.parameter_dict[k]["label"]}: {v}' for k, v in comp_para.items()]))

                        found_data = None
                        for recorded_data in self.data_output_list:
                            parameter = recorded_data['parameter']
                            shared_items = {k: comp_para[k] for k in comp_para.keys() if
                                            k in parameter and comp_para[k] == parameter[k]}
                            if len(shared_items.items()) == len(parameter.items()):
                                found_data = recorded_data['data']
                                break

                        for collected_data_key in self.collected_data:
                            if collected_data_key not in visualize_graph:
                                continue
                            collected_data_labels.append(self.collected_data_labels[collected_data_key])

                            data_source = 'level_metadata' if collected_data_key in self.meta_data_options else 'sim_data'
                            avg_value_list = list(map(
                                lambda rec_data: rec_data[data_source][
                                    collected_data_key] if data_source in rec_data else None,
                                found_data.values()
                            ))

                            cleared = [i for i in avg_value_list if i is not None and i != -1]
                            if type(cleared[0]) == bool:
                                y_data.append(len([True for value in avg_value_list if value is True]) / len(
                                    [True for value in avg_value_list if value is False]) * 100)
                            else:
                                y_data.append(np.average(cleared))

                        label_translate = dict(
                            smallest_damage = 'Smallest Damage',
                            stable = 'Stable',
                            height = 'Height',
                            width = 'Width',
                            block_amount = 'Block Amount',
                            destroyed_blocks = 'destroyed Blocks'
                        )

                        data.append(
                            go.Bar(name = label_translate[c_optimized_value], x = collected_data_labels, y = y_data, text = np.round(y_data, decimals = 2))
                        )
                    self.current_figure = go.Figure(data = data)
                    self.current_figure.update_layout(barmode = 'group')
                else:
                    comp_para = {parameter: parameter in bool_parameters for parameter in self.parameter_dict.keys()}
                    comp_para['negative_air_value'] = negative_air_value
                    comp_para['cutoff_point'] = cutoff_point

                    found_data = None
                    for recorded_data in self.data_output_list:
                        parameter = recorded_data['parameter']
                        shared_items = {k: comp_para[k] for k in comp_para.keys() if
                                        k in parameter and comp_para[k] == parameter[k]}
                        if len(shared_items.items()) == len(parameter.items()):
                            found_data = recorded_data['data']
                            break

                    collected_data_labels = []
                    y_data = []
                    data = []
                    for collected_data_key in self.collected_data:
                        if collected_data_key not in visualize_graph:
                            continue
                        collected_data_labels.append(self.collected_data_labels[collected_data_key])

                        data_source = 'level_metadata' if collected_data_key in self.meta_data_options else 'sim_data'
                        avg_value_list = list(map(
                            lambda rec_data: rec_data[data_source][collected_data_key] if data_source in rec_data else None,
                            found_data.values()
                        ))

                        cleared = [i for i in avg_value_list if i is not None and i != -1]
                        if type(cleared[0]) == bool:
                            y_data.append(len([True for value in avg_value_list if value is True]) / len(
                                [True for value in avg_value_list if value is False]) * 100)
                        else:
                            y_data.append(np.average(cleared))

                    data.append(
                        go.Bar(x = collected_data_labels, y = y_data, text = np.round(y_data, decimals = 2))
                    )
                    self.current_figure = go.Figure(data = data)


            plotly_fig = ''
            if 'Compare Values' in compare_values:
                if biggest_difference:

                    data = []
                    data_paramter = []
                    remove_options = []
                    for value in self.parameter_dict[compare_selection]['values']:
                        data_source = 'level_metadata' if compare_selection in self.meta_data_options else 'sim_data'
                        relevant_data = list(filter(
                            lambda parameter_data: parameter_data['parameter'][compare_selection] == value and \
                                                   parameter_data['parameter']['combine_layers'] is True,
                            self.data_output_list
                        ))
                        data_list = list(map(lambda ele: (ele['data'], ele['parameter']), relevant_data))
                        value_list_list = list(map(lambda ele: list(map(lambda _ele: (_ele[0], _ele[1][data_source][biggest_difference], ele[1]) if data_source in _ele[1] else None, ele[0].items())), data_list))
                        value_data_list = list(chain.from_iterable(value_list_list))
                        value_list = list(map(lambda element: element[1] if element is not None else None, value_data_list))


                        data_paramter = value_data_list

                        array = np.array(value_list)
                        remove_options.append(array == None)
                        remove_options.append(array == -1)
                        array[array == None] = -1
                        array[array == -1] = -1
                        data.append(array)

                    differences = []
                    for element_idx in range(len(data) - 1):
                        differences.append(np.abs(data[element_idx] - data[element_idx + 1]))

                    for difference in differences:
                        for remove_option in remove_options:
                            difference[remove_option] = 0

                    if len(differences) > 1:
                        differences = [np.sum(np.array(differences), axis = 0)]

                    possible_values = self.parameter_dict[compare_selection]['values']

                    arg_idx = np.argmax(differences[0])
                    self.fig, self.axs = plt.subplots(1, 1 + len(possible_values), figsize = (6.4 + len(possible_values), 4.8))

                    parameter_set = data_paramter[arg_idx][2]
                    level_idx = data_paramter[arg_idx][0]
                    gan_output = self.generated_dataset.imgs[level_idx]
                    img, norm_img = self.decoder_functions.argmax_multilayer_decoding_with_air(gan_output,
                                                                                               rescale = False)
                    self.axs[0].imshow(img)

                    for idx, param_value in enumerate(possible_values):
                        parameter_set[compare_selection] = param_value
                        self.multilayer_stack_decoder.use_negative_air_value = True
                        for key, value in parameter_set.items():
                            if key == 'negative_air_value' and value == 0:
                                self.multilayer_stack_decoder.use_negative_air_value = False

                            if hasattr(self.multilayer_stack_decoder, key):
                                setattr(self.multilayer_stack_decoder, key, value)

                        level = self.multilayer_stack_decoder.decode(gan_output, has_air_layer = True)
                        self.level_visualizer.create_img_of_level(level, ax = self.axs[idx + 1])
                        self.axs[idx + 1].set_title(f'{self.collected_data_labels[biggest_difference]} {np.round(data[idx][arg_idx])}')
                        print(f'Level {level_idx}')
                        print(f'Parameter {parameter_set}')

                    plotly_fig = fig_to_uri(self.fig)
                else:
                    possible_values = self.parameter_dict[compare_selection]['values']
                    curr_parameter = {parameter: parameter in bool_parameters for parameter in
                                      self.parameter_dict.keys()}
                    curr_parameter['negative_air_value'] = negative_air_value
                    curr_parameter['cutoff_point'] = cutoff_point
                    self.fig, self.axs = plt.subplots(1, len(possible_values) + 1, figsize = (6.4 + len(possible_values), 4.8))
                    gan_output = self.generated_dataset.imgs[level_selection]

                    img, norm_img = self.decoder_functions.argmax_multilayer_decoding_with_air(gan_output,
                                                                                               rescale = False)
                    self.axs[0].imshow(img)
                    for idx, param_value in enumerate(self.parameter_dict[compare_selection]['values']):
                        curr_parameter[compare_selection] = param_value

                        self.multilayer_stack_decoder.use_negative_air_value = True
                        for key, value in curr_parameter.items():
                            if key == 'negative_air_value' and value == 0:
                                self.multilayer_stack_decoder.use_negative_air_value = False

                            if hasattr(self.multilayer_stack_decoder, key):
                                setattr(self.multilayer_stack_decoder, key, value)

                        level = self.multilayer_stack_decoder.decode(gan_output, has_air_layer = True)
                        self.level_visualizer.create_img_of_level(level, ax = self.axs[idx + 1])
                        self.axs[idx + 1].set_title(f'Value {param_value}')

                    plotly_fig = fig_to_uri(self.fig)
            else:
                self.fig, self.axs = plt.subplots(1, 2)
                gan_output = self.generated_dataset.imgs[level_selection]
                curr_parameter = {parameter: parameter in bool_parameters for parameter in self.parameter_dict.keys()}
                curr_parameter['negative_air_value'] = negative_air_value
                curr_parameter['cutoff_point'] = cutoff_point

                self.multilayer_stack_decoder.use_negative_air_value = True
                for key, value in curr_parameter.items():
                    if key == 'negative_air_value' and value == 0:
                        self.multilayer_stack_decoder.use_negative_air_value = False

                    if hasattr(self.multilayer_stack_decoder, key):
                        setattr(self.multilayer_stack_decoder, key, value)

                level = self.multilayer_stack_decoder.decode(gan_output, has_air_layer = True)
                self.level_visualizer.create_img_of_level(level, ax = self.axs[0])

                img, norm_img = self.decoder_functions.argmax_multilayer_decoding_with_air(gan_output, rescale = False)
                self.axs[1].imshow(img)

                plotly_fig = fig_to_uri(self.fig)

            return self.current_figure, plotly_fig

    def start(self):
        print("starting")
        self._app.run_server(debug = True, port = 5052)

    def preprocess_grid_search(self):

        # Add missing comulative data
        for parameter_data in self.data_output_list:
            for data in parameter_data['data'].values():
                if 'sim_data' in data:
                    data['sim_data']['totalBlocksDestroyed'] = data['sim_data']['woodBlockDestroyed'] + data['sim_data']['iceBlockDestroyed'] + data['sim_data']['stoneBlockDestroyed']

        for parameter_data in self.data_output_list:
            combined_data = dict()
            for collected_data_key in self.collected_data:

                data_source = 'level_metadata' if collected_data_key in self.meta_data_options else 'sim_data'
                avg_value_list = list(
                        list(map(lambda rec_data: rec_data[data_source][
                            collected_data_key] if data_source in rec_data else None,
                                 parameter_data['data'].values()))
                )

                cleared = [i for i in avg_value_list if i is not None and i != -1]
                if type(cleared[0]) == bool:
                    value = len([True for value in avg_value_list if value is True]) / \
                            len([True for value in avg_value_list if value is False]) * 100
                else:
                    value = np.average(cleared)

                combined_data[collected_data_key] = value

            parameter_data['collected'] = combined_data

        with open(self.grid_search_output, 'wb') as handle:
            pickle.dump(self.data_output_list, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    dash_visualization = GridSearchEval()
    # dash_visualization.preprocess_grid_search()
    dash_visualization.start()
