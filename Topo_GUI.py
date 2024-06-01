import PySimpleGUI as sg
import matplotlib.pyplot as plt
import os.path
import igraph as ig
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as fct
import re
from algorithm.alg_scfs import *
from algorithm.alg_clink import clink_algorithm, con_rm_gen
from algorithm.alg_map import alg_map


class GUI:
    def __init__(self, role='Manager'):
        self.path_attacked = None
        self.added_func = []
        self.path_obs = None
        self.linkStateInferred = None
        self.linkStateTruth = None
        self.bad_table = None
        self.good_table = None
        self.matrix = None
        self.G = None
        self.win_active = False
        self.left_col = None
        self.right_col = None
        self.layout = None
        self.menu_list = None
        self.filter_tooltip = None
        self.choose_folder_at_top = None
        self.python_only = True
        self.theme_menu = ["Settings", ["Black", "Python", "LightGreen5", "Random"]]
        self.fig, self.ax = plt.subplots(1, 1)
        self.theme = 'GrayGrayGray'
        self.selected = False
        self.drew = False
        self.role = role
        self.topo_name, self.topo_format = None, None
        self.window = None
        self.figure_canvas_agg = None
        self.DR_att = '--'
        self.FPR_att = '--'
        self.F1_att = '--'
        self.nbt_alg = "SCFS"
        if self.role == 'ATTACK':
            self.added_func = [
                [sg.Text('Order:', size=(1, 1), border_width=0,
                         font='Any 10', expand_x=True),
                 sg.Input(focus=True, enable_events=True, size=(15, 1),
                          key='-ATTACK ORDER-', expand_x=True),
                 sg.B('Byzantine Attack Path', expand_x=True, size=(15, 1), key='-ATTACK PATH-'), ],
                [sg.Text(f'Byzantine Attack Rates: DR: {self.DR_att}  FPR: {self.FPR_att}  F1: {self.F1_att}',
                         key='-RATES ATTACK-')]
            ]

    def get_demo_path(self):
        demo_path = sg.user_settings_get_entry(
            '-demos folder-', os.path.dirname(__file__))
        return demo_path

    def get_file_list(self):
        """
        返回当前目录文件
        """
        return sorted(list(self.get_file_list_dict().keys()))

    def get_file_list_dict(self):
        demo_path = self.get_demo_path() + "/topology"
        demo_files_dict = {}
        for dirname, _, filenames in os.walk(demo_path):
            for filename in filenames:
                if self.python_only is not True or filename.endswith('.csv') or filename.endswith('.gml'):
                    fname_full = os.path.join(dirname, filename)
                    if filename not in demo_files_dict.keys():
                        demo_files_dict[filename] = fname_full
                    else:
                        # Allow up to 100 dupicated names. After that, give up
                        for i in range(1, 100):
                            new_filename = f'{filename}_{i}'
                            if new_filename not in demo_files_dict:
                                demo_files_dict[new_filename] = fname_full
                                break

        return demo_files_dict

    def generate_perf(self):
        sg.theme(self.theme)
        sg.menu_list = []

    def generate(self):
        sg.theme(self.theme)
        self.choose_folder_at_top = sg.pin(
            sg.Column([[sg.T('Choose the topology:'),
                        sg.Combo(values=self.get_file_list(), default_value='demoRoutineMatrix.csv', size=(50, 30),
                                 key='-FILE NAME-', enable_events=True, readonly=True),
                        sg.T("File format:"),
                        sg.Combo(["Routine_Matrix(*.csv)", "Adjacency_Matrix(*.csv)", "GML(*.gml)"],
                                 default_value='Routine_Matrix(*.csv)', size=(20, 30), key="-FILE FORMAT-",
                                 enable_events=True, readonly=True),
                        sg.B("Select", key="-SELECT-"),
                        sg.B("Separate display", key="-SHOW-"),
                        ]], pad=(0, 0), k='-FOLDER CHOOSE-'))

        self.filter_tooltip = "Filter files\nEnter a string in box to narrow down the list of files.\n \
                                            File list will update with list of files with string in filename."

        self.menu_list = [
            ["File", ["Open", "Save"]],
            ["Window", [self.theme_menu, "Exit"]]
        ]

        leftColumn = [
            [sg.T("Expected Scene:", size=(50, 1), pad=0)],
            [sg.Menu(self.menu_list, pad=0)],
            [
                sg.Listbox(values=[], select_mode=sg.SELECT_MODE_EXTENDED, size=(50, 8),
                           enable_events=True, key='-GOOD LIST-', expand_x=True, expand_y=True)
            ],
            [sg.T("Unexpected Scene:", size=(50, 1), pad=0)],
            [
                sg.Listbox(values=[], select_mode=sg.SELECT_MODE_EXTENDED, size=(50, 8),
                           enable_events=True, key='-BAD LIST-', expand_x=True, expand_y=True)
            ],
            [
                sg.Text('Network Boolean Tomography:', tooltip=self.filter_tooltip, size=(15, 1), border_width=0,
                        font='Any 10', expand_x=True),
                sg.Combo(["SCFS", "CLINK", "MAP"], size=(5, 30), key="-NBT ALGORITHM-",
                         enable_events=True, readonly=True),
                sg.Button('Select', expand_x=True, key='-NBT SELECT-'),
            ],
            [
                sg.Combo(["|F|", "Y"], size=(5, 30), key="-CONTROL VARIABILITY-", enable_events=True, readonly=True),
                sg.Input(focus=True, enable_events=True, size=(4, 1),
                         key='-C_VARIABILITY NUM-', tooltip=self.filter_tooltip, expand_x=True),
                sg.Button('Generate', expand_x=True, key='-GENERATE-'),
                sg.Button('Save as', expand_x=True, key='-SAVE AS-'),
            ],
            [
                sg.Text('Src:', tooltip=self.filter_tooltip, size=(1, 1), border_width=0,
                        font='Any 10', expand_x=True),
                sg.Input(focus=True, enable_events=True, size=(4, 1),
                         key='-SRC-', tooltip=self.filter_tooltip, expand_x=True),
                sg.Text('Dst:', tooltip=self.filter_tooltip, size=(1, 1), border_width=0,
                        font='Any 10', expand_x=True),
                sg.Input(focus=True, enable_events=True, size=(4, 1),
                         key='-DST-', tooltip=self.filter_tooltip, expand_x=True),
                sg.B("HighLight", size=(15, 1), key='-HIGH LIGHT-', expand_x=True),
            ]
        ]

        # extended function
        if self.role == 'ATTACK':
            leftColumn += self.added_func

        self.left_col = sg.Column(leftColumn, element_justification='l', expand_x=True, expand_y=True)

        self.right_col = [
            [sg.Canvas(size=(70, 21), expand_x=True, expand_y=True, key="-CANVAS-")],
            [sg.B('Exit')]
        ]

        self.layout = [[sg.Text('Visualization of Topography by PySimpleGUI', font='Any 20')],
                       [self.choose_folder_at_top],
                       [sg.Pane(
                           [sg.Column([[self.left_col]], element_justification='l', expand_x=True, expand_y=True),
                            sg.Column(
                                self.right_col, element_justification='c', expand_x=True, expand_y=True)],
                           orientation='h',
                           relief=sg.RELIEF_SUNKEN, expand_x=True, expand_y=True, k='-PANE-',
                           size=(1097, 600))]
                       ]
        self.window = sg.Window("%s_GUI" % self.role, self.layout, finalize=True)
        self.figure_canvas_agg = fct(self.fig, self.window['-CANVAS-'].TKCanvas)

    def update_figure(self):
        self.ax.clear()
        if self.topo_format == "Routine_Matrix(*.csv)" or self.topo_format == "Adjacency_Matrix(*.csv)":
            pos = nx.nx_agraph.graphviz_layout(self.G, prog="dot")
            nx.draw(self.G, pos=pos, with_labels=True, node_size=80, font_size=6, ax=self.ax)
        else:
            layout = self.G.layout("rt")
            ig.plot(self.G, layout=layout, target=self.ax)
        plt.plot()
        self.figure_canvas_agg.draw()
        self.figure_canvas_agg.get_tk_widget().pack()

    def draw_topo(self, A_rm: np.ndarray,
                  links_state: np.ndarray = None, links_state_inferred: np.ndarray = None,
                  paths_state: np.ndarray = None, paths_attacked=None,
                  performance=None, highlight: tuple = None):
        def idx2bin(idx, num):
            # 输入多维的序号集， 返回多维二进制数组, num为链路数
            bin = np.zeros((idx.shape[0], num), dtype=int)
            for i, arr in enumerate(idx):
                if arr.shape[0]:
                    bin[i, arr - 1] = 1
            return bin
        # 创建图像
        # 设置图像参数
        self.ax.clear()
        if paths_attacked is not None:
            paths_truth = paths_state.copy()
            paths_state = paths_attacked.copy()
            links_state_inferred = alg_SCFS(A_rm, paths_state.reshape(1, -1)).transpose()
            links_state_inferred = idx2bin(links_state_inferred, A_rm.shape[1])[0]

        tree_vector = tree_vector_rm(A_rm)
        vertices = tree_vector.shape[0] + 1
        idx = list(range(vertices))  # 转换成列表

        edges = edges_tv(tree_vector)
        leaf_nodes = get_leaf_nodes(A_rm)

        g = nx.DiGraph()
        g.add_nodes_from(idx)
        g.add_edges_from(edges)

        pos = nx.nx_pydot.pydot_layout(g, prog='dot')

        # 可以适当地封装
        # 设置节点颜色字典，将坏路径的目的节点设为"b"
        vcolor_dict = {"n": "#808080", "g": "#228B22", "b": "#CD5C5C"}

        bad_paths = []
        for i in range(paths_state.shape[0]):
            if paths_state[i] == 1:
                bad_paths.append(i + 1)

        if paths_attacked is not None:
            byzantine_path = []
            for k in range(paths_attacked.shape[0]):
                if paths_truth[k] != paths_attacked[k]:
                    byzantine_path.append(k)
            byzantineEndNodes = [leaf_nodes[i] for i in byzantine_path]

        badPathEndNodes = [leaf_nodes[i - 1] for i in bad_paths]
        vertexs_color = np.zeros(17, dtype=str)
        for i in range(17):
            if i in badPathEndNodes:
                vertexs_color[i] = 'b'
            elif i in leaf_nodes:
                vertexs_color[i] = 'g'
            else:
                vertexs_color[i] = 'n'

        # 将链路分为四类，一是链路状态是否判断正确，二是实际链路状态是否正常
        correct_normal = []
        correct_congested = []
        wrong_normal = []
        wrong_congested = []

        for i in range(links_state.shape[0]):
            if links_state[i] == 1:
                if links_state_inferred[i] == 1:
                    correct_congested.append(edges[i])
                else:
                    wrong_congested.append(edges[i])
            else:
                if links_state_inferred[i] == 1:
                    wrong_normal.append(edges[i])
                else:
                    correct_normal.append(edges[i])
        # 创建图像可视化字典
        nodes_style = {"pos": pos,
                       "node_size": 100,
                       "node_shape": 'o',
                       "node_color": [vcolor_dict[obs] for obs in vertexs_color]}
        nx.draw_networkx_nodes(g, **nodes_style, ax=self.ax)
        if paths_attacked is not None:
            g.remove_nodes_from(byzantineEndNodes)
            vertexs_color_byzantine = np.zeros(len(byzantineEndNodes), dtype=str)
            for i in range(len(byzantineEndNodes)):
                if byzantineEndNodes[i] in badPathEndNodes:
                    vertexs_color_byzantine[i] = 'b'
                else:
                    vertexs_color_byzantine[i] = 'g'
            nodes_style_byzantine = {"pos": pos,
                                     "nodelist": byzantineEndNodes,
                                     "node_size": 100,
                                     "node_shape": 's',
                                     "node_color": [vcolor_dict[obs] for obs in vertexs_color_byzantine]}
            nx.draw_networkx_nodes(g, **nodes_style_byzantine, ax=self.ax)
        correct_normal_estyle = {"pos": pos,
                                 "edgelist": correct_normal,
                                 "edge_color": "#90EE90",
                                 "arrowsize": 10,
                                 'width': 1.5,
                                 "style": 'solid'}
        correct_congested_estyle = {"pos": pos,
                                    "edgelist": correct_congested,
                                    "edge_color": "#CD5C5C",
                                    "arrowsize": 10,
                                    'width': 1.5,
                                    "style": 'solid'}

        wrong_normal_estyle = {"pos": pos,
                               "edgelist": wrong_normal,
                               "edge_color": "#90EE90",
                               "arrowsize": 10,
                               'width': 1.5,
                               "style": 'dashed'}
        wrong_congested_estyle = {"pos": pos,
                                  "edgelist": wrong_congested,
                                  "edge_color": "#CD5C5C",
                                  "arrowsize": 10,
                                  'width': 1.5,
                                  "style": 'dashed'}

        nx.draw_networkx_edges(g, **correct_normal_estyle, ax=self.ax)
        nx.draw_networkx_edges(g, **correct_congested_estyle, ax=self.ax)
        nx.draw_networkx_edges(g, **wrong_normal_estyle, ax=self.ax)
        nx.draw_networkx_edges(g, **wrong_congested_estyle, ax=self.ax)
        nx.draw_networkx_labels(
            g, pos, labels={i: i for i in idx}, font_size=7, font_color='#FFFFFF')  # 画标签

        plt.title(performance)
        if performance is None:
            plt.text(1.0, -0.5, performance, size=12,
                     bbox=dict(boxstyle="round", fc="gray", ec="1.0", alpha=0.2))
        if highlight:
            highlight_edges = []
            if highlight[0] == 0:
                highlight_edges.append((0, 1))
                highlight = (1, highlight[1])
            for i in range(A_rm.shape[0]):
                if A_rm[i][highlight[0] - 1] == 1 and A_rm[i][highlight[1] - 1] == 1:
                    highlight_path = A_rm[i]
                    edge_nodes = np.where(highlight_path == 1)[0]
                    endIndex = edge_nodes.tolist().index(highlight[1] - 1)
                    highlight_edges = [(edge_nodes[i] + 1, edge_nodes[i + 1] + 1) for i in range(endIndex)]
                    break
            draw_route = {
                "pos": pos,
                "edgelist": highlight_edges,
                "edge_color": "yellow",
                "arrowsize": 10,
                'width': 1.5,
                'alpha': 0.6,
                "style": 'solid'
            }
            nx.draw_networkx_edges(g, **draw_route, ax=self.ax)
        plt.axis('off')  # 去除图像边框
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 使图像占满输出框
        self.figure_canvas_agg.draw()
        self.figure_canvas_agg.get_tk_widget().pack()

    def event(self, event, values):

        def idx2bin(idx, num):
            # 输入多维的序号集， 返回多维二进制数组, num为链路数
            bin = np.zeros((idx.shape[0], num), dtype=int)
            for i, arr in enumerate(idx):
                if arr.shape[0]:
                    bin[i, arr - 1] = 1
            return bin

        if event in ('Exit', sg.WIN_CLOSED):
            self.window.close()
        if event == '-SELECT-':
            self.topo_name = self.get_demo_path() + "/topology/" + values["-FILE NAME-"]
            self.topo_format = values["-FILE FORMAT-"]
            try:
                self.G = show_topo(self.topo_name, self.topo_format)
                if self.G == 0:
                    pass
                # "Routine_Matrix(*.csv)", "Adjacency_Matrix(*.csv)", "Topology_Zoo(*.gml)"
                elif self.topo_format == "Routine_Matrix(*.csv)":
                    self.matrix = np.array(pd.read_csv(self.topo_name, header=None))
                else:
                    # sg.popup_notify("Error", "暂不支持该类拓扑的 alg_SCFS 算法性能分析", "不过可以可视化")
                    pass
            except Exception as e:
                # sg.popup_notify(e)
                pass
            self.selected = True
        if event == '-SHOW-':
            try:
                if self.selected:
                    self.update_figure()
                else:
                    # sg.popup_notify("Error", "Unselected effective topology")
                    pass
            except Exception as e:
                # sg.popup_notify("Error", e)
                pass

        if event == '-NBT SELECT-':
            if self.selected:
                self.nbt_alg = values["-NBT ALGORITHM-"]

        if event == '-GENERATE-':
            if self.selected:
                if self.topo_format == "Routine_Matrix(*.csv)":
                    if values["-CONTROL VARIABILITY-"] == "|F|":
                        varNum = int(values['-C_VARIABILITY NUM-'])
                        tName = os.path.splitext(os.path.basename(self.topo_name))[0]
                        # links_traversal_F
                        fName = f"./cache/{tName}-{self.nbt_alg}-F{varNum}.csv"
                        if os.path.isfile(fName):
                            df = pd.read_csv(fName)
                        else:
                            multi_conditions = traversal_through_F(self.matrix, varNum)
                            header_emulation = ['f%s' % i for i in range(1, multi_conditions.shape[1] + 1)]
                            df_1 = pd.DataFrame(multi_conditions, columns=header_emulation)
                            paths_obs = paths_stat(self.matrix, multi_conditions)

                            pro = np.random.uniform(.1, .1, self.matrix.shape[1])
                            if self.nbt_alg == "SCFS":
                                # scfs诊断
                                links_state_inferred = alg_SCFS(self.matrix, paths_obs).transpose()
                                links_state_inferred = idx2bin(links_state_inferred, self.matrix.shape[1])
                            elif self.nbt_alg == "CLINK":
                                # clink诊断
                                # # 生成路由矩阵
                                R_set = con_rm_gen(self.matrix, paths_obs, False, None, False)
                                links_state_inferred, _ = clink_algorithm(R_set, pro)
                                links_state_inferred = idx2bin(links_state_inferred, self.matrix.shape[1])
                            else:
                                # map诊断
                                links_state_inferred, _ = alg_map(paths_obs.transpose(), self.matrix, pro)
                                links_state_inferred = idx2bin(links_state_inferred, self.matrix.shape[1])

                            header_inferred = ['y%s' % i for i in range(1, multi_conditions.shape[1] + 1)]
                            df_2 = pd.DataFrame(links_state_inferred, columns=header_inferred)

                            state_truth = df_1.to_numpy()
                            state_inferrd = df_2.to_numpy()
                            DR, FPR, F1 = detection(state_truth, state_inferrd, multiple=True)
                            header_rates = ['DR', 'FPR', 'F1', 'performance']
                            per = ['not modified'] * DR.shape[0]
                            df_3 = pd.DataFrame([[DR[i], FPR[i], F1[i], per[i]] for i in range(DR.shape[0])],
                                                columns=header_rates)
                            for i in range(DR.shape[0]):
                                df_3.loc[i, 'performance'] = 'bad' if df_3.loc[i, 'F1'] < 1 else 'good'

                            df = pd.concat([df_1, df_2, df_3], axis=1).sort_values(by=['F1'],
                                                                                   ascending=False).reset_index(
                                drop=True)
                            df.to_csv(fName, index=False)

                        self.good_table = np.array(df.loc[df['performance'] == 'good'])
                        self.bad_table = np.array(df.loc[df['performance'] == 'bad'])

                        self.window["-GOOD LIST-"].update(
                            list(enumerate(self.good_table.transpose()[-4:-1].transpose().astype(int))))
                        self.window["-BAD LIST-"].update(list(enumerate(self.bad_table.transpose()[-4:-1].transpose())))

                    elif values["-CONTROL VARIABILITY-"] == "Y":
                        varNum = int(values['-C_VARIABILITY NUM-'])
                        tName = os.path.splitext(os.path.basename(self.topo_name))[0]
                        # links_traversal_Y
                        fName = f"./cache/{tName}-{self.nbt_alg}-Y{varNum}.csv"
                        if os.path.isfile(fName):
                            df = pd.read_csv(fName)
                        else:
                            multi_conditions = traversal_through_Y(self.matrix, varNum)
                            header_emulation = ['f%s' % i for i in range(1, multi_conditions.shape[1] + 1)]
                            df_1 = pd.DataFrame(multi_conditions, columns=header_emulation)

                            paths_obs = paths_stat(self.matrix, multi_conditions)

                            pro = np.random.uniform(.1, .1, self.matrix.shape[1])
                            if self.nbt_alg == "SCFS":
                                # scfs诊断
                                links_state_inferred = alg_SCFS(self.matrix, paths_obs).transpose()
                                links_state_inferred = idx2bin(links_state_inferred, self.matrix.shape[1])
                            elif self.nbt_alg == "CLINK":
                                # clink诊断
                                # # 生成路由矩阵
                                R_set = con_rm_gen(self.matrix, paths_obs, False, None, False)
                                links_state_inferred, _ = clink_algorithm(R_set, pro)
                                links_state_inferred = idx2bin(links_state_inferred, self.matrix.shape[1])
                            else:
                                # map诊断
                                links_state_inferred, _ = alg_map(paths_obs.transpose(), self.matrix, pro)
                                links_state_inferred = idx2bin(links_state_inferred, self.matrix.shape[1])

                            header_inferred = ['y%s' % i for i in range(1, multi_conditions.shape[1] + 1)]
                            df_2 = pd.DataFrame(links_state_inferred, columns=header_inferred)

                            state_truth = df_1.to_numpy()
                            state_inferrd = df_2.to_numpy()
                            DR, FPR, F1 = detection(state_truth, state_inferrd, multiple=True)
                            header_rates = ['DR', 'FPR', 'F1', 'performance']
                            per = ['not modified'] * DR.shape[0]
                            df_3 = pd.DataFrame([[DR[i], FPR[i], F1[i], per[i]] for i in range(DR.shape[0])],
                                                columns=header_rates)
                            for i in range(DR.shape[0]):
                                df_3.loc[i, 'performance'] = 'bad' if df_3.loc[i, 'F1'] < 1 else 'good'

                            df = pd.concat([df_1, df_2, df_3], axis=1).sort_values(by=['F1'],
                                                                                   ascending=False).reset_index(
                                drop=True)
                            df.to_csv(fName, index=False)

                        self.good_table = np.array(df.loc[df['performance'] == 'good'])
                        self.bad_table = np.array(df.loc[df['performance'] == 'bad'])

                        self.window["-GOOD LIST-"].update(
                            list(enumerate(self.good_table.transpose()[-4:-1].transpose().astype(int))))
                        self.window["-BAD LIST-"].update(list(enumerate(self.bad_table.transpose()[-4:-1].transpose())))

                    else:
                        # sg.popup_notify("Error", "Unselected control variability")
                        pass
                else:
                    # sg.popup_notify("Error", "暂不支持该类拓扑的 alg_SCFS 算法性能分析")
                    pass

        if event == '-GOOD LIST-':
            if len(values['-GOOD LIST-']) != 0:
                index = values['-GOOD LIST-'][0][0]
                self.linkStateTruth = self.good_table[index][:16]
                self.linkStateInferred = self.good_table[index][16:32]
                self.path_obs = paths_stat_single(self.matrix, self.linkStateTruth)
                self.draw_topo(self.matrix, self.linkStateTruth, self.linkStateInferred, self.path_obs)
                self.drew = True

        if event == '-BAD LIST-':
            if len(values['-BAD LIST-']) != 0:
                index = values['-BAD LIST-'][0][0]
                self.linkStateTruth = np.array(self.bad_table[index][:16])
                self.linkStateInferred = np.array(self.bad_table[index][16:32])
                self.path_obs = paths_stat_single(self.matrix, self.linkStateTruth)
                self.draw_topo(self.matrix, self.linkStateTruth, self.linkStateInferred, self.path_obs)
                self.drew = True

        if event == '-SAVE AS-':
            imgPath = sg.popup_get_file(title='Save as', message='Select a path', default_path='./image/demo.png',
                                        default_extension=".png", save_as=True)
            if imgPath:
                plt.savefig(imgPath)

        if event == '-HIGH LIGHT-':
            if self.drew:
                src = int(values['-SRC-'])
                dst = int(values['-DST-'])
                if src < dst:
                    route = (src, dst)
                    self.draw_topo(self.matrix, self.linkStateTruth, self.linkStateInferred, self.path_obs,
                                   highlight=route)

        if event in self.theme_menu[1]:
            self.window.close()

        if event == '-ATTACK PATH-':
            if self.drew and self.selected:
                self.path_obs = paths_stat_single(self.matrix, self.linkStateTruth)
                paths_order = []
                self.path_attacked = self.path_obs.copy()
                if values['-ATTACK ORDER-'] is not None:
                    raw_str = re.split(',| |，', values['-ATTACK ORDER-'])
                    for i in raw_str:
                        if i.isdigit():
                            paths_order.append(int(i) - 1)
                    for j in paths_order:
                        if j < self.path_obs.shape[0]:
                            self.path_attacked[j] = 0 if self.path_attacked[j] == 1 else 1
                    # renew the topo graph
                    self.draw_topo(self.matrix, self.linkStateTruth, self.linkStateInferred, self.path_obs,
                                   self.path_attacked)
                    # renew the rates
                    pro = np.random.uniform(.1, .1, self.matrix.shape[1])
                    if self.nbt_alg == "SCFS":
                        # scfs诊断
                        links_state_inferred_att = alg_SCFS(self.matrix, self.path_attacked.reshape(1, -1)).transpose()
                        links_state_inferred_att = idx2bin(links_state_inferred_att, self.matrix.shape[1])
                    elif self.nbt_alg == "CLINK":
                        # clink诊断
                        # # 生成路由矩阵
                        R_set = con_rm_gen(self.matrix, self.path_attacked.reshape(1, -1), False, None, False)
                        links_state_inferred_att, _ = clink_algorithm(R_set, pro)
                        links_state_inferred_att = idx2bin(links_state_inferred_att, self.matrix.shape[1])
                    else:
                        # map诊断
                        links_state_inferred_att, _ = alg_map(self.path_attacked.reshape(1, -1).transpose(),
                                                              self.matrix, pro)
                        links_state_inferred_att = idx2bin(links_state_inferred_att, self.matrix.shape[1])

                    self.DR_att, self.FPR_att, self.F1_att = detection(self.linkStateTruth, links_state_inferred_att[0],
                                                                       False)

                    self.window['-RATES ATTACK-'].update(
                        f'Byzantine Attack Rates: DR: {self.DR_att}  FPR: {self.FPR_att}  F1: {self.F1_att}')
