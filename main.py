import PySimpleGUI as sg
from Topo_GUI import GUI


if __name__ == '__main__':
    sg.theme('GrayGrayGray')

    layout = [[sg.Button('Manager', expand_x=True),
               sg.Button('Attacker', expand_x=True),],
              [sg.Button('Exit', expand_x=True, expand_y=True)]
              ]
    # 创建主菜单
    window = sg.Window('Alter Scenes', layout, size=(400, 90))
    # 设置窗口状态
    win1_active = False
    win2_active = False

    win1 = None
    win2 = None
    # 事件计时器
    while True:
        ev0, vals0 = window.read(timeout=100)
        if ev0 == sg.WIN_CLOSED or ev0 == 'Exit':
            break

        if not win1_active and ev0 == 'Manager':
            win1_active = True
            win1 = GUI()
            win1.generate()

        if not win2_active and ev0 == 'Attacker':
            win2_active = True
            win2 = GUI('ATTACK')
            win2.generate()

        if win1_active:
            ev1, vals1 = win1.window.read(timeout=100)
            if ev1 == sg.WIN_CLOSED or ev1 == 'Exit1':
                win1_active = False
                win1.window.close()
            else:
                win1.event(ev1, vals1)

        if win2_active:
            ev2, vals2 = win2.window.read(timeout=100)
            if ev2 == sg.WIN_CLOSED or ev2 == 'Exit2':
                win2_active = False
                win2.window.close()
            else:
                win2.event(ev2, vals2)

    window.close()
