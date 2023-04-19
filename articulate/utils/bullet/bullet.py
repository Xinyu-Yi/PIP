r"""
    Utils for pybullet.
"""


__all__ = ['remove_collision', 'change_color', 'load_debug_params_into_bullet_from_json',
           'read_debug_param_values_from_bullet', 'read_debug_param_values_from_json', 'save_debug_params_to_json',
           'Button', 'Slider']


import pybullet as p
import json


_param_attrs = {}
_param_ids = {}
_rbdl_to_bullet = None


def remove_collision(id_a, id_b):
    r"""
    Remove collisions between two robots.
    """
    for i in range(p.getNumJoints(id_a)):
        for j in range(p.getNumJoints(id_b)):
            p.setCollisionFilterPair(id_a, id_b, i, j, 0)


def change_color(id_robot, color):
    r"""
    Change the color of a robot.

    :param id_robot: Robot id.
    :param color: Vector3 for rgb or Vector4 for rgba in [0, 1].
    """
    if len(color) == 3:
        color = (color[0], color[1], color[2], 1)
    for j in range(p.getNumJoints(id_robot)):
        p.changeVisualShape(id_robot, j, rgbaColor=color)


def load_debug_params_into_bullet_from_json(file_path: str):
    r"""
    Load debug parameters into bullet from a json file. See `_example_debug_params.json` for example.
    """
    global _param_attrs, _param_ids
    with open(file_path, 'r') as f:
        _param_attrs = json.load(f)
    for attr in _param_attrs:
        _param_ids[attr['name']] = p.addUserDebugParameter(attr['name'], attr['min'], attr['max'], attr['value'])


def read_debug_param_values_from_bullet():
    r"""
    Read current debug parameter values from bullet.

    :return: A dict for all debug parameters.
    """
    result = {}
    for name, pid in _param_ids.items():
        result[name] = p.readUserDebugParameter(pid)
    return result


def read_debug_param_values_from_json(file_path: str):
    r"""
    Read debug parameter values from a json file.

    :return: A dict for all debug parameters.
    """
    with open(file_path, 'r') as f:
        result = {param['name']: param['value'] for param in json.load(f)}
    return result


def save_debug_params_to_json(param_values=None, file_path='saved_debug_params.json'):
    r"""
    Save debug parameters to a json file. If `param_values` is None, values will be read from bullet.
    """
    if param_values is None:
        param_values = read_debug_param_values_from_bullet()

    for name in _param_attrs.keys():
        _param_attrs[name] = param_values[name]

    with open(file_path, 'w') as f:
        json.dump(_param_attrs, f)


class Button:
    r"""
    Add a pybullet button.
    """
    def __init__(self, name: str, pybullet_server_id=0):
        r"""
        Add a debug pybullet button to GUI.

        :param name: Button name.
        :param pybullet_server_id: Pybullet server id.
        """
        self.pid = pybullet_server_id
        self.btn = p.addUserDebugParameter(' %s ' % name, 1, 0, 0, pybullet_server_id)
        self.n = 0

    def is_click(self) -> bool:
        r"""
        Check if the button is clicked.

        :return: True if the button is once clicked since the last call.
        """
        c = p.readUserDebugParameter(self.btn, self.pid)
        r = c != self.n
        self.n = c
        return r

    def num_clicks(self) -> int:
        r"""
        Return the total number of clicks.
        """
        return int(p.readUserDebugParameter(self.btn, self.pid))


class Slider:
    r"""
    Add a pybullet slider.
    """
    def __init__(self, name: str, range=(0, 1), current=None, pybullet_server_id=0):
        r"""
        Add a debug pybullet slider to GUI.

        :param name: Slider name.
        :param range: Slider value range (min, max).
        :param current: Slider current value. If None, use the min value.
        :param pybullet_server_id: Pybullet server id.
        """
        self.pid = pybullet_server_id
        self.sld = p.addUserDebugParameter(' %s ' % name, range[0], range[1], current or range[0], pybullet_server_id)

    def get_float(self) -> float:
        r"""
        Return the current slider float value.
        """
        return float(p.readUserDebugParameter(self.sld, self.pid))

    def get_int(self) -> int:
        r"""
        Return the current slider int value.
        """
        return int(round(p.readUserDebugParameter(self.sld, self.pid)))