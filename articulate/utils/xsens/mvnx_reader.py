r"""
    Reader for Xsens .mvnx file.
"""


__all__ = ['read_mvnx']


import xml.etree.ElementTree as ET
import torch


def quaternion_product(q1, q2):
    r"""
    Quaternion in w, x, y, z.

    :param q1: Tensor [N, 4].
    :param q2: Tensor [N, 4].
    :return: Tensor [N, 4].
    """
    w1, xyz1 = q1.view(-1, 4)[:, :1], q1.view(-1, 4)[:, 1:]
    w2, xyz2 = q2.view(-1, 4)[:, :1], q2.view(-1, 4)[:, 1:]
    xyz = torch.cross(xyz1, xyz2) + w1 * xyz2 + w2 * xyz1
    w = w1 * w2 - (xyz1 * xyz2).sum(dim=1, keepdim=True)
    q = torch.cat((w, xyz), dim=1).view_as(q1)
    return q


def quaternion_inverse(q):
    r"""
    Quaternion in w, x, y, z.

    :param q: Tensor [N, 4].
    :return: Tensor [N, 4].
    """
    invq = q.clone().view(-1, 4)
    invq[:, 1:].neg_()
    return invq.view_as(q)


def quaternion_normalize(q):
    r"""
    Quaternion in w, x, y, z.

    :param q: Tensor [N, 4].
    :return: Tensor [N, 4].
    """
    q_normalized = q.view(-1, 4) / q.view(-1, 4).norm(dim=1, keepdim=True)
    return q_normalized.view_as(q)


def read_mvnx(file: str):
    r"""
    Parse a mvnx file. All measurements are converted into the SMPL coordinate frame. The result is saved in a dict:

    return dict:
        framerate: int                                 --- fps

        timestamp ms: Tensor[nframes]                  --- timestamp in ms

        center of mass: Tensor[nframes, 3]             --- CoM position

        joint:
            name: List[njoints]                        --- joint order (name)
            <others>: Tensor[nframes, njoints, ndim]   --- other joint properties computed by Xsens

        imu:
            name: List[nimus]                          --- imu order (name)
            <others>: Tensor[nframes, nimus, ndim]     --- other imu measurements

        foot contact:
            name: List[ncontacts]                      --- foot contact order (name)
            label: Tensor[nframes, ncontacts]          --- contact labels

        tpose: ...                                     --- tpose information

    :param file: Xsens file `*.mvnx`.
    :return: The parsed dict.
    """
    tree = ET.parse(file)

    # read framerate
    frameRate = int(tree.getroot()[2].attrib['frameRate'])

    # read joint order
    segments = tree.getroot()[2][1]
    n_joints = len(segments)
    joints = []
    for i in range(n_joints):
        assert int(segments[i].attrib['id']) == i + 1
        joints.append(segments[i].attrib['label'])

    # read imu order
    sensors = tree.getroot()[2][2]
    n_imus = len(sensors)
    imus = []
    for i in range(n_imus):
        imus.append(sensors[i].attrib['label'])

    # read contact order
    footContactDefinition = tree.getroot()[2][5]
    n_contacts = len(footContactDefinition)
    contacts = []
    for i in range(n_contacts):
        assert int(footContactDefinition[i].attrib['index']) == i
        contacts.append(footContactDefinition[i].attrib['label'])

    # read frames
    frames = tree.getroot()[2][6]
    data = {'framerate': frameRate,
            'timestamp ms': [],
            'center of mass': [],
            'joint': {'orientation': [], 'position': [], 'velocity': [], 'acceleration': [],
                      'angular velocity': [], 'angular acceleration': []},
            'imu': {'free acceleration': [], 'magnetic field': [], 'orientation': []},
            'foot contact': {'label': []},
            'tpose': {}
            }
    for i in range(len(frames)):
        if frames[i].attrib['index'] == '':   # tpose
            data['tpose'][frames[i].attrib['type']] = {
                'orientation': torch.tensor([float(_) for _ in frames[i][0].text.split(' ')]).view(n_joints, 4),
                'position': torch.tensor([float(_) for _ in frames[i][1].text.split(' ')]).view(n_joints, 3)
            }
            continue

        assert frames[i].attrib['type'] == 'normal' and \
               int(frames[i].attrib['index']) == len(data['timestamp ms'])

        orientation = torch.tensor([float(_) for _ in frames[i][0].text.split(' ')]).view(n_joints, 4)
        position = torch.tensor([float(_) for _ in frames[i][1].text.split(' ')]).view(n_joints, 3)
        velocity = torch.tensor([float(_) for _ in frames[i][2].text.split(' ')]).view(n_joints, 3)
        acceleration = torch.tensor([float(_) for _ in frames[i][3].text.split(' ')]).view(n_joints, 3)
        angularVelocity = torch.tensor([float(_) for _ in frames[i][4].text.split(' ')]).view(n_joints, 3)
        angularAcceleration = torch.tensor([float(_) for _ in frames[i][5].text.split(' ')]).view(n_joints, 3)
        footContacts = torch.tensor([float(_) for _ in frames[i][6].text.split(' ')]).view(n_contacts)
        sensorFreeAcceleration = torch.tensor([float(_) for _ in frames[i][7].text.split(' ')]).view(n_imus, 3)
        sensorMagneticField = torch.tensor([float(_) for _ in frames[i][8].text.split(' ')]).view(n_imus, 3)
        sensorOrientation = torch.tensor([float(_) for _ in frames[i][9].text.split(' ')]).view(n_imus, 4)
        centerOfMass = torch.tensor([float(_) for _ in frames[i][14].text.split(' ')]).view(3)

        data['timestamp ms'].append(int(frames[i].attrib['time']))
        data['center of mass'].append(centerOfMass)
        data['foot contact']['label'].append(footContacts)
        data['joint']['orientation'].append(orientation)
        data['joint']['position'].append(position)
        data['joint']['velocity'].append(velocity)
        data['joint']['acceleration'].append(acceleration)
        data['joint']['angular velocity'].append(angularVelocity)
        data['joint']['angular acceleration'].append(angularAcceleration)
        data['imu']['free acceleration'].append(sensorFreeAcceleration)
        data['imu']['magnetic field'].append(sensorMagneticField)
        data['imu']['orientation'].append(sensorOrientation)

    data['timestamp ms'] = torch.tensor(data['timestamp ms'])
    data['center of mass'] = torch.stack(data['center of mass'])
    for k, v in data['joint'].items():
        data['joint'][k] = torch.stack(v)
    for k, v in data['imu'].items():
        data['imu'][k] = torch.stack(v)
    for k, v in data['foot contact'].items():
        data['foot contact'][k] = torch.stack(v)
    data['joint']['name'] = joints
    data['imu']['name'] = imus
    data['foot contact']['name'] = contacts

    # to smpl coordinate frame
    def convert_quaternion_(q):
        r""" inplace convert
            R = [[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]]
            smpl_pose = R mvnx_pose R^T
        """
        oldq = q.view(-1, 4).clone()
        q.view(-1, 4)[:, 1] = oldq[:, 2]
        q.view(-1, 4)[:, 2] = oldq[:, 3]
        q.view(-1, 4)[:, 3] = oldq[:, 1]

    def convert_point_(p):
        r""" inplace convert
            R = [[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]]
            smpl_point = R mvnx_point
        """
        oldp = p.view(-1, 3).clone()
        p.view(-1, 3)[:, 0] = oldp[:, 1]
        p.view(-1, 3)[:, 1] = oldp[:, 2]
        p.view(-1, 3)[:, 2] = oldp[:, 0]

    convert_point_(data['center of mass'])
    convert_quaternion_(data['joint']['orientation'])
    convert_point_(data['joint']['position'])
    convert_point_(data['joint']['velocity'])
    convert_point_(data['joint']['acceleration'])
    convert_point_(data['joint']['angular velocity'])
    convert_point_(data['joint']['angular acceleration'])
    convert_quaternion_(data['imu']['orientation'])
    convert_point_(data['imu']['free acceleration'])
    convert_point_(data['imu']['magnetic field'])
    convert_quaternion_(data['tpose']['identity']['orientation'])
    convert_quaternion_(data['tpose']['tpose']['orientation'])
    convert_quaternion_(data['tpose']['tpose-isb']['orientation'])
    convert_point_(data['tpose']['identity']['position'])
    convert_point_(data['tpose']['tpose']['position'])
    convert_point_(data['tpose']['tpose-isb']['position'])

    # use first 150 frames for calibration
    n_frames_for_calibration = 150
    imu_idx = [data['joint']['name'].index(_) for _ in data['imu']['name']]
    q_off = quaternion_product(quaternion_inverse(data['imu']['orientation'][:n_frames_for_calibration]), data['joint']['orientation'][:n_frames_for_calibration, imu_idx])
    ds = q_off.abs().mean(dim=0).max(dim=-1)[1]
    for i, d in enumerate(ds):
        q_off[:, i] = q_off[:, i] * q_off[:, i, d:d+1].sign()
    q_off = quaternion_normalize(quaternion_normalize(q_off).mean(dim=0))
    data['imu']['calibrated orientation'] = quaternion_product(data['imu']['orientation'], q_off.repeat(data['imu']['orientation'].shape[0], 1, 1))

    return data
