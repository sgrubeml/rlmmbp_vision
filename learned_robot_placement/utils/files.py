import learned_robot_placement
from pathlib import Path
from os.path import join
from omni.isaac.core.utils.nucleus import get_assets_root_path

# get paths
def get_root_path():
    path = Path(learned_robot_placement.__path__[0]).resolve() / '..' / '..'
    return path


def get_urdf_path():
    path = get_root_path() / 'urdf'
    return path


def get_usd_path():
    path = Path(learned_robot_placement.__path__[0]).resolve()/ 'usd'
    return path


def get_cfg_path():
    path = path = Path(learned_robot_placement.__path__[0]).resolve()/ 'cfg'
    return path

def get_ycb_assets_path():
    path = join(get_assets_root_path(), "/Isaac/Props/YCB/Axis_Aligned")
    return path

def get_shapenet_assets_path():
    return None
