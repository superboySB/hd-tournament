from .grounds import GROUNDS
from .mounts import MISSILES
from .ships import SHIPS
from .uavs import UAVS

def construct_add_list(type_infos, l):
    for v in l:
        type_infos[v['type']] = v

TYPE_INFOS = {}
construct_add_list(TYPE_INFOS, GROUNDS)
construct_add_list(TYPE_INFOS, MISSILES)
construct_add_list(TYPE_INFOS, SHIPS)
construct_add_list(TYPE_INFOS, UAVS)

AIR_TYPE_LIST = [info['type'] for info in UAVS]
MISSILE_TYPE_LIST = [info['type'] for info in MISSILES]
SHIP_TYPE_LIST = [info['type'] for info in SHIPS]
GROUND_TYPE_LIST = [info['type'] for info in GROUNDS]
