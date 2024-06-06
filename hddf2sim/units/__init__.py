
from .base import *
from .jsb_units import *

CLASS_TYPE_DICT = {
    JSBAir:['fighter'],
    MidRangeMissile:['mid_missile'], 
    ShortRangeMissile:['short_missile']
}

UNIT_CLASS_DICT = {}
for k, v in CLASS_TYPE_DICT.items():
    for x in v:
        UNIT_CLASS_DICT[x] = k
        UNIT_CLASS_DICT.update({"default": HdUnit})