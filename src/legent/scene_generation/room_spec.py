import random
from typing import Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

from attrs import Attribute, define, field

from legent.scene_generation.constants import OUTDOOR_ROOM_ID
#leafroom，单个房间的规格卡片或者蓝图，定义房间不可分割的最基本的单元
class LeafRoom:
    def __init__(#初始化
        self,
        room_id: int,#房间的唯一标识符，在程序中区分不同的房间，即使类型是相同的
        ratio: int,#这个房间所占的面积比例
        room_type: Optional[Literal["Kitchen", "LivingRoom", "Bedroom", "Bathroom"]],#定义房间的功能类型，Literal[...]标识这个参数的值必须是括号里列出来的字符串之一
        avoid_doors_from_metarooms: bool = False,#一个行为提示，用来指导后续的门窗安放逻辑，大概是和门的合法性有关的
    ):
        """
        Parameters:
        - avoid_doors_from_metarooms: prioritize having only 1 door, if possible.
          For example, bathrooms often only have 1 door.
        """
        assert room_type in {"Kitchen", "LivingRoom", "Bedroom", "Bathroom", None}
        if room_id in {0, OUTDOOR_ROOM_ID}:
            raise Exception(f"room_id of 0 and {OUTDOOR_ROOM_ID} are reserved!")

        self.avoid_doors_from_metarooms = avoid_doors_from_metarooms
        self.room_id = room_id
        self.room_type = room_type
        self.ratio = ratio

    def __repr__(self):#生成一个官方的明确的字符串表示，可以打印出这个字符串重新创造出同样的对象
        return (
            "LeafRoom(\n"
            f"    room_id={self.room_id},\n"
            f"    ratio={self.ratio},\n"
            f"    room_type={self.room_type}\n"
            ")"
        )

    def __str__(self):#也是生成一个字符串表示，但是是为了人阅读的
        return self.__repr__()


class MetaRoom:
    def __init__(
        self,
        ratio: int,
        children: Sequence[Union[LeafRoom, "MetaRoom"]],
        room_type: Optional[str] = None,
    ):
        self.ratio = ratio
        self.children = children
        self.room_type = room_type
        self._room_id = None

    @property
    def room_id(self):
        if self._room_id is None:
            raise RuntimeError("room_id not set in MetaRoom!")
        return self._room_id

    @room_id.setter
    def room_id(self, room_id: int):
        if room_id in {0, OUTDOOR_ROOM_ID}:
            raise Exception(f"room_id of 0 and {OUTDOOR_ROOM_ID} are reserved!")
        self._room_id = room_id

    def __repr__(self):
        return f"MetaRoom(ratio={self.ratio}, children={self.children})"

    def __str__(self):
        return self.__repr__()


@define
class RoomSpec:#是整个房间的建筑方案
    room_spec_id: str#建筑方案的名字
    sampling_weight: float = field()#建筑方案多常见或者多么重要
    spec: List[Union[LeafRoom, MetaRoom]]#核心！！！包含了构成房屋的所有房间单元

    dims: Optional[Callable[[], Tuple[int, int]]] = None
    """The (x_size, z_size) dimensions of the house.

    Note that this size will later be scaled up by interior_boundary_scale.
    """
    #map方便程序后续快速访问房间的信息
    room_type_map: Dict[
        int, Literal["Bedroom", "Bathroom", "Kitchen", "LivingRoom"]
    ] = field(init=False)
    room_map: Dict[int, Union[LeafRoom, MetaRoom]] = field(init=False)

    @sampling_weight.validator
    def ge_0(self, attribute: Attribute, value: float):
        if value <= 0:
            raise ValueError(f"sampling_weight must be > 0! You gave {value}.")
    #处理容器Meataroom的id
    def _set_meta_room_ids(
        self,
        spec: List[Union[LeafRoom, MetaRoom]],
        used_ids: Set[int],
        start_at: int = 2,
    ) -> Set[int]:
        """Assign a room_id to each MetaRoom in the RoomSpec."""
        used_ids = used_ids.copy()
        for room in spec:
            if isinstance(room, MetaRoom):#如果是Metaroom
                i = 0
                while (room_id := start_at + i) in used_ids:#给它创建一个新的id作为MetaRoom的ID
                    i += 1

                used_ids.add(room_id)
                room.room_id = room_id
                self.room_map[room_id] = room

                used_ids = self._set_meta_room_ids(
                    spec=room.children, used_ids=used_ids, start_at=room_id + 1
                )
            else:
                self.room_map[room.room_id] = room
        return used_ids
    #遍历spec列表，以及所有嵌套的metaroom，找出leafroom，创建一个房间id->房间类型的字典
    def _get_room_type_map(
        self, spec: List[Union[MetaRoom, LeafRoom]]
    ) -> Dict[int, str]:
        """Set room_type_map to the room_id -> room type for each room in the room spec."""
        room_ids = dict()
        for room in spec:
            if isinstance(room, MetaRoom):#遇到的是MetaRoom，就会调用自己去处理MetaRoom的子房间列表，即roomchildren
                room_ids.update(self._get_room_type_map(room.children))
            else:#遇到的是最小单元LeafRoom，就直接把roomif和toomtype存入字典
                room_ids[room.room_id] = room.room_type
        return room_ids
    #核心入口，在对象的所有基本属性被赋值后自动运行
    def __attrs_post_init__(self) -> None:
        self.room_type_map = self._get_room_type_map(spec=self.spec)#调用getroomtypemap函数来填充roomtypemap
        self.room_map = dict()#初始化selfroommap为一个空字典
        self._set_meta_room_ids(spec=self.spec, used_ids=set(self.room_type_map.keys()))#给Metaroom分配id，Meatroom可以嵌套包含别的房间，是逻辑存在不是物理存在，用来组织和包含其它房间，是一个容器


@define#attrs管理，会自动生成__init__等方法
class RoomSpecSampler:#带权重的抽取一个蓝图
    room_specs: List[RoomSpec] = field()#创建roomspecsampler时必须提供的数据，里面包含了各种各样的房间规格

    room_spec_map: Dict[str, RoomSpec] = field(init=False)#通过id查到roomspec对象
    weights: List[float] = field(init=False)#按顺序存放了每个蓝图的抽样权重
    #确保id时独特的
    @room_specs.validator
    def unique_room_spec_ids(self, attribute: Attribute, value: List[RoomSpec]) -> None:
        room_spec_ids = set()
        for room_spec in value:
            if room_spec.room_spec_id in room_spec_ids:
                raise ValueError(
                    "Each RoomSpec must have a unique room_spec_id."
                    f" You gave duplicate room_spec_id: {room_spec.room_spec_id}."
                )
            room_spec_ids.add(room_spec.room_spec_id)
    #传入room_specs列表，建立起id->蓝图的快速通道，并且把roomspecs的权重提取出来
    def __attrs_post_init__(self) -> None:
        self.room_spec_map = dict()
        self.weights = []
        for room_spec in self.room_specs:
            self.room_spec_map[room_spec.room_spec_id] = room_spec
            self.weights.append(room_spec.sampling_weight)

    def __getitem__(self, room_spec_id: str) -> RoomSpec:
        return self.room_spec_map[room_spec_id]
    #根据设置的权重抽取roomspec
    def sample(self, k: int = 1) -> Union[RoomSpec, List[RoomSpec]]:
        """Return a RoomSpec with weighted sampling."""
        sample = random.choices(self.room_specs, weights=self.weights, k=k)#实现带权重随机抽样的核心，weights列表里的值越高，对应的roomspec被选中的概率越大
        return sample[0] if k == 1 else sample#只抽取1个就直接返回那个roomspec，抽取多个返回列表

#全局标量，提前规定好了不同的room_spec
ROOM_SPEC_SAMPLER = RoomSpecSampler(
    [
        RoomSpec(
            dims=lambda: (random.randint(13, 16), random.randint(5, 8)),
            room_spec_id="8-room-3-bed",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=4,
                    children=[
                        MetaRoom(
                            ratio=2,
                            children=[
                                LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                                LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=4, ratio=2, room_type="LivingRoom"),
                                LeafRoom(
                                    room_id=5,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=1,
                    children=[
                        LeafRoom(room_id=6, ratio=1, room_type="Bedroom"),
                    ],
                ),
                MetaRoom(
                    ratio=1,
                    children=[
                        LeafRoom(room_id=7, ratio=1, room_type="Bedroom"),
                    ],
                ),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=8, ratio=1, room_type="Bedroom"),
                        LeafRoom(
                            room_id=9,
                            ratio=1,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="7-room-3-bed",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=3,
                    children=[
                        MetaRoom(
                            ratio=2,
                            children=[
                                LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                                LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=4, ratio=2, room_type="LivingRoom"),
                                LeafRoom(
                                    room_id=5,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                        LeafRoom(room_id=7, ratio=2, room_type="Bedroom"),
                        LeafRoom(room_id=8, ratio=2, room_type="Bedroom"),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="12-room-3-bed",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=1,
                    children=[
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                                LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=4, ratio=1, room_type="LivingRoom"),
                                LeafRoom(room_id=5, ratio=1, room_type="LivingRoom"),
                            ],
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=1,
                    children=[
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=7,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=8, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=9,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=10, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=11,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="12-room",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                                LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=5,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=3,
                    children=[
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=7,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=8, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=9,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=10, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=11,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="4-room",
            sampling_weight=5,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                        LeafRoom(
                            room_id=5,
                            ratio=1,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=6, ratio=3, room_type="Kitchen"),
                        LeafRoom(room_id=7, ratio=2, room_type="LivingRoom"),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="2-bed-1-bath",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                        LeafRoom(
                            room_id=3,
                            ratio=2,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                        LeafRoom(room_id=4, ratio=3, room_type="LivingRoom"),
                    ],
                ),
                LeafRoom(room_id=5, ratio=1, room_type="Bedroom"),
                LeafRoom(room_id=6, ratio=1, room_type="Bedroom"),
            ],
        ),
        RoomSpec(
            room_spec_id="5-room",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                        LeafRoom(
                            room_id=5,
                            ratio=1,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                    ],
                ),
                LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=8, ratio=3, room_type="Kitchen"),
                        LeafRoom(room_id=9, ratio=2, room_type="LivingRoom"),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="2-bed-2-bath",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                        LeafRoom(
                            room_id=5,
                            ratio=1,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                        LeafRoom(
                            room_id=7,
                            ratio=1,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=8, ratio=3, room_type="Kitchen"),
                        LeafRoom(room_id=9, ratio=2, room_type="LivingRoom"),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="bedroom-bathroom",
            sampling_weight=2,
            spec=[
                LeafRoom(room_id=2, ratio=2, room_type="Bedroom"),
                LeafRoom(room_id=3, ratio=1, room_type="Bathroom"),
            ],
        ),
        RoomSpec(
            room_spec_id="kitchen-living-bedroom-room",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=6, ratio=3, room_type="Kitchen"),
                        LeafRoom(room_id=7, ratio=2, room_type="LivingRoom"),
                    ],
                ),
                LeafRoom(room_id=2, ratio=1, room_type="Bedroom"),
            ],
        ),
        RoomSpec(
            room_spec_id="kitchen-living-bedroom-room2",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=6, ratio=1, room_type="Kitchen"),
                        LeafRoom(room_id=7, ratio=1, room_type="LivingRoom"),
                    ],
                ),
                LeafRoom(room_id=2, ratio=1, room_type="Bedroom"),
            ],
        ),
        RoomSpec(
            room_spec_id="kitchen-living-room",
            sampling_weight=2,
            spec=[
                LeafRoom(room_id=2, ratio=1, room_type="Kitchen"),
                LeafRoom(room_id=3, ratio=1, room_type="LivingRoom"),
            ],
        ),
        RoomSpec(
            room_spec_id="kitchen",
            sampling_weight=1,
            spec=[LeafRoom(room_id=2, ratio=1, room_type="Kitchen")],
        ),
        RoomSpec(
            room_spec_id="living-room",
            sampling_weight=1,
            spec=[LeafRoom(room_id=2, ratio=1, room_type="LivingRoom")],
        ),
        RoomSpec(
            room_spec_id="bedroom",
            sampling_weight=1,
            spec=[LeafRoom(room_id=2, ratio=1, room_type="Bedroom")],
        ),
        RoomSpec(
            # scale=1.25?
            room_spec_id="bathroom",
            sampling_weight=1,
            spec=[LeafRoom(room_id=2, ratio=1, room_type="Bathroom")],
        ),
    ]
)
