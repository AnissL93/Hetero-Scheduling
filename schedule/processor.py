"""
Define the unique name, type for devices.
"""
class Processor(object):

    def __init__(self, _type) -> None:
        self.type = _type
        pass

    def __str__(self) -> str:
        return self.type

    def __hash__(self) -> int:
        return hash(self.type)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, type(self)): return NotImplemented
        return (self.type == __value.type)


class Chip(object):
    """
    {
      id(str) : Processor
    }
    """

    def __init__(self, ps: dict, _groups : dict = None, _proc_group = None) -> None:
        self.processors = ps
        self.groups = _groups
        self.proc_groups = _proc_group
        pass

    def ids(self) -> list:
        return list(self.processors.keys())

    def types(self) -> list:
        return [p.type for p in self.processors.values()]

    def types_set(self) -> list:
        ret = []
        for p in self.processors.values():
            if p not in ret:
                ret.append(p)
        return ret

    def get_first_processor_by_type(self, type):
        for p in self.processors:
            if p.type == type:
                return p
        return None

    def get_processor_by_type(self, type):
        return [p for p in self.processors.values() if (p.type == type)]

    def get_processor_by_id(self, id):
        assert id in self.processors.keys()
        return self.processors[id]

    def get_combinations(self):
        ret = []
        for d1 in self.processors:
            for d2 in self.processors:
                if d1 != d2:
                    ret.append([d1, d2])

        return ret

    def get_id_combinations(self):
        """
        Get combinations for different ids.
        """
        ret = []
        for d1 in self.processors.keys():
            for d2 in self.processors.keys():
                ret.append([d1, d2])

        return ret

    def get_type_combinations(self):
        """
        Get combinations for different types.
        """
        type_set = self.types_set()
        ret = []
        for t1 in type_set:
            for t2 in type_set:
                if [t1, t2] not in ret:
                    ret.append([t1, t2])

        return ret

    def __str__(self):
        """
        <CPU_B(CPU_B), CPU_S(CPU_S)>
        """
        ret = "<"
        for id, p in self.processors.items():
            ret += f"{id}({p.type})"
            if id is list(self.processors.keys())[-1]:
                continue
            else:
                ret += ", "
        ret += ">"
        return ret

    def get_main_core(self):
        """The first core is the main core."""
        return self.ids()[0]

    def get_group_pid(self, i):
        assert self.groups is not None
        return self.groups[i]

    def get_group_as_chip(self, i):
        assert self.groups is not None
        print(self.groups)
        return Chip({k : self.processors[k] for k in self.groups[i]})

    def get_proc_groups(self):
        return self.proc_groups


maca = Processor("maca")
cv_dsp = Processor("cv_dsp")

bst_chip = Chip({
    "cv_dsp0": cv_dsp,
    "maca": maca,
    "cv_dsp1": cv_dsp
},
{
    "group0": ["cv_dsp0", "maca"], 
    "group1": ["cv_dsp0", "maca", "cv_dsp1"],
    "group2": ["cv_dsp0", "cv_dsp1"],
    "group_dsp0": ["cv_dsp0"],
    "group_dsp1": ["cv_dsp1"],
    "group_maca": ["maca"],
},
[
    ["group0", "group_dsp1"],
    ["group_dsp1", "group0"],
    ["group2", "group_maca"],
    ["group_maca", "group2"],
    ["group1"],
    ["group_maca", "group_dsp0", "group_dsp1"],
    ["group_dsp0", "group_maca", "group_dsp1"],
    ["group_dsp0", "group_dsp1", "group_maca"],
    ["group_dsp0", "group_dsp1"],
    ["group_dsp0", "group_maca"],
    ["group_maca", "group_dsp1"],
]
)

bst_chip_dsp_only = Chip({
    "cv_dsp0": cv_dsp,
    "cv_dsp1": cv_dsp
})

cpu_big = Processor("cpu_b")
cpu_small = Processor("cpu_s")
gpu = Processor("gpu")

khadas_chip = Chip(
    {
        "cpu_b": cpu_big,
        "cpu_s": cpu_small,
        "gpu": gpu
    }, 
    {
        "group0": ["cpu_b", "cpu_s"],
        "group1": ["cpu_b", "cpu_s", "gpu"],
        "group_gpu": ["gpu"],
        "group_big": ["cpu_b"],
        "group_small": ["cpu_s"]
    },
    [
        ["group1"],
        ["group0", "group_gpu"],
        ["group_gpu", "group0"],
        # ["group_gpu", "group_big", "group_small"],
        # ["group_gpu", "group_small", "group_big"],
        # ["group_big", "group_gpu", "group_small"],
        # ["group_big", "group_small", "group_gpu"],
        # ["group_small", "group_gpu", "group_big"],
        # ["group_small", "group_big", "group_gpu"]
    ]
)

khadas_chip_cpu_only = Chip(
    {
        "cpu_b": cpu_big,
        "cpu_s": cpu_small
    }
)

khadas_chip_big_core_gpu = Chip(
    {
        "cpu_b": cpu_big,
        "gpu": gpu
    }
)

supported_chips = {
    "khadas": khadas_chip,
    "bst": bst_chip,
    "bst_dsp_only": bst_chip_dsp_only,
    "khadas_cpu_only": khadas_chip_cpu_only,
    "khadas_cpu_b_gpu" : khadas_chip_big_core_gpu
}

def test_chip():
    assert bst_chip.types_set() == ["maca", "cv_dsp"]