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

    def __init__(self, ps: dict) -> None:
        self.processors = ps
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
                if d1 != d2:
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
            if id is self.processors.keys()[-1]:
                continue
            else:
                ret += ", "
        ret += ">"
        return ret


maca = Processor("MACA")
cv_dsp = Processor("CV_DSP")

bst_chip = Chip({
    "maca": maca,
    "cv_dsp0": cv_dsp,
    "cv_dsp1": cv_dsp
})

cpu_big = Processor("CPU_B")
cpu_small = Processor("CPU_S")
gpu = Processor("GPU")

khadas_chip = Chip(
    {
        "cpu_b": cpu_big,
        "cpu_s": cpu_small,
        "gpu": gpu
    }
)

khadas_chip_cpu_only = Chip(
    {
        "cpu_big": cpu_big,
        "cpu_s": cpu_small
    }
)

supported_chips = {
    "khadas": khadas_chip,
    "bst": bst_chip,
    "khadas_cpu_only": khadas_chip_cpu_only
}


def test_chip():
    assert bst_chip.types_set() == ["MACA", "CV_DSP"]