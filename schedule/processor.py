"""
Define the unique name, type for devices.
"""

class Processor(object):

    def __init__(self, _id, _type) -> None:
        self.id = _id
        self.type = _type
        pass

    def __str__(self) -> str:
        return self.id

    def __hash__(self) -> int:
        return hash((self.id, self.type))

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, type(self)): return NotImplemented
        return self.id == __value.id and self.type == __value.type

class Chip(object):

    def __init__(self, ps : list) -> None:
        self.processors = ps
        pass

    def ids(self) -> list:
        return [p.id for p in self.processors]

    def types(self) -> list:
        return [p.types for p in self.processors]

    def types_set(self) -> list:
        ret = []
        for p in self.processors:
            if p.type not in ret:
                ret.append(p.type)
        return ret

    def get_processor_by_type(self, type):
        for p in self.processors:
            if p.type == type:
                return p

        return None

    def get_processor_by_id(self, id):
        for p in self.processors:
            if p.id == id:
                return p

        return None

    def get_combinations(self):
        ret = []
        for d1 in self.processors:
            for d2 in self.processors:
                if d1 != d2:
                    ret.append([d1, d2])

        return ret

    def get_id_combinations(self):
        ret = []
        for d1 in self.processors:
            for d2 in self.processors:
                if d1 != d2:
                    ret.append([d1.id, d2.id])

        return ret

    
    def get_type_combinations(self):

        def included(l, a, b):
            for x in l:
                if x[0] == a and x[1] == b:
                    return True
            return False

        ret = []
        for d1 in self.processors:
            for d2 in self.processors:
                if d1.type != d2.type and not included(ret, d1.type, d2.type):
                    ret.append([d1.type, d2.type])

        return ret

    def __str__(self):
        ret = ""
        for p in self.processors:
            ret += str(p)
            if p is self.processors[-1]:
                continue
            else:
                ret += ", "
        return ret


bst_chip = Chip([
    Processor("MACA", "MACA"),
    Processor("CV_DSP_0", "CV_DSP"),
    Processor("CV_DSP_1", "CV_DSP")
])
  
cpu_big = Processor("CPU_B", "CPU_B")
cpu_small = Processor("CPU_S", "CPU_S")
gpu = Processor("GPU", "GPU")

khadas_chip = Chip(
    [
        cpu_big, cpu_small, gpu
    ]
)

khadas_chip_cpu_only = Chip(
    [
        cpu_big, cpu_small 
    ]
)

supported_chips = {
    "khadas": khadas_chip,
    "bst" : bst_chip,
    "khadas_cpu_only": khadas_chip_cpu_only
}

# supported_processors = {
#     "cpu_big" : cpu_big,
#     "cpu_small" : cpu_small,
#     "gpu" : gpu,
# }
