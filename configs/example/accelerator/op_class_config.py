"""FU descriptors (gem5 idiom) for the Vector CPU's functional units"""

from enum import Enum

try:
    from m5.objects.FuncUnit import FUDesc, OpDesc
except ImportError:
    class OpDesc:
        def __init__(self, opClass, opLat=1, pipelined=True):
            self.opClass = opClass
            self.opLat = opLat
            self.pipelined = pipelined

    class FUDesc:
        pass


# NOTE: https://github.com/gem5/gem5/blob/stable/src/cpu/FuncUnit.py
class OpClass(str, Enum):
    IntAlu = "IntAlu"
    IntMult = "IntMult"
    IntDiv = "IntDiv"
    FloatAdd = "FloatAdd"
    FloatCmp = "FloatCmp"
    FloatCvt = "FloatCvt"
    Bf16Cvt = "Bf16Cvt"
    FloatMult = "FloatMult"
    FloatMultAcc = "FloatMultAcc"
    FloatMisc = "FloatMisc"
    FloatDiv = "FloatDiv"
    FloatSqrt = "FloatSqrt"
    SimdAdd = "SimdAdd"
    SimdAddAcc = "SimdAddAcc"
    SimdAlu = "SimdAlu"
    SimdCmp = "SimdCmp"
    SimdCvt = "SimdCvt"
    SimdMisc = "SimdMisc"
    SimdMult = "SimdMult"
    SimdMultAcc = "SimdMultAcc"
    SimdMatMultAcc = "SimdMatMultAcc"
    SimdShift = "SimdShift"
    SimdShiftAcc = "SimdShiftAcc"
    SimdDiv = "SimdDiv"
    SimdSqrt = "SimdSqrt"
    SimdFloatAdd = "SimdFloatAdd"
    SimdFloatAlu = "SimdFloatAlu"
    SimdFloatCmp = "SimdFloatCmp"
    SimdFloatCvt = "SimdFloatCvt"
    SimdFloatDiv = "SimdFloatDiv"
    SimdFloatMisc = "SimdFloatMisc"
    SimdFloatMult = "SimdFloatMult"
    SimdFloatMultAcc = "SimdFloatMultAcc"
    SimdFloatMatMultAcc = "SimdFloatMatMultAcc"
    SimdFloatSqrt = "SimdFloatSqrt"
    SimdReduceAdd = "SimdReduceAdd"
    SimdReduceAlu = "SimdReduceAlu"
    SimdReduceCmp = "SimdReduceCmp"
    SimdFloatReduceAdd = "SimdFloatReduceAdd"
    SimdFloatReduceCmp = "SimdFloatReduceCmp"
    SimdExt = "SimdExt"
    SimdFloatExt = "SimdFloatExt"
    SimdConfig = "SimdConfig"
    SimdDotProd = "SimdDotProd"
    SimdAes = "SimdAes"
    SimdAesMix = "SimdAesMix"
    SimdSha1Hash = "SimdSha1Hash"
    SimdSha1Hash2 = "SimdSha1Hash2"
    SimdSha256Hash = "SimdSha256Hash"
    SimdSha256Hash2 = "SimdSha256Hash2"
    SimdShaSigma2 = "SimdShaSigma2"
    SimdShaSigma3 = "SimdShaSigma3"
    SimdSha3 = "SimdSha3"
    SimdSm4e = "SimdSm4e"
    SimdCrc = "SimdCrc"
    SimdBf16Add = "SimdBf16Add"
    SimdBf16Cmp = "SimdBf16Cmp"
    SimdBf16Cvt = "SimdBf16Cvt"
    SimdBf16DotProd = "SimdBf16DotProd"
    SimdBf16MatMultAcc = "SimdBf16MatMultAcc"
    SimdBf16Mult = "SimdBf16Mult"
    SimdBf16MultAcc = "SimdBf16MultAcc"
    Matrix = "Matrix"
    MatrixMov = "MatrixMov"
    MatrixOP = "MatrixOP"
    System = "System"
    SimdPredAlu = "SimdPredAlu"
    MemRead = "MemRead"
    MemWrite = "MemWrite"
    FloatMemRead = "FloatMemRead"
    FloatMemWrite = "FloatMemWrite"
    SimdUnitStrideLoad = "SimdUnitStrideLoad"
    SimdUnitStrideMaskLoad = "SimdUnitStrideMaskLoad"
    SimdUnitStrideSegmentedLoad = "SimdUnitStrideSegmentedLoad"
    SimdStridedLoad = "SimdStridedLoad"
    SimdIndexedLoad = "SimdIndexedLoad"
    SimdUnitStrideFaultOnlyFirstLoad = "SimdUnitStrideFaultOnlyFirstLoad"
    SimdUnitStrideSegmentedFaultOnlyFirstLoad = "SimdUnitStrideSegmentedFaultOnlyFirstLoad"
    SimdWholeRegisterLoad = "SimdWholeRegisterLoad"
    SimdStrideSegmentedLoad = "SimdStrideSegmentedLoad"
    SimdUnitStrideStore = "SimdUnitStrideStore"
    SimdUnitStrideMaskStore = "SimdUnitStrideMaskStore"
    SimdUnitStrideSegmentedStore = "SimdUnitStrideSegmentedStore"
    SimdStridedStore = "SimdStridedStore"
    SimdIndexedStore = "SimdIndexedStore"
    SimdWholeRegisterStore = "SimdWholeRegisterStore"
    SimdStrideSegmentedStore = "SimdStrideSegmentedStore"

    def __str__(self) -> str:
        return self.value


class IntALU(FUDesc):
    opList = [OpDesc(opClass=OpClass.IntAlu, opLat=1)]
    count = 6


class IntMultDiv(FUDesc):
    opList = [
        OpDesc(opClass=OpClass.IntMult, opLat=3),
        OpDesc(opClass=OpClass.IntDiv, opLat=20, pipelined=False),
    ]
    count = 2


class FP_ALU(FUDesc):
    opList = [
        OpDesc(opClass=OpClass.FloatAdd, opLat=2),
        OpDesc(opClass=OpClass.FloatCmp, opLat=2),
        OpDesc(opClass=OpClass.FloatCvt, opLat=2),
        OpDesc(opClass=OpClass.Bf16Cvt, opLat=2),
    ]
    count = 4


class FP_MultDiv(FUDesc):
    opList = [
        OpDesc(opClass=OpClass.FloatMult, opLat=4),
        OpDesc(opClass=OpClass.FloatMultAcc, opLat=5),
        OpDesc(opClass=OpClass.FloatMisc, opLat=3),
        OpDesc(opClass=OpClass.FloatDiv, opLat=12, pipelined=False),
        OpDesc(opClass=OpClass.FloatSqrt, opLat=24, pipelined=False),
    ]
    count = 2


class VecFpuUnit(FUDesc):
    opList = [
        OpDesc(opClass=OpClass.SimdAdd),
        OpDesc(opClass=OpClass.SimdAddAcc),
        OpDesc(opClass=OpClass.SimdAlu),
        OpDesc(opClass=OpClass.SimdCmp),
        OpDesc(opClass=OpClass.SimdCvt),
        OpDesc(opClass=OpClass.SimdMisc),
        OpDesc(opClass=OpClass.SimdMult),
        OpDesc(opClass=OpClass.SimdMultAcc),
        OpDesc(opClass=OpClass.SimdMatMultAcc),
        OpDesc(opClass=OpClass.SimdShift),
        OpDesc(opClass=OpClass.SimdShiftAcc),
        OpDesc(opClass=OpClass.SimdDiv),
        OpDesc(opClass=OpClass.SimdSqrt),
        OpDesc(opClass=OpClass.SimdFloatAdd),
        OpDesc(opClass=OpClass.SimdFloatAlu),
        OpDesc(opClass=OpClass.SimdFloatCmp),
        OpDesc(opClass=OpClass.SimdFloatCvt),
        OpDesc(opClass=OpClass.SimdFloatDiv),
        OpDesc(opClass=OpClass.SimdFloatMisc),
        OpDesc(opClass=OpClass.SimdFloatMult),
        OpDesc(opClass=OpClass.SimdFloatMultAcc),
        OpDesc(opClass=OpClass.SimdFloatMatMultAcc),
        OpDesc(opClass=OpClass.SimdFloatSqrt),
        OpDesc(opClass=OpClass.SimdReduceAdd),
        OpDesc(opClass=OpClass.SimdReduceAlu),
        OpDesc(opClass=OpClass.SimdReduceCmp),
        OpDesc(opClass=OpClass.SimdFloatReduceAdd, opLat=4),
        OpDesc(opClass=OpClass.SimdFloatReduceCmp, opLat=4),
        OpDesc(opClass=OpClass.SimdExt),
        OpDesc(opClass=OpClass.SimdFloatExt),
        OpDesc(opClass=OpClass.SimdConfig),
        OpDesc(opClass=OpClass.SimdDotProd),
        OpDesc(opClass=OpClass.SimdAes),
        OpDesc(opClass=OpClass.SimdAesMix),
        OpDesc(opClass=OpClass.SimdSha1Hash),
        OpDesc(opClass=OpClass.SimdSha1Hash2),
        OpDesc(opClass=OpClass.SimdSha256Hash),
        OpDesc(opClass=OpClass.SimdSha256Hash2),
        OpDesc(opClass=OpClass.SimdShaSigma2),
        OpDesc(opClass=OpClass.SimdShaSigma3),
        OpDesc(opClass=OpClass.SimdSha3),
        OpDesc(opClass=OpClass.SimdSm4e),
        OpDesc(opClass=OpClass.SimdCrc),
        OpDesc(opClass=OpClass.SimdBf16Add),
        OpDesc(opClass=OpClass.SimdBf16Cmp),
        OpDesc(opClass=OpClass.SimdBf16Cvt),
        OpDesc(opClass=OpClass.SimdBf16DotProd),
        OpDesc(opClass=OpClass.SimdBf16MatMultAcc),
        OpDesc(opClass=OpClass.SimdBf16Mult),
        OpDesc(opClass=OpClass.SimdBf16MultAcc),
    ]
    count = 1


class Matrix_Unit(FUDesc):
    opList = [
        OpDesc(opClass=OpClass.Matrix),
        OpDesc(opClass=OpClass.MatrixMov),
        OpDesc(opClass=OpClass.MatrixOP),
    ]
    count = 1


class System_Unit(FUDesc):
    opList = [OpDesc(opClass=OpClass.System)]
    count = 1


class PredALU(FUDesc):
    opList = [OpDesc(opClass=OpClass.SimdPredAlu)]
    count = 1


class ReadPort(FUDesc):
    opList = [
        OpDesc(opClass=OpClass.MemRead),
        OpDesc(opClass=OpClass.FloatMemRead),
        OpDesc(opClass=OpClass.SimdUnitStrideLoad),
        OpDesc(opClass=OpClass.SimdUnitStrideMaskLoad),
        OpDesc(opClass=OpClass.SimdUnitStrideSegmentedLoad),
        OpDesc(opClass=OpClass.SimdStridedLoad),
        OpDesc(opClass=OpClass.SimdIndexedLoad),
        OpDesc(opClass=OpClass.SimdUnitStrideFaultOnlyFirstLoad),
        OpDesc(opClass=OpClass.SimdUnitStrideSegmentedFaultOnlyFirstLoad),
        OpDesc(opClass=OpClass.SimdWholeRegisterLoad),
        OpDesc(opClass=OpClass.SimdStrideSegmentedLoad),
    ]
    count = 0


class WritePort(FUDesc):
    opList = [
        OpDesc(opClass=OpClass.MemWrite),
        OpDesc(opClass=OpClass.FloatMemWrite),
        OpDesc(opClass=OpClass.SimdUnitStrideStore),
        OpDesc(opClass=OpClass.SimdUnitStrideMaskStore),
        OpDesc(opClass=OpClass.SimdUnitStrideSegmentedStore),
        OpDesc(opClass=OpClass.SimdStridedStore),
        OpDesc(opClass=OpClass.SimdIndexedStore),
        OpDesc(opClass=OpClass.SimdWholeRegisterStore),
        OpDesc(opClass=OpClass.SimdStrideSegmentedStore),
    ]
    count = 0


class RdWrPort(FUDesc):
    opList = [
        OpDesc(opClass=OpClass.MemRead),
        OpDesc(opClass=OpClass.MemWrite),
        OpDesc(opClass=OpClass.FloatMemRead),
        OpDesc(opClass=OpClass.FloatMemWrite),
        OpDesc(opClass=OpClass.SimdUnitStrideLoad),
        OpDesc(opClass=OpClass.SimdUnitStrideStore),
        OpDesc(opClass=OpClass.SimdUnitStrideMaskLoad),
        OpDesc(opClass=OpClass.SimdUnitStrideMaskStore),
        OpDesc(opClass=OpClass.SimdUnitStrideSegmentedLoad),
        OpDesc(opClass=OpClass.SimdUnitStrideSegmentedStore),
        OpDesc(opClass=OpClass.SimdStridedLoad),
        OpDesc(opClass=OpClass.SimdStridedStore),
        OpDesc(opClass=OpClass.SimdIndexedLoad),
        OpDesc(opClass=OpClass.SimdIndexedStore),
        OpDesc(opClass=OpClass.SimdUnitStrideFaultOnlyFirstLoad),
        OpDesc(opClass=OpClass.SimdUnitStrideSegmentedFaultOnlyFirstLoad),
        OpDesc(opClass=OpClass.SimdWholeRegisterLoad),
        OpDesc(opClass=OpClass.SimdWholeRegisterStore),
        OpDesc(opClass=OpClass.SimdStrideSegmentedLoad),
        OpDesc(opClass=OpClass.SimdStrideSegmentedStore),
    ]
    count = 4


FU_POOL = [
    IntALU,
    IntMultDiv,
    FP_ALU,
    FP_MultDiv,
    ReadPort,
    VecFpuUnit,
    Matrix_Unit,
    System_Unit,
    PredALU,
    WritePort,
    RdWrPort,
]


def op_lat(unit, op_class):
    for desc in unit.opList:
        if desc.opClass == op_class:
            return desc.opLat
    return unit.opList[0].opLat


class Category(Enum):
    VECTOR_FPU = "vector_fpu"
    VECTOR_REDUCTION = "vector_reduction"
    VECTOR_MEMORY = "vector_memory"
    SCALAR = "scalar"


CATEGORY_BASE_COST = {
    Category.SCALAR: (IntALU, OpClass.IntAlu),
    Category.VECTOR_FPU: (VecFpuUnit, OpClass.SimdFloatAdd),
    Category.VECTOR_REDUCTION: (VecFpuUnit, OpClass.SimdFloatReduceAdd),
    Category.VECTOR_MEMORY: (ReadPort, OpClass.MemRead),
}

FPU_CATEGORIES = (
    Category.VECTOR_FPU,
    Category.VECTOR_REDUCTION,
)


def category_base_cost(category: Category) -> int:
    unit, op_class = CATEGORY_BASE_COST[category]
    return op_lat(unit, op_class)
