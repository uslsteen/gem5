from m5.objects.Device import BasicPioDevice
from m5.params import *
from m5.proxy import *


class SystolicArray(BasicPioDevice):
    type = "SystolicArray"
    cxx_header = "dev/systolic_array.hh"

    pio_size = Param.Addr(
        40, "MMR size (flags byte + 8B-aligned a/b/c base and k_len)"
    )
    local = RequestPort(
        "Master port to the scratchpad side (functional SPM traffic)"
    )

    clock_period = Param.Int(
        1, "Clock period in ns (1 GHz domain)"
    )
