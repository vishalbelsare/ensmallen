"""Module offering fast graph processing and graph datasets."""

from .compilation_flags import FLAGS, TARGET_TRIPLE, OS, CPU_ARCH

import cpuinfo

# In some systems cpuinfo is not able to detect some flags no matter what.
# In order to avoid false positives, we remove by default these flags.
collision_flags = {
    'ermsb',
    'xsaveopt',
    'xsavec',
    'xsaves',
    'cmpxchg16b',
    # TODO! the following are very weird to not be detected! Likely needs a dictionary for renormalization!
    'sse3',
    'sse4.2',
    'lzcnt',
    'sse4.1'
}

unavailable_flags = set(FLAGS) - set(cpuinfo.get_cpu_info()["flags"]) - collision_flags

if len(unavailable_flags) > 0:
    raise ValueError(
        (
            "On the current machine, the flags '{}' are not available.\n"
            "This library was compiled assuming that the following instruction "
            "sets are available: {}\n"
            "You can solve this issue by manually compiling ensmallen tailoring"
            " it to your hardware following the guides on our Github repository."
            " Ensmallen has no strict dependancy on any flag or cpu_arch as it "
            "can be compiled for any arch supported by LLVM (Arm, AArch64, Mips,"
            " ...).\n"
            "In particular, the pip pre-compiled versions are aimed to x86_64"
            "Cpus with architecture newer or equal to Haswell (2013)."
            "This is done because we heavily explioit both SSE and AVX "
            "instructions which allows great speed-ups of several core routines"
            "and providing pre-compiled versions for all cpu arches with "
            "all flags combinations for all operating systems would have a "
            "prohibitive cost."
        ).format(unavailable_flags, FLAGS)
    )


from .ensmallen import preprocessing  # pylint: disable=import-error
from .ensmallen import Graph  # pylint: disable=import-error
from .ensmallen import edge_list_utils  # pylint: disable=import-error

# The import of dataset should ALWAYS be under the imports from the compiled bindings
# Because otherwise it generate a Circular import and crash
from . import datasets

__all__ = ["edge_list_utils", "Graph", "preprocessing", "datasets"]
