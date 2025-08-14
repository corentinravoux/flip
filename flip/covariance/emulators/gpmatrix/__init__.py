from flip.utils import create_log

log = create_log()

try:
    import GPy
except:
    log.add(
        "Install GPy to use the gpmatvel emulator",
        level="warning",
    )


_emulator_type = "matrix"
