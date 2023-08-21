import logging, time


def create_log(log_level="info"):
    log = Logger(log_level=log_level)
    log.setup_logging()
    return log


_logging_handler = None


class Logger(object):
    def __init__(self, name="Python_Report", log_level="info"):
        self.name = name
        self.log_level = log_level

    def setup_logging(self):
        levels = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }

        logger = logging.getLogger()
        t0 = time.time()

        class Formatter(logging.Formatter):
            def format(self, record):
                s1 = "[ %09.2f ]: " % (time.time() - t0)
                return s1 + logging.Formatter.format(self, record)

        fmt = Formatter(
            fmt="%(asctime)s %(name)-15s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M ",
        )

        global _logging_handler
        if _logging_handler is None:
            _logging_handler = logging.StreamHandler()
            logger.addHandler(_logging_handler)

        _logging_handler.setFormatter(fmt)
        logger.setLevel(levels[self.log_level])

    def setup_report_logging(self):
        levels = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }
        logging.basicConfig(
            filename=self.name,
            filemode="w",
            level=levels[self.log_level],
            format="%(asctime)s :: %(levelname)s :: %(message)s",
        )

    @staticmethod
    def add(line, level="info"):
        if level == "info":
            logging.info(line)
        if level == "warning":
            logging.warning(line)
        if level == "debug":
            logging.debug(line)

    @staticmethod
    def add_array_statistics(arr, char):
        if arr is not None:
            Logger.add(f"Min of {char}: {arr.min()}")
            Logger.add(f"Max of {char}: {arr.max()}")
            Logger.add(f"Mean of {char}: {arr.mean()}")
            Logger.add(f"Standard deviation of {char}: {arr.std()}")

    @staticmethod
    def close():
        logging.shutdown()
