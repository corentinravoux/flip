import logging, time


def create_log(log_level="info"):
    """
    The create_log function creates a logger object that can be used to log messages.
    The function takes one argument, the log_level, which is set to &quot;info&quot; by default.
    The function returns a Logger object with the specified logging level.

    Args:
        log_level: Set the logging level

    Returns:
        A logger object

    """
    log = Logger(log_level=log_level)
    log.setup_logging()
    return log


_logging_handler = None


class Logger(object):
    def __init__(self, name="Python_Report", log_level="info"):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the logger and defines a few variables that will be used later.

        Args:
            self: Represent the instance of the class
            name: Set the name of the report
            log_level: Set the log level

        Returns:
            Nothing

        """
        self.name = name
        self.log_level = log_level

    def setup_logging(self):
        """
        The setup_logging function is used to set up the logging module.

        Args:
            self: Refer to the current instance of a class

        Returns:
            Nothing

        """
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
        """
        The setup_report_logging function is used to set up the logging for a report.

        Args:
            self: Represent the instance of the class

        Returns:
            None

        """
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
        """
        The add function takes a line of text and adds it to the log file.
        It also takes an optional level argument, which can be set to &quot;info&quot;, &quot;warning&quot; or &quot;debug&quot;.
        If no level is specified, the default value is used (&quot;info&quot;).


        Args:
            line: Pass the line of text to be logged
            level: Determine the type of log message

        Returns:
            None

        """
        if level == "info":
            logging.info(line)
        if level == "warning":
            logging.warning(line)
        if level == "debug":
            logging.debug(line)

    @staticmethod
    def add_array_statistics(arr, char):
        """
        The add_array_statistics function takes in an array and a character, and prints out the min, max, mean, and standard deviation of that array.

        Args:
            arr: Store the array that is passed to the function
            char: Specify which array is being used

        Returns:
            The minimum, maximum, mean and standard deviation of the array

        """
        if arr is not None:
            Logger.add(f"Min of {char}: {arr.min()}")
            Logger.add(f"Max of {char}: {arr.max()}")
            Logger.add(f"Mean of {char}: {arr.mean()}")
            Logger.add(f"Standard deviation of {char}: {arr.std()}")

    @staticmethod
    def close():
        """
        The close function shuts down the logging module.


        Args:

        Returns:
            The return value of the logging

        """
        logging.shutdown()
