"""Opt-in file logging that leaves the caller's root logger unchanged."""
import logging
import os

class Logger():
    def __init__(self, root):
        # Set path
        self.log_root_path = root 
        self.category = ['factor_proposal', 'factor_annotation', 'causal_graph', 'feedbacks']

        # init log files
        if not os.path.exists(self.log_root_path):
            os.makedirs(self.log_root_path)

        # 注意：构造器不触碰 root logger，也不劫持 sys.stdout/stderr；
        # 库运行不改变调用者的全局日志环境。

    def straight_write(self, cate, content, mode='a'):
        assert cate in self.category
        with open(os.path.join(self.log_root_path, cate+'.log'), mode) as file:
            file.write('\n'+content+'\n')

    def attach_root_file_logging(self, filename="experiment.log", level=logging.DEBUG):
        """
        显式可选（opt-in）：把 root logger 的输出镜像到本日志目录下的文件。

        该方法只能由实验脚本/facade 在明确需要时调用（构造器与库流程都不会
        调用它）。调用后会修改全局 logging 状态：
        - 向 root logger 追加一个 FileHandler（重复调用不会重复添加）；
        - 调低第三方库（httpx/httpcore/openai._base_client）的日志级别；
        - root logger 自身的 level 不改动：若希望捕获 INFO 及以下日志，
          请由调用者自行设置，例如 logging.getLogger().setLevel(logging.INFO)。
        """
        log_path = os.path.join(self.log_root_path, filename)

        root = logging.getLogger()
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    if os.path.samefile(handler.baseFilename, log_path):
                        return root
                except (FileNotFoundError, AttributeError):
                    continue

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        for noisy in (
            "httpx",
            "httpcore",
            "httpcore.http11",
            "httpcore.connection",
            "openai._base_client",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        return root

