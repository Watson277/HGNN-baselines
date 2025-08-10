import os
import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')  # 使用 'a' 模式追加

    def write(self, message):
        self.terminal.write(message)  # 打印到控制台
        self.log.write(message)       # 写入文件

    def flush(self):
        pass  # 为兼容 print 的 flush 参数

if __name__ == "__main__":
    # 获取当前脚本名并构造同名 txt 文件
    py_file = sys.argv[0]
    base_name = os.path.splitext(os.path.basename(py_file))[0]
    txt_filename = base_name + ".txt"

    # 重定向标准输出
    sys.stdout = Logger(txt_filename)

