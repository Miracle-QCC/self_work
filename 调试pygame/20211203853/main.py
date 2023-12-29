import os
import sys
import time
try:
    import pygame
except ModuleNotFoundError:
    print('正在安装pygame，请稍等...')
    os.system('pip install pygame') # 安装pygame模块
import tools
__MAJOR,__MINOR,__MICRO = sys.version_info[0],sys.version_info[1],sys.version_info[2]
if __MAJOR < 3:
    print('Python版本号过低，当前版本为%d.%d.%d，请重装Python解释器'%(__MAJOR,__MINOR,__MICRO))
    time.sleep(2)
    exit()
if __name__ == '__main__':
    paint = tools.Paint()
    try:
        paint.run()# 启动主窗口
    except Exception as e:
        print(e)