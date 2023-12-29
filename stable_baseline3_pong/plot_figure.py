from matplotlib.pyplot import plot as plt
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('path_to_log_dir2/0.monitor.csv')
r = []
time = []

first_time = None
with open('path_to_log_dir/0.monitor.csv') as f:
    lines = f.readlines()
    for line in lines[2:]:

        r.append(float(line.split(",")[0]))
        time.append(float(line.split(",")[2]))

        if not first_time and r[-1] == 21:
            first_time = time[-1]


# 绘制奖励曲线
print(f"Reaching 21 for the first time: {first_time}s")
plt.plot(time, r)
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Reward over Time')
plt.show()