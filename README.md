# hd-tournament
2024年洪都杯智能空战大赛

## 配置说明
### 依赖
- [TacView](https://www.tacview.net/download/latest/en/)软件提供可视化，专业版可以提供实时单步调试，免费版可以使用回放
- 运行测试的时候应该推荐用官方提供的Python环境hdbisai（解压大小为3G左右），然后因为它是一个conda环境，需要放到`C:\ProgramData\Miniconda3\envs\`这个目录下骗一下它有conda。
- 鉴于仿真只能windows（本机ip：172.16.0.213），运行训练用的机器可以先用部门服务器(ubuntu@172.16.12.210)或算力池

### Windows环境准备（功能：单机调试+比赛测试）
首先在系统变量的Path添加三个conda环境所需的路径（解决SSL问题）
```sh
C:\ProgramData\Miniconda3\envs\hdbisai\Scripts
C:\ProgramData\Miniconda3\envs\hdbisai\Library\bin
C:\ProgramData\Miniconda3\envs\hdbisai
```
现在的系统本机python命令默认为比赛方环境了。为了尽量单机调试方便，需要在Windows机器的原有hdbisai环境扩充一些与ray训练、服务器通信相关的库。(下面暂时是为了开发方便，后续在官方测试阶段，暂时不允许装依赖，应该默认就是要用`requirements_raw.txt`)
```sh
# 如果需要代理要用全局模式
C:\ProgramData\Miniconda3\envs\hdbisai\python.exe -m pip install --upgrade pip --proxy=127.0.0.1:10809 

# 目测py3.7能使用的最新版本也就是2.7.2
C:\ProgramData\Miniconda3\envs\hdbisai\python.exe -m pip install -r requirements.txt --proxy=127.0.0.1:10809  
```

### 运行智能体对打的简单Demo
先保证`sim = HDDF2Sim(scen, use_tacview=True, save_replay=True, replay_path="replay.acmi")`之后运行demo
```sh
python demo_raw.py
```
然后进入Tacview，给端口`5555`可以看到实时遥测
[![image.png](https://i.postimg.cc/wvjNDhWX/image.png)](https://postimg.cc/GB69Csnt)

save_replay后得到的新的`replay.acmi`可以得到非实时的回放，那个时候可以拖动进度条，详细用法可以咨询李超

## 提交代码记录
提交代码要求编译pyd，并且不要留有任何打印的调试信息。

### 最新方案
初赛已过，发现当前规则还是存在一些弱点，需要继续完善
```sh
python demo_jiehu_final.py
```

## Tips
### 编译pyd
搞一个同名的my_agent_demo.pyx文件，然后我们尝试在最外面编辑`setup.py`进行封装，运行下面的代码。
```sh
C:\ProgramData\Miniconda3\envs\hdbisai\python.exe -m pip install cython --proxy=127.0.0.1:10809  

python setup.py build_ext --inplace
```

### 关于Tacview的快捷键用法
[![image.png](https://i.postimg.cc/mg1C4h6G/image.png)](https://postimg.cc/pmxpYVH0)