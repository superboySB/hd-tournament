# hd-tournament
2024年洪都杯智能空战大赛

## RoadMap
需要与超哥同步
- [X] 按理说，`use_tacview=False, save_replay=False`的话应该就是后台交互？看起来应该没问题，step是正常执行的
- [X] attack_move()方法根据当前飞机信息和要追击的敌机信息，计算追击角度，以得到飞机的机动控制参数（此处attack_move()方法仅作示例演示,并且效果也是真的差，选手需自行设计实现类似功能）目前李超应该是已经实现了简单的基于PID做跟随的DEMO，还需要输入输出完善到能作为一个low level policy。
- [X] 为了方便计算，创建了名为 Vec3 的类，该类支持该智能体中使用到的三维坐标的运算（Vec3 类仅作为示例演示，不过看起来逻辑是对的，选手需自行设计实现类似功能）

需要与叶神同步
- [X] 把上述过程可以封装在docker内部？(哎，还得是叶神，专业人干专业事，那我不尝试封装这个win+python的依赖了)
- [ ] 目前感觉Rllib这个联盟训练机制好像有一些隐藏的问题（跑openspiel会奇怪崩溃，报错`KeyError: ('action_dist_inputs',)`，不知道是不是C++层面的问题），需要细细去看，然后洪都的完整适配应该也是时间问题。目前这部分直接把问题复现的机器交付给叶神来亲手调试了。

## 配置说明
### 依赖
- [TacView](https://www.tacview.net/download/latest/en/)软件提供可视化，专业版可以提供实时单步调试，免费版可以使用回放
- 运行测试的时候应该推荐用官方提供的Python环境hdbisai（解压[压缩包](https://superboysb-my.sharepoint.cn/:u:/g/personal/admin_superboysb_partner_onmschina_cn/EfA3s4y1CLZPg_--J3d5tOsBT112TyhezIcbC2N6W9JMmw?e=SYPKtG)大小为3G左右，然后因为它是一个conda环境，需要放到`C:\ProgramData\Miniconda3\envs\`这个目录下骗一下它有conda。
- 鉴于仿真只能windows（本机ip：172.16.0.213），运行训练用的机器可以先用部门服务器(ubuntu@172.16.12.210)或算力池

### Windows环境准备（功能：单机调试+比赛测试）
首先在系统变量的Path添加三个conda环境所需的路径（解决SSL问题）
```sh
C:\ProgramData\Miniconda3\envs\hdbisai\Scripts
C:\ProgramData\Miniconda3\envs\hdbisai\Library\bin
C:\ProgramData\Miniconda3\envs\hdbisai
```
现在的系统本机python命令默认为比赛方环境了。为了尽量单机调试方便，需要在Windows机器的原有hdbisai环境扩充一些与ray训练、服务器通信相关的库。(下面暂时是为了开发方便，后续建议还是要考虑官方测试阶段，可能不允许装依赖的情况)
```sh
# 如果需要代理要用全局模式
C:\ProgramData\Miniconda3\envs\hdbisai\python.exe -m pip install --upgrade pip --proxy=127.0.0.1:10809 

# Windows基本很难装Open Spiel，但是在linux里面可以一步pip到位
# C:\ProgramData\Miniconda3\envs\hdbisai\python.exe -m pip install paramiko open_spiel --proxy=127.0.0.1:10809

# 目测py3.7能使用的最新版本也就是2.7.2
C:\ProgramData\Miniconda3\envs\hdbisai\python.exe -m pip install --upgrade ray[rllib] --proxy=127.0.0.1:10809  
```

### 运行智能体对打的简单Demo
先保证`sim = HDDF2Sim(scen, use_tacview=True, save_replay=True, replay_path="replay.acmi")`之后运行demo
```sh
python demo_raw.py
```
然后进入Tacview，给端口`5555`可以看到实时遥测
[![image.png](https://i.postimg.cc/wvjNDhWX/image.png)](https://postimg.cc/GB69Csnt)

save_replay后得到的新的`replay.acmi`可以得到非实时的回放，那个时候可以拖动进度条，详细用法可以咨询李超

## 测试李超demo
```sh
python demo_chao.py
```

## 训练Pipeline
环境接口定义与单步调试
```sh
python env_wrappers.py
```
基于上述接口实现，先跑通一个分布式自博弈+联盟训练
```sh
python self_play_on_windows.py
```
上述demo还存在一些运行问题、环境设计等问题，还是要深挖rllib提供的[主要参考](https://github.com/ray-project/ray/blob/ray-2.7.1/rllib/examples/self_play_league_based_with_open_spiel.py)去进一步研究,不过研究open_spiel这个demo的时候建议用linux，然后pip装一下就可以，比较方便。
```sh
python open_spiel_env_wrappers.py
python open_spiel_self_play_demo.py
```

## 关于Tacview的快捷键用法
[![image.png](https://i.postimg.cc/mg1C4h6G/image.png)](https://postimg.cc/pmxpYVH0)