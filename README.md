# hd-tournament
2024年洪都杯智能空战大赛

## 配置说明
### 依赖
- [TacView](https://www.tacview.net/download/latest/en/)软件提供可视化
- Miniconda3安装ProgramData目录
- 运行测试的时候应该只能用官方提供的Python环境hdbisai（解压[压缩包](https://superboysb-my.sharepoint.cn/:u:/g/personal/admin_superboysb_partner_onmschina_cn/EfA3s4y1CLZPg_--J3d5tOsBT112TyhezIcbC2N6W9JMmw?e=SYPKtG)大小为3G左右，或者是叶神准备封装的docker）
- 鉴于仿真只能windows（本机ip：172.16.0.213），运行训练用的机器可以先用部门服务器(ubuntu@172.16.12.210)

### Windows环境准备（功能：比赛测试+通信）
为了尽量运行方便，需要在Windows机器的hdbisai环境扩充一些与服务器通信相关的库，
```sh
D:\projects\hdbisai\python.exe -m pip install --upgrade pip --proxy="127.0.0.1:10809"
D:\projects\hdbisai\python.exe -m pip install paramiko ray[rllib]==2.27.2 --proxy="127.0.0.1:10809"
```
注意有SSL的问题要先添加系统变量（如：`D:\projects\hdbisai\Library\bin`），代理开全局模式。


### 运行智能体对打的简单Demo
先保证`sim = HDDF2Sim(scen, use_tacview=True, save_replay=True, replay_path="replay.acmi")`之后运行demo
```sh
D:\projects\hdbisai\python.exe demo_raw.py
```
然后进入Tacview，给端口`5555`可以看到实时遥测
[![image.png](https://i.postimg.cc/wvjNDhWX/image.png)](https://postimg.cc/GB69Csnt)

save_replay后得到的新的`replay.acmi`可以得到非实时的回放，那个时候可以拖动进度条，详细用法可以咨询李超

## TODO
- [X] 按理说，`use_tacview=False, save_replay=False`的话应该就是后台交互？看起来应该没问题，step是正常执行的
- [ ] 如果第一个todo没问题的话，把上述过程可以封装在docker内部？(哎，还得是叶神，专业人干专业事，那我不尝试封装这个win+python的依赖了)
- [ ] 为了方便计算，创建了名为 Vec3 的类，该类支持该智能体中使用到的三维坐标的运算（Vec3 类仅作为示例演示，选手需自行设计实现类似功能）
- [ ] attack_move()方法根据当前飞机信息和要追击的敌机信息，计算追击角度，以得到飞机的机动控制参数（此处attack_move()方法仅作示例演示，选手需自行设计实现类似功能）

## 训练Pipeline

### Lowevel
先训练一个single_control的task