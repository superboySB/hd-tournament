# hd-tournament
2024年洪都杯智能空战大赛

## 依赖
- [TacView](https://www.tacview.net/download/latest/en/)软件提供可视化
- 官方提供的Python环境（或者是叶神准备封装的docker）

## 使用指南
先保证`sim = HDDF2Sim(scen, use_tacview=True, save_replay=True, replay_path="replay.acmi")`之后运行demo
```sh
D:\projects\hdbisai\python.exe demo.py
```
然后进入Tacview，给端口`5555`可以看到实时遥测
[![image.png](https://i.postimg.cc/wvjNDhWX/image.png)](https://postimg.cc/GB69Csnt)

save_replay后得到的新的`replay.acmi`可以得到非实时的回放，那个时候可以拖动进度条，详细用法可以咨询李超

## TODO
- [X] 按理说，`use_tacview=False, save_replay=False`的话应该就是后台交互？看起来应该没问题，step是正常执行的
- [ ] 如果第一个todo没问题的话，把上述过程可以封装在docker内部？