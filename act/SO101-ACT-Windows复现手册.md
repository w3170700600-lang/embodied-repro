# SO-101 + LeRobot ACT 复现手册（Windows 原生版）

> 原教程：<https://www.cherish.wang/wiki/2025-12-01-lerobot101-act-suan-fa-fu-xian>（9 章，Linux 环境）  
> 本手册把它改成 **Windows 原生 + COM 口** 的可执行版本，并标注了教程里过时/笔误的地方。  
> 环境：`D:\lerobot-venv`（Python 3.12.0），lerobot 版本见文末。

---

## 0. 先看这三条差异（照抄教程会踩坑）

| 教程写的                                           | 实际应该是                                    | 原因                                 |
| ---------------------------------------------- | ---------------------------------------- | ---------------------------------- |
| `lerobot-find-out`                             | `lerobot-find-port`                      | 教程笔误                               |
| `sudo chmod 666 /dev/ttyACM0`                  | **整段跳过**                                 | Windows 没有设备节点权限概念，COM 口即插即用       |
| `lerobot-record --policy.path=...`（第 8 章部署）    | `lerobot-rollout --policy.path=...`      | 0.6.x 已把「用策略驱动机器人」拆成独立命令           |
| `--robot.port=/dev/ttyACM1`                    | `--robot.port=COM5`（你自己的口）               | Windows 串口命名                       |
| `index_or_path: /dev/video4`                   | `index_or_path: 1`（整数索引）                 | Windows 走 OpenCV 索引，没有 /dev/video* |
| `--dataset.repo_id="Embodied-AI-6/so101_test"` | `--dataset.repo_id=<你的HF用户名>/so101_test` | 那是原作者的仓库，你推不上去                     |
| `HF_USER=$(hf auth whoami \| head -n 1)`       | `HF_USER=$(hf auth whoami -q)`           | 见第 7 节，这条拿到的是 `✓ Logged in` 不是用户名  |

### 教程的覆盖缺口（图里才有，正文没写）

原作者第 1 章那张「复现大体流程」图实际有 10 步，正文只覆盖了其中 7 步：

| 图中步骤                      | 正文有没有                                                   |
| ------------------------- | ------------------------------------------------------- |
| 购买材料                      | 无                                                       |
| 1. 配置 lerobot 环境          | 有（第 2 章）                                                |
| 2. 配置舵机驱动                 | 只在图里                                                    |
| **3. 组装机械臂**              | **完全没有** —— 第 2 章直接跳到串口，第 3 章就是写 ID。组装本体只能看官方 SO-101 文档 |
| 4. 校准机械臂                  | 有（第 3 章）                                                |
| 5-7. 遥操 / 带相机 / 采数据       | 有（第 4、6 章）                                              |
| 本地有显卡？→ 本地训练 / 租 4090 云训练 | 有（第 7 章只给了命令）                                           |
| 8/9. 上传数据集 → 训练 → 下载模型    | 有                                                       |
| 10. 本地评估测试                | 有（第 8 章）                                                |

> 所以「组装」这块你得另找官方文档（<https://huggingface.co/docs/lerobot/so101> 有逐关节装配与减速比表），别指望这篇补齐。

---

## 1. 本机环境现状（2026-11-02 实测）

| 项      | 状态                                                            |
| ------ | ------------------------------------------------------------- |
| GPU    | RTX 3050 Laptop **4GB**，驱动 512.28 / CUDA 11.6（老，cu126 轮子跑不起来） |
| Python | 3.12.0 @ `D:\Python\Python3.12.0`                             |
| ffmpeg | 已装（N-127252）                                                  |
| git    | 已装 2.55.0                                                     |
| 磁盘     | C: 14.5G / **D: 53.6G** → 所有大件放 D 盘                           |
| venv   | `D:\lerobot-venv`                                             |

**已装好并实测通过的部分：**

| 组件         | 版本                            | 实测                                                        |
| ---------- | ----------------------------- | --------------------------------------------------------- |
| lerobot    | **0.6.1**                     | 18 个 `lerobot-*` 命令入口全部可用                                 |
| torch      | 2.11.0+cu126                  | `torch.cuda.is_available()` → **False**（驱动 11060 太老，符合预期） |
| torchcodec | 0.11.1                        | 加载失败（ffmpeg 是静态版无 DLL）→ 自动降级 pyav，**不影响录制**               |
| av (PyAV)  | 15.1.0                        | 视频编解码实际走这个                                                |
| 相机         | `lerobot-find-cameras opencv` | ✅ 已识破 Camera #0（DSHOW 后端，640×480），截帧验证为真实画面               |

> 结论：本机负责 **校准 / 遥操 / 录制 / 回放**（纯 CPU + 串口 + 相机就够）。  
> 训练不在这台机器上做（显存 4GB 且驱动过老），等用到 A4000 时再单独配环境。
>
> ⚠️ `lerobot-train` 会打一堆 `Could not load libtorchcodec` 的 WARNING——**属于正常现象**。  
> 编码端在 0.6.1 里写死了 `video_backend="pyav"`（`configs/video.py:97`），解码端也会自动降级，不用管。

---

## 2. 激活环境

CMD / PowerShell：

```bat
D:\lerobot-venv\Scripts\activate
```

Git Bash：

```bash
source /d/lerobot-venv/Scripts/activate
```

验证：

```bash
python -V                 # 3.12.0
lerobot-find-port --help  # 有输出即 CLI 可用
```

---

## 3. 接硬件 & 找到 COM 口（对应教程 2.2）

1. **先只插一根臂**（从臂优先），把 USB 经拓展坞接到 MotorBus。
2. 看设备管理器 → 「端口 (COM 和 LPT)」是否出现 `USB-SERIAL CH340 (COM5)` 这类条目。

用命令列一下：

```powershell
Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'COM\d+' } | Select-Object Name
```

或者用 LeRobot 自带：

```bash
lerobot-find-port
```

**分辨主臂/从臂**：拔掉一根，看哪个 COM 消失；插回来再拔另一根。记成：

- 从臂（follower）→ 例如 `COM5`
- 主臂（leader）→ 例如 `COM4`

> 如果插上根本没 COM 口出现 → 缺 USB 转串口驱动。Waveshare 舵机控制板实测是 **CH343**（`VID:PID=1A86:55D3`），Win10/11 基本自带驱动，不用装 CH341SER。

### 3.1 接不通时的排查顺序（2026-11-03 实测走完一遍）

板子是 **Waveshare Bus Servo Adapter (A)**，42×33mm，底座上。三个坑按概率排：

**① DC 动力电没接（最常见的坑）**

`PWR` 红灯**只反映 USB 那 5V**，不代表舵机有电。舵机的 `V` 脚是直接从 DC 输入取的。

> **判据：把 USB 拔掉，看 `PWR` 还亮不亮。**
> 灭 → 板上只有 USB 5V，DC 没通 → 舵机完全没电 → 总线一个字节都不回
> 还亮 → DC 到位了

要求：**9~12.6V，≥3A（6 个舵机建议 5A）**，DC5521 内正外负，插到底。

**② 跳线帽位置错**

板上丝印 `A | UART - SERVO` / `B | USB - SERVO`。**接电脑 USB 必须放 B 位**（官方文档原文 "the jumper cap must be in the B position"）。放 A 位时总路线被接到左侧 `TX RX GND` 的 UART 排针上，症状是**串口能开、但总线全静默**。

**③ 总线没插 / 插错口**

板子左侧有两个 `D V G` 3-pin 口，接任意一个都行；确认插到底。

### 3.2 三个诊断脚本（都在 `D:\lerobot-work\`）

| 脚本 | 用途 |
|---|---|
| `check_motors.py COM4` | 扫全部波特率，列 ID 并和 1–6 对比 |
| `probe_ids.py COM4` | 逐 ID × proto 0/1 × 4 档波特率，用于广播 ping 失效时 |
| `raw_probe.py COM4` | **裸字节嗅探**：收到乱码=波特率错；零字节=物理层不通 |

`raw_probe.py` 是关键——它把"软件配置错"和"硬件没通电"两种完全不同的故障分开了，别跳过。

---

## 4. 写舵机 ID（对应教程 3.1）

### 4.1 这个命令到底在干什么（源码级）

`lerobot-setup-motors` → `SO101Follower.setup_motors()` → 对 `reversed(self.bus.motors)` 循环 6 次 → 每次 `FeetechMotorsBus.setup_motor()`。

因为字典顺序是 `shoulder_pan…gripper`，**反转后就是从末端往底座走**：

| 第几轮 | 电机名             | 目标 ID |
| --- | --------------- | ----- |
| 1   | `gripper`       | 6     |
| 2   | `wrist_roll`    | 5     |
| 3   | `wrist_flex`    | 4     |
| 4   | `elbow_flex`    | 3     |
| 5   | `shoulder_lift` | 2     |
| 6   | `shoulder_pan`  | 1     |

每轮内部五步（源码 `motors_bus.py:589` + `feetech.py:165`）：

1. 打印 `Connect the controller board to the '<motor>' motor only and press enter.` —— **此时还没建立任何电机通信**，串口号写错也是你回车后才报错。
2. 按 `1_000_000 → 500_000 → 250_000 → 128_000 → 115_200 → 57_600 → 38_400 → 19_200` 逐档 `broadcast_ping`，找到第一个有响应的就停。
3. 校验型号号，STS3215 必须是 **777**，否则抛 `RuntimeError`。
4. `Torque_Enable=0` + `Lock=0`（关闭力矩、解锁 EEPROM，否则改不了 ID）。
5. 写 ID，再**用新 ID** 写 `Baud_Rate`（默认 `1_000_000`）。第二次写能成功，等价于证明新 ID 已生效。

成功后打印：`'gripper' motor id set to 6`

> 没有任何回读值比对，但第 5 步的第二笔写入发往**新 ID**，能写通就说明 ID 真的改了。所以看到 `id set to N` 可以信。

### 4.2 ⚠️ 最大的坑：多台电机同时接着会静默写错

`broadcast_ping()` 返回的是**总线上所有**响应电机的 `{id: 型号}` 字典，而 `_find_single_motor` 用的是：

```python
found_id, found_model = next(iter(id_model.items()))   # 直接取第一台，不检查总数
```

也就是说：**同时接 2 台以上时它不会报错，而是随便挑一台改掉 ID**，然后进入下一轮再挑一台。后果是两台电机拿到同一个 ID，全总线从此混乱。

更麻烦的是写 ID 的指令是按 ID 寻址的：如果两台电机本来就都是默认 ID `1`，那一条「1→6」的写命令会**同时命中两台**，两台一起变成 6。

**所以顺序必须是**：插一个 → 写一个 → 拔掉 → 换下一个。

### 4.3 标准操作流程（照这个顺序做）

1. 控制板 USB 插电脑，确认 COM 口在（Waveshare 板子两个跳线拨到 **B** 通道）。
2. 只把 `gripper` 那个电机接到板上，其余全部断开。
3. 跑命令 → 看到提示 → 回车 → 等 `'gripper' motor id set to 6`。
4. 拔掉 gripper，接 `wrist_roll` → 回车 → 等 `id set to 5`。
5. 依次做完 4→3→2→1，**顺序不能跳**，因为脚本是按固定顺序问的。
6. 6 个都写完后，按链路串起来（shoulder_pan 起 → gripper 止）。
7. 用 `check_motors.py` 验一遍再进校准。

命令：

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=COM5
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=COM4
```

主臂用的是同一套 STS3215，ID 表也一样是 1–6，流程完全相同，只是参数前缀换成 `--teleop.*`。

> 想换默认波特率加 `--robot.baudrate=500000`（默认 1000000）。除非你明确知道为什么要改，别动。

### 4.4 写完之后验证

```bash
python D:\lerobot-work\check_motors.py COM5
```

它会扫全部波特率、列出真实 ID，并和 1–6 对比，报缺失/多余/数量不对。输出 `[OK]` 才算过。

### 4.5 写错了怎么救

| 症状                                                   | 原因                  | 修法                           |
| ---------------------------------------------------- | ------------------- | ---------------------------- |
| `Motor 'xxx' (model 'sts3215') was not found`        | 电机没上电 / 线序反 / 没接到板上 | 查线、查供电，重跑该轮                  |
| `model number 'xxx' different than the one expected` | 接错型号（不是 STS3215）    | 换回正确电机                       |
| 多个 ID 变成同一个                                          | 一轮里接了多台电机           | **物理隔离**：一次只接一台，按想要的名字重跑对应轮次 |
| 总线上只剩一两个 ID 响应                                       | 已有 ID 冲突，总线互相顶      | 一台一台单独接，逐个重设成 1–6            |

> ID 冲突没有「一键修复」，只能一台一台单独接上重写。这也是为什么第 2 步的纪律比速度重要。

### 4.6 完成后

校准（下一节）时会写 `Homing_Offset / Min_Position_Limit / Max_Position_Limit`，这些是**另一套 EEPROM 参数**，和 ID 无关，ID 不会因为校准被改掉。

---

## 5. 校准（对应教程 3.2）

**先把每个关节摆到行程中间位置**，回车后按提示把每个关节推满全程。

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=my_leader
```

校准文件落在 `C:\Users\<你>\.cache\huggingface\lerobot\calibration\`。  
**`--robot.id` / `--teleop.id` 这之后每次都要一字不差地带上**，否则会重新要求校准。

---

## 6. 遥操 + 相机（对应教程 4）

先单臂空跑，确认关节方向和限位正常：

```bash
lerobot-teleoperate ^
  --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower ^
  --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=my_leader
```

（上面是 CMD 的换行符 `^`；PowerShell 用反引号 `` ` ``，Git Bash 用 `\`）

再找相机索引：

```bash
lerobot-find-cameras opencv
```

> **注意 0.6.1 的行为变了**：它**不开预览窗口**，而是把抓到的帧存成 PNG，打印 `Saved image: outputs\captured_images\opencv_0.png`，最后列出 `--- Detected Cameras ---`。  
> 所以判断「相机能不能用」要看那张 PNG，不是看有没有弹窗。本机实测：内置相机 = **索引 0**（DSHOW 后端，640×480 YUY2）。

会列出每个索引的截图。假设外接相机是 `1`、内置是 `0`，带相机遥操：

```bash
lerobot-teleoperate ^
  --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower ^
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" ^
  --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=my_leader
```

相机参数含义：`front` 是相机名（进数据集列名）、`index_or_path` 是 OpenCV 索引、`width/height/fps` 建议先 640×480@30。

> `--display_data=true` 会额外拉起 rerun 可视化窗口。**第一次调链路时先别开**，少一个变量少一个坑。

---

## 7. HuggingFace 认证与建仓库（对应教程 5）

```bash
pip install -U "huggingface_hub[cli]"
setx HF_TOKEN "hf_xxxxxxxx"        # CMD；PowerShell 用 $env:HF_TOKEN="hf_xxx"
hf auth login --token %HF_TOKEN% --add-to-git-credential
hf auth whoami
```

### ⚠️ 教程里取用户名的写法是坏的，别照抄

教程写：

```bash
HF_USER=$(hf auth whoami | head -n 1)     # 错
```

在 huggingface_hub 1.x 上 `hf auth whoami` 打印的是**三行**（`cli/_output.py:173`）：

```
✓ Logged in
  user: <你的用户名>
  orgs: ...
```

`head -n 1` 拿到的是 `✓ Logged in`，不是用户名。原作者那张截图里 `echo $HF_USER` 输出成了 `user: Embodied-AI-6`，说明他那版拿到的是带前缀的第二行——**两种都是错的**，后面 `${HF_USER}/xxx` 拼出来的 repo id 会直接非法。

正确写法（quiet 模式只输出第一个值，也就是用户名）：

```bash
HF_USER=$(hf auth whoami -q)
echo "$HF_USER"
```

拿不到就 `hf auth whoami --format json` 看一眼键名，或者干脆手填用户名，别在这上面耗。

> 顺带：`--add-to-git-credential` 在没配 credential helper 的机器上会警告 `Cannot authenticate through git-credential as no helper is defined`（原作者的截图里就有）。不影响 token 保存，但要推 git 的话先跑一次 `git config --global credential.helper store`。

然后在网页上建两个 **public** 仓库（右上角头像 → **New Dataset** / **New Model**）：

- Dataset：`<你的用户名>/so101_test`
- Model：`<你的用户名>/lerobot_new`

> 如果你想**完全本地跑、不碰 HF**，可以全程加 `--dataset.push_to_hub=false` 和 `--dataset.root=D:/lerobot-data/so101_test`，训练时用 `--dataset.root` 指本地。但注意：**教程第 8 章的 `--policy.path` 若填 HF repo id 就需要网络**，本地就填本地 checkpoint 目录。

---

## 8. 录制 / 重录 / 回放（对应教程 6）

### 8.1 录 10 条

```bash
lerobot-record ^
  --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower ^
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" ^
  --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=my_leader ^
  --dataset.repo_id=<你的用户名>/so101_test ^
  --dataset.root=D:/lerobot-data/so101_test ^
  --dataset.num_episodes=10 ^
  --dataset.single_task="Grab the green cube" ^
  --dataset.episode_time_s=30 ^
  --dataset.reset_time_s=15 ^
  --dataset.push_to_hub=false ^
  --dataset.streaming_encoding=true ^
  --dataset.encoder_threads=2
```

要点（都已用 `--help` 核对过真实默认值）：

- `--dataset.num_episodes` 默认 **50**，是**本次要录的条数**，不是总数。
- `--dataset.episode_time_s` 默认 **60**，`--dataset.reset_time_s` 默认 **60**。第一次调链路建议压到 30/15，不然一条没录完就烦了。
- `--dataset.push_to_hub` 默认 **True**！没登录 HF 会直接报错——先本地跑就显式写 `false`。
- `--dataset.root` 建议固定下来，落盘可控，也方便后面 `--resume`。
- 录制中的按键（结束本条 / 丢弃重录 / 退出）以它跑起来打印的提示为准。

### 8.2 追加录制（对应教程 6.2 的「重录前清理」）

教程教你 `rm -rf` 缓存目录，Windows 上不要这么干（容易删错）。正确姿势：

```bash
lerobot-record ... --resume=true --dataset.num_episodes=5 --dataset.root=D:/lerobot-data/so101_test
```

即：**加 `--resume=true` + `num_episodes` 填新增条数**，数据会追加，不用删任何东西。

### 8.3 回放

```bash
lerobot-replay ^
  --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower ^
  --dataset.repo_id=<你的用户名>/so101_test ^
  --dataset.root=D:/lerobot-data/so101_test ^
  --dataset.episode=0
```

---

## 9. 训练 ACT（对应教程 7）

### 9.1 先跑冒烟测试（已在本机实测通过 ✅）

真机采数据之前，先用公开数据集把训练链路走一遍。用的是原作者名下的  
`Embodied-AI-6/blue_so101_color_blocks_10ep_v1_20260526_175604`（10 条真机数据，110 MB，`codebase_version=v3.0`，与本地 lerobot 完全对齐）。

```bash
hf download Embodied-AI-6/blue_so101_color_blocks_10ep_v1_20260526_175604 ^
  --repo-type dataset --local-dir D:/lerobot-data/blue_so101_10ep

lerobot-train ^
  --dataset.repo_id=Embodied-AI-6/blue_so101_color_blocks_10ep_v1_20260526_175604 ^
  --dataset.root=D:/lerobot-data/blue_so101_10ep ^
  --policy.type=act ^
  --policy.device=cpu ^
  --policy.push_to_hub=false ^
  --output_dir=D:/lerobot-work/outputs/train/act_smoke ^
  --job_name=act_smoke ^
  --steps=20 --batch_size=2 --num_workers=0 ^
  --log_freq=5 --save_freq=1000 --wandb.enable=false
```


**实测结果（2026-11-03，本机 CPU）**：

| 项                  | 数值                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------- |
| 数据集                | 10 episodes / 6534 帧 / 30fps / 1 个 front 相机                                         |
| ACT 参数量            | **51,597,190（52M）**                                                                 |
| 速度                 | **1.6 秒/步**（CPU，batch=2）                                                            |
| loss               | 59.32 → 28.38 → 18.86 → **14.59**（20 步，明显在降 ✅）                                      |
| l1_loss / kld_loss | 0.89→0.70 / 5.84→1.37                                                               |
| 产物                 | `checkpoints/000020/pretrained_model/model.safetensors` = **206,699,736 字节（207MB）** |

> 207MB 正好等于 52M 参数 × 4 字节，和原作者图里那个 207MB 的 `model.safetensors` 完全吻合——说明配置没跑偏。

### 9.2 ⚠️ 必须加 `--policy.push_to_hub=false`

不加会直接报错退出：

```
ValueError: 'repo_id' argument missing. Please specify it to push the model to the hub.
```

原因：`configs/policies.py:70` 里 `push_to_hub: bool = True` 是**默认开启**的，  
`configs/train.py:289` 的校验因此要求必须提供 `repo_id`。本地训练不需要推 Hub，显式关掉即可。

### 9.3 正式训练命令（换到有显卡的机器）

```bash
lerobot-train ^
  --dataset.repo_id=<你的用户名>/so101_test ^
  --policy.type=act ^
  --policy.device=cuda ^
  --policy.push_to_hub=false ^
  --output_dir=outputs/train/act_so101_test ^
  --job_name=act_so101_test ^
  --wandb.enable=false
```

真正开训时（A4000 那台）要注意：

- ACT 默认 `batch_size=8`，A4000 16G 很宽裕；显存紧就 `--batch_size=4 --grad_accum_steps=2`（保持等效 batch）。
- 数据少（<50 条）时 `--policy.n_action_steps` / chunk 大小对效果影响很大，别急着调，先看 loss 曲线。
- 训练完权重在 `outputs/train/act_so101_test/checkpoints/last/pretrained_model/`。

**训练产物长什么样**（原作者推到 Hub 的 `lerobot_new` 仓库，207MB）：

| 文件                                                       | 说明                                                       |
| -------------------------------------------------------- | -------------------------------------------------------- |
| `model.safetensors`                                      | **207 MB** —— ACT 在这套 6 关节数据上的实际体积，别被「大模型」吓到             |
| `config.json`                                            | 策略结构配置（chunk 大小、backbone 等）                              |
| `train_config.json`                                      | 训练超参，`lerobot-train --config_path=... --resume=true` 就靠它 |
| `policy_preprocessor.json` / `policy_postprocessor.json` | 归一化管线配置                                                  |
| `policy_postprocessor_step_0_unnormalize...`             | 具体的归一化统计量                                                |

> 也就是说：**同步一个 ACT 模型只要 200 多 MB**，拷 U 盘都行，完全不必依赖 Hub。

---

## 10. 用训练好的策略驱动机械臂（对应教程 8，**API 已变**）

教程写的是 `lerobot-record --policy.path=...`，0.6.x 改成：

```bash
lerobot-rollout ^
  --strategy.type=base ^
  --policy.path=<你的用户名>/lerobot_new ^
  --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower ^
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" ^
  --task="Grab the green cube" ^
  --duration=60
```

- `--policy.path` 可以是 HF repo id，也可以是本地 `.../pretrained_model` 目录。
- 想同时录下策略执行过程做 DAgger 数据，用 `--strategy.type=dagger`（新版才有）。
- 顺带一提：`--policy.type` 的枚举里已经有 **`lingbot_va`**，跟你那条线是通的，后面可以留意。

---

## 11. Windows 专属坑清单

1. **COM 口没出现** → 装 CH340/CH343 驱动；换 USB 口；别用无供电 Hub。
2. **COM 号 > 9**（COM10+）——pySerial 一般没问题，但个别老工具要写 `\\.\COM10`。
3. **相机索引漂移**：插拔顺序变了索引就可能变，每次录数据前先 `lerobot-find-cameras opencv` 确认。
4. **分辨率不支持**：OpenCV 会静默改成最近的可用值，录出来的图和预期不一样。固定 640×480@30 最稳。
5. **`Could not load libtorchcodec` 警告**：本机已实测必现，**可以忽略**。0.6.1 编码端写死 pyav（`configs/video.py:97`），解码端自动降级，`av` 已装好。除非你非要装 ffmpeg 的 *full-shared* 版本（带 DLL），否则别折腾。
6. **路径不要带中文和空格**（`D:/lerobot-data/...` 这种最省事）。
7. **串口被占用**：先关掉厂商上位机、串口助手等，否则 `PermissionError`。
8. **PowerShell 引号**：`--robot.cameras="{...}"` 里的花括号整体当字符串传，PowerShell 下用双引号包住即可；如果被拆开，改成单引号 `'{"front": {...}}'` 或干脆用 CMD。
9. **USB 供电不足**：主从臂同时插一个 Hub 上，遥操时会出现随机关节掉线，用带独立供电的 Hub。
10. **笔记本 4GB 显存别开 wandb + rerun 同时跑**，会抢显存和内存，采集阶段直接全关。
11. **Windows 上训训练用 `--num_workers=0`**：多进程 dataloader 在 Windows 走 spawn，容易卡死或报 `RLock` 相关错误。冒烟测试就是这么过的。
12. **resnet18 预训练权重会下到 C 盘**（`C:\Users\<你>\.cache\torch\hub\checkpoints`，约 45MB）。C 盘紧张就设 `TORCH_HOME=D:/lerobot-data/torch`。
13. **训练结尾那行 `Exception ignored ... _recursion_count` 是 multiprocess 库的退出期噪音**，不影响结果，退出码也是 0。

---

## 12. 一步步验收清单

- [ ] `lerobot-find-port` 能看到两个 COM 口，且能分清主/从
- [ ] `lerobot-setup-motors` 12 个舵机 ID 全部写入成功
- [ ] `lerobot-calibrate` 主从臂都通过，校准文件已生成
- [ ] 无相机遥操正常，关节方向/限位都对
- [ ] `lerobot-find-cameras opencv` 找到外接相机索引
- [ ] 带相机遥操，画面正常、无明显延迟
- [ ] `hf auth whoami` 返回自己的用户名
- [x] **训练链路冒烟测试通过**（2026-11-03，公开数据集 + CPU 20 步，loss 59→14.6）
- [ ] `lerobot-record` 录 2 条测试 episode，本地 `D:/lerobot-data` 能看到 parquet + mp4
- [ ] `lerobot-replay` 能原样回放
- [ ] （训练环境就绪后）`lerobot-train` 跑通 100 步不报错
