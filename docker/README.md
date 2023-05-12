### 使用 docker 进行工作

使用 docker 可以保持系统环境的干净 ，避免各种库版本的兼容问题 ，影响工作环境 。 毕竟 删除 docker ， 比清理 系统 容易多了。

使用 docker 的时候 ， 希望遵循以下的一些规则 ，能够帮助开发工作更好的进行 。

* 尽量不要使用 root
* 将数据保留在 host 宿主机中 
* 启动 容器 进行测试的时候 ， 尽量带着 -rm 参数 ， 这样退出容器的时候 ， 会自动 删除 容器进程
* 启动 容器 的 命令 ， 建议写在一个 脚本里 。


### 本 docker 的一些参数

* 镜像是 基于 nvidia 的 11.8 - ubuntu 2004 制作
* pip 源设置为 https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
* 用户为  lk ， 密码是 password@lk
* lk 可以 sudo
* 用户lk 的 id 和 组 id 都 为 8001 
* 建议在 host 中 ， 新建一个 id 为 8001 的 用户 ， 在这个新用户 环境下 使用 docker ，以 解决 文件访问 权限 问题 。
* 新建用户和组的 命令 类似下面 ， 请 自行修改
```
groupadd -g 8001  lk && useradd -u 8001 -r -g lk lk 
```
* 将用户 加入 docker 组 ， 该用户就可以使用 docker 了 ， 无需 sudo 。

### 构建镜像命令 

其中 t 参数 指定 在 docker 中的镜像名 ， f 参数 ， 指定 dockerfile ， 请自行 修改 。 注意命令最后的 .  是当前目录的意思 。

```
docker build -t  dev-moss  -f  ./Dockerfile   .
```

### 运行 docker 容器的脚本如下 
* 这个脚本假设 镜像名 叫 dev-moss
* 用 -v参数 将 host 目录 /home/lk/workplace/cache 映射到 docker容器内 的 /home/lk/.cache  目录 ， 以便保存 下载 的 模型数据等文件 
* 用 -v参数 将 host 目录 /home/lk/workplace/code 映射到 docker容器内 的 /home/lk/code  目录 ， 以便保存 代码 等文件 
* -w 参数 ， 指明进入容器后的开始目录 

```
#!/usr/bin/sh
docker run \
  -v /home/lk/workplace/code:/home/lk/code \
  -v /home/lk/workplace/cache:/home/lk/.cache \
  --net=host \
  --gpus=all \
  --shm-size=64g \
  -e WANDB_DISABLED=true \
  -it \
  --rm \
  -w /home/lk/code \
  dev-moss \
  bash
```

