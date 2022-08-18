## 配置环境

- 运行环境
    python3.7

- 安装 
    $ python3.7 -m pip install -r requirements1.txt
    $ python3.7 -m pip install -r requirements2.txt

## 数据下载

    $ rsync -av XXX@172.30.0.56:/volume1/先进研究院/public_data/腹部器官分割/FLARE2022/对flare官方提交结果/SubmitCode/example/model/ ./model/
    $ rsync -av XXX@172.30.0.56:/volume1/先进研究院/public_data/腹部器官分割/FLARE2022/对flare官方提交结果/SubmitCode/example/inputs/ ./inputs/

## 测试运行
    $ sh predict.sh
