- 任务介绍

    - 视频超分，输入为一段较短的视频，输出为分辨率增加4倍的视频

- ### 数据集格式

    - 训练和测试
        - 原始数据集：http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip
        - 处理好可以训练的数据集：
        - 用于调试的子集：
        - 自定义数据集规则：视频段抽帧为7张或更多图片(更多，对资源要求更高)，放在一个文件夹下，构成一个样本，结构如下：
            - dataset_name
                ├─sample_1
                │ ├─ frame_1.png
                │ ├─ frame_2.png
                │ ├─ ...
                ├─sample_2
                │ ├─ frame_1.png
                │ ├─ frame_2.png
                │ ├─ ...
                ├─...
    - infer样本
        - 视频文件，分辨率在200x200以下，视频时长不超过5s

### 训练

- 输入接口

    - ```python
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_path", type=str, default="/app/publicdata/dataset/gp/vimeo90k_sample", help='')
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--trial_name", type=str, default="autotables-2503-train-1")
        parser.add_argument("--annotation_data", type=str, default="{'advance_settings':{}}")
        ```

-  输出目录结构

    - ```python
        # 训练阶段结果产出的文件结构
        /app/tianji/
        ├───── client_train.py
        ├───── runs
        ├─────── models
        ├────────── {uuid}  # {uuid}自己生成，文件夹下存放产出模型文件等
        ├─────── metric
        ├────────── trial.txt  # 文件内的model_path为上述uuid的绝对路径
        ```

- metric文件格式

    - ```python
        {
            "recom_model": "FRVSR_GAN",
            "gened_model_name": "FRVSR_GAN_autotables-2503-train-1",
            "metric_type": "PSNR",
            "metric_value": 5.6487,
            "used_models": "FRVSR_GAN",
            "model_path": "/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc",
            "model_params": {},
            "eval_result": {
                "FRVSR_GAN": {
                    "eval": {
                        "MSE": 5.7194,
                        "SSIMs": 2.5659,
                        "PSNR": 5.6487,
                        "SSIM": 0.1222
                    }
                }
            },
            "predict_prob": {},
            "train_consuming": {
                "total": 447.784,
                "train_time": 444.5943,
                "valid_time": 3.1897
            }
        }
        ```

- 指标结果

输出格式为表格：

| 数据集 | 指标    | 备注                   |
| :----- | :------ | :--------------------- |
|        | PSNR    | 峰值信噪比             |
|        | `MSE`   | 均方误差               |
|        | `SSIMs` | 所有样本结构相似性之和 |
|        | `SSIM`  | 结构相似性均值         |





**离线验证**

- 输入接口

    - ```python
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_path", type=str, default="/app/publicdata/dataset/gp/vimeo90k_sample/", help='')
        parser.add_argument("--model_path", type=str, default=r"/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc", help='')
        parser.add_argument("--used_model", type=str, default="FRVSR_GAN_autotables-2503-train-1.pth", help='')
        parser.add_argument("--device", type=str, default='cpu')
        ```

- 输出目录结构

    - ```python
        # 训练阶段结果产出的文件结构
        /app/tianji/
        ├───── client_test.py
        ├───── runs
        ├─────── models
        ├────────── {uuid}  # {uuid}自己生成，无需和训练的uuid一致
        ├───────────── evaluation_1618803641.xlsx  # evaluation_{timestamp}.xlsx 为{uuid}文件夹下的离线结果文件
        ├─────── metric
        ├────────── trial.txt  # 文件内的model_path为上述uuid的绝对路径
        ```

- metric文件格式

    - ```python
        {
            "model_path": "/app/tianji/runs/models/",
            "predict_result_filename": "evaluation_1632987110.json",
            "preview_data": {},
            "test_result": {
                "FRVSR_GAN": {
                    "test": {
                        "MSE": 33.0073,
                        "SSIMs": 21.9846,
                        "PSNR": 6.2752,
                        "SSIM": 0.157
                    }
                }
            },
            "predict_prob": {},
            "test_consuming": {
                "test": 22.0962
            }
        }
        ```

### 推理

服务请求路由：/autotable/predict
服务端口: 8080

- 服务启动：

    - ```python
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, default=r"c:/Users/qing2012/Downloads", help='')
        parser.add_argument("--used_model", type=str, default="FRVSR_GAN_autotables-2503-train-1.pth", help='')
        parser.add_argument("--device", type=str, default='cpu')
        parser.add_argument("--service_port", type=int, default=8080)
        ```

- 请求示例：

    - ```python
        {
            "content": b'UklGRorgIgBBVkkgTElTVOwRAABoZHJs ...  # "视频的base64编码串"
        }
        ```

- 返回示例：

    - ```python
        {
            "code": 200,
            "data": {
                "content": "b'UklGRorgIgBBVkkgTElTVOwRAABoZHJs ..., # 超分结果的base64编码
                "elapsed_time": 0.4024 # 预测耗时s
            }
            "msg": "ok"
        }
        ```

        