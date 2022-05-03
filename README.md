### 任务介绍

  - 细粒度分类任务sota 模型的训练，输入为图片，输出为类比及其对应的score

- ### 数据集格式

    - 训练和测试
        - 原始数据集：
        - 处理好可以训练的数据集：
        - 用于调试的子集：
        - 自定义数据集规则：使用 total_text 数据集格式的数据：
            ```
                |- dataset_name
                    |- train
                        |- images/                   # 自定义数据集的训练数据
                            |- class1
                                |-img1.jpg
                                |-img2.jpg
                                ...
                            |- class2  
                                 |-img1.jpg
                                 |-img2.jpg
                                 ...
                            ...
                        |- annotations/
                    |- valid
                        |- images/                   # 自定义数据集的验证数据
                            |- class1
                                |-img1.jpg
                                |-img2.jpg
                                ...
                            |- class2  
                                 |-img1.jpg
                                 |-img2.jpg
                                 ...
                            ...
                        |- annotations/
                    |- test
                        |- images/                   # 自定义数据集的测试数据
                            |- class1
                                |-img1.jpg
                                |-img2.jpg
                                ...
                            |- class2  
                                 |-img1.jpg
                                 |-img2.jpg
                                 ...
                            ...
                        |- annotations/
             ```
           
            **help：分类数据集的annotations文件夹为空，类别直接从images的文件名中读取**。
            
                
   - infer 样本
        - 图片，分辨率不限。

### 训练

- 输入接口
                
```
                python
                parser = argparse.ArgumentParser()
                parser.add_argument("--dataset_path", type=str, default="/data/CUB_200/", help='')
                parser.add_argument("--device", type=str, default="cuda")
                parser.add_argument("--trial_name", type=str, default="autotables-2503-train-1")
                parser.add_argument("--annotation_data", type=str, default="{'advance_settings':{}}")
```
-  输出目录结构
```
                python
                # 训练阶段结果产出的文件结构
                /app/tianji/
                ├───── client_train.py
                ├───── runs
                ├─────── models
                ├────────── {uuid}  # {uuid}自己生成，文件夹下存放产出模型文件等
                ├─────── metric
                ├────────── trial.txt  # 文件内的model_path为上述uuid的绝对路径
```
- metric 文件格式

```
        python
        {
            "recom_model": "FGVC_PIM",
            "gened_model_name": "FGVC_PIM_autotables-2503-train-1",
            "metric_type": "acc",
            "metric_value": 81.6506,
            "used_models": "FGVC_PIM",
            "model_path": "/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc",
            "model_params": {},
            "eval_result": {
                "FGVC_PIM": {
                    "eval": {
                        "acc": 81.6506
                    }
                }
            },
            "predict_prob": {},
            "train_consuming": {
                "FGVC_PIM": {
                    "total": 793.2808,
                    "train_time": 681.339,
                    "valid_time": 100.2668
                }
            }
        }
```

- 指标结果

    输出格式为表格：
    
    | 数据集  | 指标          | 备注          |
    |:-----|:------------|:------------         |
    |      | `acc` | 分类准确率指标 |

### 离线验证

- 输入接口

     ```python
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_path", type=str, default="/data/CUB_200/", help='')
        parser.add_argument("--model_path", type=str, default=r"/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc", help='')
        parser.add_argument("--used_model", type=str, default="FGVC_PIM_autotables-2503-train-1.pth", help='')
        parser.add_argument("--device", type=str, default='cuda')
        ```

- 输出目录结构

     ```python
        # 训练阶段结果产出的文件结构
        /app/tianji/
        ├───── client_test.py
        ├───── runs
        ├─────── models
        ├────────── {uuid}  # {uuid}自己生成，无需和训练的uuid一致
        ├───────────── evaluation_1649384314.json  # evaluation_{timestamp}.json 为{uuid}文件夹下的离线结果文件
        ├─────── metric
        ├────────── trial.txt  # 文件内的model_path为上述uuid的绝对路径
     ```

- metric文件格式

     ```python
             {
            "model_path": "/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc",
            "predict_result_filename": "evaluation_1649929022.json",
            "preview_data": {},
            "test_result": {
                "FGVC_PIM": {
                    "test": {
                        "acc": 81.563
                    }
                }
            },
            "predict_prob": {},
            "test_consuming": {
                "FGVC_PIM": {
                    "test": 42.2669
                }
            }
        }
       ```

### 推理

服务请求路由：/autotable/predict
服务端口: 8080

- 服务启动：

      ```python
            parser = argparse.ArgumentParser()
            parser.add_argument("--model_path", type=str, default=r"/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc")
            parser.add_argument("--used_model", type=str, default="FGVC_PIM_autotables-2503-train-1", help='')
            parser.add_argument("--device", type=str, default='cuda')
            parser.add_argument("--service_port", type=int, default=8080)
      ```

- 请求示例：

     ```python
        post_data = {
            'content': image,'need_visual_data': True, 'need_visual_result': True
                    }
     ```

- 返回示例：

     ```python
     {
        'code': 200,
        'msg': 'ok', 
        'data': [{
                'infer_data': ['020.Yellow_breasted_Chat_1.0', '1.0'], 
                'visual_data':
                       {'output_type': 'text', 
                        'annotation_type': None, 
                        'data': '020.Yellow_breasted_Chat_1.0'
                        },
               'visual_result': '020.Yellow_breasted_Chat_1.0'
        }]
     }
     
     ```

- 推理示例
      
    输入：
    
    ![输入](Yellow_Breasted_Chat_0012_21961.jpg)
    
    输出：'020.Yellow_breasted_Chat_1.0'

 
