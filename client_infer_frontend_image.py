# @Author  : guozebin (guozebin@fuzhi.ai)
# @Desc    :
import imageio
import os

from utils.service_tools import get_host_ip, HttpClient
from utils.image_io import encode_image_base64, decode_base64_image


def main_func():
    image_path = '/home/liuguangcan/internship/Fine-Grained-Classification/FGVC_PIM/datas/CUB_200/valid/images/155.Warbling_Vireo/Warbling_Vireo_0076_158500.jpg'
    url = f'http://{get_host_ip()}:8080/autotable/predict'
    path = '/home/liuguangcan/internship/Fine-Grained-Classification/FGVC_PIM/datas/CUB_200/valid/images'
    for i in sorted(os.listdir(path)):
        print(i)
        for j in sorted(os.listdir(os.path.join(path, i))):
            print(j)
            image_path = os.path.join(path, i, j)
            image = imageio.imread(image_path)
            image = encode_image_base64(image)
            post_data = {'content': image,'need_visual_data': True, 'need_visual_result': True}
            #post_data = {'content': image}
            client = HttpClient(url)
            result = client.post(json=post_data)
            #predict_decode = decode_base64_image(result['data'][0]['visual_result'])

            #imageio.imwrite(save_path, predict_decode)
            print(result)
            print('finished ...')


if __name__ == '__main__':
    main_func()

