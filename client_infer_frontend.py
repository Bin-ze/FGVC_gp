# @Author  : zhangyouyuan (zyyzhang@fuzhi.ai)
# @Desc    :
from utils.service_tools import get_host_ip, HttpClient
from utils.image_io import encode_video_base64_from_file, decode_base64_video, save_video


def main_func():
    video_path = 'sample.avi'
    save_path = 'sample_sr.mp4'
    url = f'http://{get_host_ip()}:8080/autotable/predict'

    video = encode_video_base64_from_file(video_path)
    post_data = {'content': video}
    client = HttpClient(url)
    # result = client.post(header=None, data=post_data, json=None)
    result = client.post(json=post_data)
    predict_decode = decode_base64_video(result['data'][0]['content'])

    save_video(predict_decode, save_path, fps=6)

    print('finished ...')


if __name__ == '__main__':
    main_func()
