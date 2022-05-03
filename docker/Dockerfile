FROM ccr.ccs.tencentyun.com/deepwisdom/dw_ubuntu1804_py37_torch180_x86_64_455.45_cuda111:v1.0

COPY . /app/source_code/
RUN rm -rf /app/source_code/.git

ENV LANG=C.UTF-8 LANGUAGE=zh_CN:zh:en_US:en

RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r /app/source_code/requirement.txt
RUN pip uninstall -y drpc
RUN pip install -i https://mirrors.aliyun.com/pypi/simple drpc -i https://pypi.deepwisdomai.com/root/stable

RUN mkdir -p /app/tianji && cp -r /app/source_code/* /app/tianji
RUN cd /app/tianji/FGVC-PIM && pip install -i https://mirrors.aliyun.com/pypi/simple -r requirement.txt && pip install -e .

WORKDIR /app/tianji
