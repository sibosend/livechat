#! /usr/bin/env python
# coding=utf-8
import os
import time
import json
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import subprocess
import shlex

# 创建AcsClient实例
client = AcsClient(
        '',
        '',
        "cn-beijing"
)

# 创建request，并设置参数。
request = CommonRequest()
request.set_method('POST')
request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
request.set_version('2019-02-28')
request.set_action_name('CreateToken')


def refresh_asr_token(token):
    cmd_sed = f"/usr/bin/sed -i 's/asr_token = \".*\"/asr_token = \"{token}\"/g' web/demo.html"
    cmd_sed = f"/usr/bin/find /root/data/chazing_incubator/LiveTalking/web -type f -name '*.html' | xargs sed -i 's/asr_token = \".*\"/asr_token = \"{token}\"/g'"

    _chk_out = subprocess.run(cmd_sed, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(cmd_sed)
    print(_chk_out)
    
try : 
    response = client.do_action_with_exception(request)
    print(response)

    jss = json.loads(response)
    if 'Token' in jss and 'Id' in jss['Token']:
        token = jss['Token']['Id']
        expireTime = jss['Token']['ExpireTime']
        print("token = " + token)
        print("expireTime = " + str(expireTime))
        refresh_asr_token(token)
        
except Exception as e:
    print(e)
