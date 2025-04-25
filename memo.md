docker run --rm --env CANDIDATE=36.140.65.192 \
  -p 1935:1935 -p 8080:8080 -p 1985:1985 -p 8000:8000/udp \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 \
  objs/srs -c conf/rtc.conf


firewall-cmd --zone=public --add-port=8000/tcp --permanent
firewall-cmd --zone=public --add-port=8010/tcp --permanent
firewall-cmd --zone=public --add-port=1985/tcp --permanent
firewall-cmd --zone=public --add-port=8000/udp --permanent


export FFMPEG_PATH=/root/autodl-tmp/third_requires/ffmpeg-linux-x64

# start tts
cd /root/data/chazing_incubator/GPT-SoVITS

python api_v2.py

nohup python api_v2.py >> /root/data/chazing_incubator/GPT-SoVITS/log.info &

# 视频编排

```
ffmpeg -i luoxiang.wait.480p.mp4 -s 720x480 -vf fps=25 -qmin 1 -q:v 1 -start_number 0 ./customvideo/image/%08d.png
ffmpeg -i luoxiang.wait.480p.mp4 -vn -acodec pcm_s16le -ac 1 -ar 16000 ./customvideo/luoxiang.wait.wav
```

# 启动脚本

python app.py --model ernerf \
--transport rtcpush --push_url 'http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream' \
--tts gpt-sovits --TTS_SERVER http://127.0.0.1:9880 --REF_FILE /root/data/chazing_incubator/GPT-SoVITS/animale/ref1.wav \
--REF_TEXT "可是在当年没有，那个年头没有电，家里有点儿灯有谁舍得老点儿，天一黑，七八点钟几乎就路上没有行人了。" 


python app.py --model musetalk \
--transport rtcpush --push_url 'http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream' \
--tts gpt-sovits --TTS_SERVER http://127.0.0.1:9880 --REF_FILE /root/data/chazing_incubator/GPT-SoVITS/animale/ref1.wav \
--REF_TEXT "可是在当年没有，那个年头没有电，家里有点儿灯有谁舍得老点儿，天一黑，七八点钟几乎就路上没有行人了。"  \
--avatar_id avator_2  --customvideo_config data/custom_config.json

nohup python app.py --model musetalk \
--transport rtcpush --push_url 'http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream' \
--tts gpt-sovits --TTS_SERVER http://127.0.0.1:9880 --REF_FILE /root/data/chazing_incubator/GPT-SoVITS/animale/ref1.wav \
--REF_TEXT "可是在当年没有，那个年头没有电，家里有点儿灯有谁舍得老点儿，天一黑，七八点钟几乎就路上没有行人了。"  \
--avatar_id avator_3  --customvideo_config data/custom_config_liyubin.json \
--bbox_shift 9 \
>> ./logs/web.log 2>&1 &

-i liyubin60s.noaudio.720p.mp4 -vf 'colorkey=0x3AB04E:similarity=0.2:blend=0.3' -c:a copy -c:v vp9 -f webm output2.webm


nohup  python app.py --model musetalk \
--transport rtcpush --push_url 'http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream' \
--tts gpt-sovits --TTS_SERVER http://127.0.0.1:9880 --REF_FILE /root/data/chazing_incubator/GPT-SoVITS/demo/input/huang.ref02.8s.wav \
--REF_TEXT "根据行业报告，仅在中国，元宇宙市场的规模就预计达到8万亿美元。"  \
--avatar_id avator_6  --customvideo_config data/custom_config_huangjinbiao.json  \
--max_session 1 \
>> ./logs/web.log 2>&1 &
