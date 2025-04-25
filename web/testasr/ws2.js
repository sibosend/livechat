const startBtn = document.getElementById('btnStart_asr');
const stopBtn = document.getElementById('btnStop_asr');
const statusDiv = document.getElementById('status');
const input_mode_d2 = document.getElementById('input_mode');
const message_v2 = document.getElementById('message');
// const timeSpan = document.getElementById('time');

//初始化录音实例
let recorder = new Recorder({
    sampleBits: 16,                 // 采样位数，，默认是16
    sampleRate: 16000,              //音频采样率，默认是16000Hz，
    numChannels: 1,                 // 声道，支持 1 或 2， 默认是1
    compiling: true                 // 是否边录边转换，默认是false
});


//获取ali的token
// let token = "7dc73a9275b043f6ab2184c23e3006ff";
// let appkey = "aDGiZcvxOq2cpJ2d";

let appkey = document.getElementById('wssappkey').value;

var interval; // 定时器

//定义ws的相关参数
let websocket = null; //websocket实例
let timer_websocket = null; //websocket定时器, 用于实时获取语音转文本的结果
let websocket_task_id = null; //websocket任务id 整个实时语音识别的会话ID，整个请求中需要保持一致，32位唯一ID。
let websocket_audio2txt_time = 0; //websocket 语音转文本  一句话收集完毕的时间，用于判断间隔 
let websocket_audio2txt_result_msg = null; //websocket实例 音频转文字的结果
let websocket_audio2txt_result_msg_temp = null; //websocket实例 音频转文字的结果
let websocket_audio2txt_complete_b = false; //websocket 语音转文本  是否完成   true:完毕  false:未完毕
let websocket_audio2txt_complete_time_end = 100;  //websocket 语音转文本 判断文本是否收集完毕的阈值  单位毫秒 


//设置cookies
const cookies = '';

//请求api获取token的信息
// getToken();
async function getToken() {
    const url = 'https://xxxxxx/ali/token';
    await fetch(url, {
        method: 'GET',
        headers: {
            "Content-Type": "application/json",
            'Authorization': cookies
        }
    })
        .then(res => res.json())
        .then(res => {
            //设置token
            token = res.data.token;
            appkey = res.data.appkey;
        })
        .catch(error => {
            console.log('error:', error);
            alert('获取token失败');
        });
}


//开始录音
startBtn.onclick = function () {

    input_mode_d2.value = 1;

    recorder.start().then(() => {
        // timeSpan.innerText = '0s';
        statusDiv.innerHTML = '';

        tran_message();

        websocket_audio2txt_result_msg = "";//置空
        initWebSocket();
    }, (error) => {
        console.log(`出错了`);
    });

    change_recording_state("on");

    recorder.onprogress = function (params) {
        // timeSpan.innerText = Math.floor(params.duration) + 's';
        //  console.log('--------------START---------------')
        // console.log('录音时长(秒)', params.duration);
        //  console.log('录音大小(字节)', params.fileSize);
        //  console.log('录音音量百分比(%)', params.vol);
        // console.log('当前录音的总数据)', params.data);
        //  console.log('--------------END---------------')
    }

};


stopBtn.onclick = function () {
    console.log('结束录音');

    send_new_chat();

    change_recording_state("off");

    recorder.stop();
    //initWebSocket();
    clearInterval(interval);

    waitSpeakingEnd();
};


// playBtn.onclick = function () {
//     console.log('播放录音');
//     recorder.play();//播放录音
// };


//初始化websocket
function initWebSocket() {
    console.log("初始化weosocket");

    //初始化参数
    websocket_audio2txt_complete_b = false;
    websocket_audio2txt_time = 0;

    //检测如果未关闭、则先关闭在重连
    if (websocket !== null) {
        websocket.close();
        websocket = null;
    }

    //ali的websocket地址
    //const wsuri = `wss://nls-gateway.cn-shanghai.aliyuncs.com/ws/v1?token=${token}`;
    const wsuri = document.getElementById('wssip').value;

    //连接wss服务端
    websocket = new WebSocket(wsuri);
    //指定回调函数
    websocket.onopen = websocketOnOpen;
    websocket.onmessage = websocketOnMessage;
    websocket.onerror = websocketOnError;
    websocket.onclose = websocketClose;
}

//建立连接
function websocketOnOpen() {
    // console.log("向 websocket 发送 链接请求");

    //生成新的任务id
    websocket_task_id = getRandomStrNum();
    //生成ali的请求参数message_id
    let message_id = getRandomStrNum();
    let actions = {
        "header": {
            "namespace": "SpeechTranscriber",    //固定值  
            "name": "StartTranscription",       //发送请求的名称，固定值
            "appkey": appkey,                   //appkey
            "message_id": message_id,           //消息id
            "task_id": websocket_task_id,       //任务id  
        },
        "payload": {
            "format": "PCM",//音频编码格式，默认是PCM（无压缩的PCM文件或WAV文件），16bit采样位数的单声道。
            "sample_rate": 16000, //需要与录音采样率一致、默认是16000，单位是Hz。
            "enable_intermediate_result": true, //是否返回中间识别结果，默认是false。
            "enable_punctuation_prediction": true, //是否在后处理中添加标点，默认是false。
            "enable_inverse_text_normalization": true, //是否在后处理中执行数字转写，默认是false。
            "max_sentence_silence": 500,//	语音断句检测阈值，静音时长超过该阈值会被认为断句，参数范围200ms～2000ms，默认值800ms。
        }
    }

    //发送请求
    websocketSend(JSON.stringify(actions));
}


/****************ws 请求处理 start *********************/
//发送数据
function websocketSend(data) {
    //console.log('websocket 数据发送',data);
    //判断是否连接成功,连接成功再发送数据过去
    if (websocket.readyState === 1) {
        websocket.send(data);
    } else {
        console.log('websock未连接-------------------');
    }
}

//接收数据
function websocketOnMessage(e) {
    //接受ali 语音返回的数据
    const ret = JSON.parse(e.data);
    //判断返回的数据类型
    if (ret.header.name === 'TranscriptionResultChanged') {
        //数据在收集中 一句话的中间结果
        // console.log('数据在收集中');
        //实时获取语音转文本的结果
        ingText(ret.payload.result);
    } else if (ret.header.name === 'SentenceBegin') {
        //一句话开始后，就可以启动录音了
        console.log('检测到了一句话的开始');
        //添加一个新的p标签、用于显示中间变化状态
        var span = document.createElement("p")
        span.innerText = ""
        statusDiv.appendChild(span);

    } else if (ret.header.name === 'TranscriptionStarted') {
        console.log("服务端已经准备好了进行识别，客户端可以发送音频数据了");
        //获取音频信息，定时获取并发送
        interval = setInterval(() => {
            getPCMAndSend();
        }, 100)
    } else if (ret.header.name === 'SentenceEnd') {
        console.log('数据接收结束', ret);
        endText(ret.payload.result);
    } else if (ret.header.name === 'TranscriptionCompleted') {
        console.log('服务端已停止了语音转写', ret);
    }
}

//错误处理
function websocketOnError(e) {
    console.log("连接建立失败重连");
    //initWebSocket();
}

//关闭处理
function websocketClose(e) {
    console.log('websocketClose断开连接', e);
}

//wss 连接建立之后发送 StopTranscription指令
function websocketSendStop() {
    console.log("向  websocket 发送 Stop指令");
    let message_id = getRandomStrNum();
    //actions 是首次连接需要的参数,可自行看阿里云文档
    let actions = {
        "header": {
            "message_id": message_id,
            "task_id": websocket_task_id,
            "namespace": "SpeechTranscriber",
            "name": "StopTranscription",
            "appkey": appkey,
        }
    };

    //发送结束指令
    websocketSend(JSON.stringify(actions));
}


function ingText(text) {
    let ps = statusDiv.querySelectorAll('p');
    //更新中间变化状态 
    let lastP = ps[ps.length - 1];
    lastP.innerText = text;

    tran_message();
}


//设置定时器-websocket 实时获取语音转文本的结果
function endText(text) {
    let ps = statusDiv.querySelectorAll('p');
    //更新最后的识别结果
    let lastP = ps[ps.length - 1];
    lastP.innerText = text;

    tran_message();

    //获取全文
    websocket_audio2txt_result_msg += text
    console.log('websocket_audio2txt_result_msg:', websocket_audio2txt_result_msg);

    //todo 可以进行匹配语音匹配了

}


/****************ws 请求处理  end *********************/


//获取音频信息，并发送
function getPCMAndSend() {
    //获取音频信息
    let NextData = recorder.getNextData();
    let blob = new Blob([NextData])
    let blob_size = blob.size;
    // console.log("获取音频信息，并发送,blob_size:" + blob_size, blob);

    //ali最大支持3200字节的音频
    let max_blob_size = 3200;//支持1600 或3200
    let my_num = blob_size / max_blob_size;
    my_num = my_num + 1;

    //切分音频发送
    for (let i = 0; i < my_num; i++) {
        var end_index_blob = max_blob_size * (i + 1);
        //判断结束时候的分界
        if (end_index_blob > blob_size) {
            end_index_blob = blob_size;
        }
        //切分音频
        var blob2 = blob.slice(i * max_blob_size, end_index_blob);
        //生成新的blob
        const newbolb = new Blob([blob2], { type: 'audio/pcm' })
        //发送
        websocketSend(newbolb);
    }
}


//生成32位随机数UUID
function getRandomStrNum() {
    var s = [];
    var hexDigits = "0123456789abcdef";
    for (var i = 0; i < 32; i++) {
        s[i] = hexDigits.substr(Math.floor(Math.random() * 0x10), 1);
    }
    s[14] = "4";  // bits 12-15 of the time_hi_and_version field to 0010
    s[19] = hexDigits.substr((s[19] & 0x3) | 0x8, 1);  // bits 6-7 of the clock_seq_hi_and_reserved to 01
    s[8] = s[13] = s[18] = s[23];

    var uuid = s.join("");
    return uuid;
}



const sleep = (delay) => new Promise((resolve) => setTimeout(resolve, delay))
async function is_speaking() {
    const response = await fetch('/is_speaking', {
        body: JSON.stringify({
            sessionid: 0,
        }),
        headers: {
            'Content-Type': 'application/json'
        },
        method: 'POST'
    });
    const data = await response.json();
    // console.log('is_speaking res:', data)
    return data.data
}

async function waitSpeakingEnd() {

    change_recording_state("wait");

    // rec.stop() //关闭录音
    for (let i = 0; i < 10; i++) {  //等待数字人开始讲话，最长等待10s
        bspeak = await is_speaking()
        if (bspeak) {
            break
        }
        await sleep(1000)
    }

    //已经等到了
    render_bot(chat_bot_response);

    while (true) {  //等待数字人讲话结束
        bspeak = await is_speaking()
        if (!bspeak) {
            break
        }
        await sleep(1000)
    }
    await sleep(2000)
    // rec.start()

    change_recording_state("off");
}