from funasr import AutoModel

model = AutoModel(model="iic/emotion2vec_base")

res = model(input='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav', output_dir="./outputs", granularity="utterance", extract_embedding=True)
print(res)