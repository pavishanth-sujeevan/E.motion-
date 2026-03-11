---
license: other
license_name: model-license
license_link: https://github.com/alibaba-damo-academy/FunASR
frameworks:
- Pytorch
tasks:
- emotion-recognition
---

<div align="center">
    <h1>
    EMOTION2VEC
    </h1>
    <p>
    emotion2vec: universal speech emotion representation model <br>
    <b><em>emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation</em></b>
    </p>
    <p>
    <img src="logo.png" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
</div>

# Guides
emotion2vec is the first universal speech emotion representation model. Through self-supervised pre-training, emotion2vec has the ability to extract emotion representation across different tasks, languages, and scenarios.

The version is an pre-trained representation model without fine-tuning, which can be used for feature extraction. 

# Model Card
GitHub Repo: [emotion2vec](https://github.com/ddlBoJack/emotion2vec)
|Model|⭐Model Scope|🤗Hugging Face|Fine-tuning Data (Hours)|
|:---:|:-------------:|:-----------:|:-------------:|
|emotion2vec|[Link](https://www.modelscope.cn/models/iic/emotion2vec_base/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_base)|/|
emotion2vec+ seed|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_seed/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_seed)|201|
emotion2vec+ base|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_base/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_base)|4788|
emotion2vec+ large|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_large)|42526|

# Installation

`pip install -U funasr modelscope`

# Usage
input: 16k Hz speech recording

granularity:
- "utterance": Extract features from the entire utterance
- "frame": Extract frame-level features (50 Hz)

extract_embedding: Whether to extract features

## Inference based on ModelScope

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_base")

rec_result = inference_pipeline('https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav', output_dir="./outputs", granularity="utterance", extract_embedding=True)
print(rec_result)
```


## Inference based on FunASR

```python
from funasr import AutoModel

model = AutoModel(model="iic/emotion2vec_base")

res = model(input='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav', output_dir="./outputs", granularity="utterance", extract_embedding=True)
print(res)
```
Note: The model will automatically download.

Supports input file list, wav.scp (Kaldi style):
```cat wav.scp
wav_name1 wav_path1.wav
wav_name2 wav_path2.wav
...
```

Outputs are emotion representation, saved in the output_dir in numpy format (can be loaded with np.load())

# Note

This repository is the Huggingface version of emotion2vec, with identical model parameters as the original model and Model Scope version.

Original repository: [https://github.com/ddlBoJack/emotion2vec](https://github.com/ddlBoJack/emotion2vec)

Model Scope repository: [https://www.modelscope.cn/models/iic/emotion2vec_plus_large/summary](https://www.modelscope.cn/models/iic/emotion2vec_plus_large/summary)

Hugging Face repository: [https://huggingface.co/emotion2vec](https://huggingface.co/emotion2vec)

FunASR repository: [https://github.com/alibaba-damo-academy/FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/funasr1.0/examples/industrial_data_pretraining/emotion2vec)

# Citation
```BibTeX
@article{ma2023emotion2vec,
  title={emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation},
  author={Ma, Ziyang and Zheng, Zhisheng and Ye, Jiaxin and Li, Jinchao and Gao, Zhifu and Zhang, Shiliang and Chen, Xie},
  journal={arXiv preprint arXiv:2312.15185},
  year={2023}
}
```