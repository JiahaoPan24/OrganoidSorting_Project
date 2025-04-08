import tensorflow as tf
import pathlib
import PySimpleGUI as sg
from PIL import Image, ImageOps
import numpy as np

def load_full_model(path: pathlib.Path):
    return tf.keras.models.load_model(path)

model_path = pathlib.Path('./OptimizedModel')
model = load_full_model(model_path)
layout = [
    [sg.Text('请选择图片：'), sg.Input(), sg.FileBrowse('浏览'), sg.Button('上传')],
    [sg.Image(key='-IMAGE-')],  # 用于显示图片
    [sg.Button('识别'), sg.Button('退出')]
]

window = sg.Window('图片识别', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == '退出':
        break
    elif event == '上传':
        image_path = values[0]  # 获取用户选择的图片路径
        if image_path:
            window['-IMAGE-'].update(filename=image_path)  # 显示图片
    elif event == '识别':
        if image_path:
            # 加载图片，转换成模型需要的格式
            img = Image.open(image_path)
            img = ImageOps.fit(img, (256, 256))  # 假设你的模型需要256x256的输入
            img = np.array(img) / 255.0  # 归一化
            img = img.astype(np.float32)
            img = np.expand_dims(img, axis=0)  # 增加一个维度，以匹配模型的输入
            
            # 使用模型进行预测
            prediction = model.predict(img)
            # 这里添加你的代码来处理预测结果，例如显示预测的掩码或分类结果

window.close()

