import os
import torch.nn.parallel
import torchvision.transforms as T
import torch.nn.functional as F
from models import emotion_model
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2

emo_dict = {0: 'Surprise',
            1: 'Fear',
            2: 'Disgust',
            3: 'Happy',
            4: 'Sad',
            5: 'Angry',
            6: 'Neutral'}


def load_model(model_path):
    # Load model
    model = emotion_model.Model(model_path='./models/resnet18_msceleb.pth')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()
    return model


def run_on_test_images(raf_path, model, transform_val):
    base_dataset = pd.read_csv(os.path.join(raf_path, 'rafdb/list_patition_label.txt'), sep=' ', header=None,
                               names=['img', 'label'])
    # change the names to actual names of the images and make labels start from 0
    add_align = lambda x: str(x).split('.')[0] + '_aligned.jpg'
    base_dataset['img'] = base_dataset['img'].apply(add_align)
    base_dataset['label'] = base_dataset['label'] - 1
    test = base_dataset[base_dataset['img'].str.startswith('test')].values.tolist()

    # Show some results
    random.shuffle(test)

    # Optional filter for a specific emotion
    # test = list(filter(lambda x: x[1] == 1, test))

    test = test[:20]
    plt.figure()
    for i, (img_name, target) in enumerate(test):
        img_path = os.path.join(raf_path, 'rafdb/aligned', img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = transform_val(img).cuda()
        with torch.no_grad():
            scores = model(input_img.unsqueeze(0))
            scores = F.softmax(scores, dim=1)
            pred = torch.argmax(scores, dim=1).cpu().item()
            conf = scores[0][pred].item()
            if emo_dict[pred] == emo_dict[target]:
                color = 'green'
            else:
                color = 'red'
            plt.subplot(5, 4, i + 1), plt.imshow(img), plt.title(f'{emo_dict[pred]}:{conf:.2f}', color=color), plt.axis(
                'off')

    plt.show()


def main():
    raf_path = './data/RafDB'
    model_path = './result/exp025_mixup_224/model_best.pth.tar'
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_val = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize
    ])
    model = load_model(model_path)
    run_on_test_images(raf_path, model, transform_val)


if __name__ == '__main__':
    main()