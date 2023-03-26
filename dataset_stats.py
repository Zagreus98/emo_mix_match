import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
emo_dict = {0: 'Surprise',
            1: 'Fear',
            2: 'Disgust',
            3: 'Happy',
            4: 'Sad',
            5: 'Angry',
            6: 'Neutral'}
raf_path = './data/RafDB'
base_dataset = pd.read_csv(os.path.join(raf_path, 'rafdb/list_patition_label.txt'),
                           sep=' ', header=None,
                           names=['img', 'label'])
add_align = lambda x: str(x).split('.')[0] + '_aligned.jpg'
base_dataset['img'] = base_dataset['img'].apply(add_align)
base_dataset['label'] = base_dataset['label'] - 1
train = base_dataset[base_dataset['img'].str.startswith('train')]
print(f'Number of train images: {len(train)}')
test = base_dataset[base_dataset['img'].str.startswith('test')]
print(f'Number of test images: {len(test)}')
emotions = list(emo_dict.values())

labels_train = train.groupby('label').count().to_dict()['img']
emo_count = labels_train.values()
plt.figure()
plt.bar(emotions, emo_count)
plt.title('Train dataset emotion distribution')

labels_test = test.groupby('label').count().to_dict()['img']
emo_count = labels_test.values()
plt.figure()
plt.bar(emotions, emo_count)
plt.title('Test dataset emotion distribution')

data = base_dataset.values.tolist()[:20]
plt.figure()
for i, (img_name, target) in enumerate(data):
    img_path = os.path.join(raf_path, 'rafdb/aligned', img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(5, 4, i + 1), plt.imshow(img), plt.title(f'{emo_dict[target]}'),
    plt.axis('off')
plt.show()


