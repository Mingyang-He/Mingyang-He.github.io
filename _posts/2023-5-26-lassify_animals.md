# Classify 10 animals

This example aims to use Duck Duck Go to scrape sample images off the Web based on the fast.ai course
example on birds and design an appropriate multiclass loss function. :alien:

**Step One: searching and downlowding images**
```
from fastbook import *
from fastai.vision.widgets import *

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(search_images_ddg(term, max_images=max_images))
```
:memo:

Imports all of the fastai.vision and fastbook library.

define a function, **_search_images_**, which labels images based on a filename rule provided by the dataset creators.

```
searches = 'dog','bird','cat','goat','lizard','bull','snake','bear','fish','kangaroo'
path = Path('bird_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    resize_images(path/o, max_size=400, dest=path/o)
```
:memo:

Grab some examples of animal photos, and save each group of photos to a different folder.

**Step Two: train the model**
```
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```
:memo:

Sometimes photos may not download correctly which could cause fail model training, so they have to be removed.

```
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
```
:memo:

This part tells fastAI what kind of dataset we have and how it is structured. 
In fastAI we can create that easily using a DataBlock, and view sample images from it.

**_blocks_** cleassify our dataset. The inputs to our model are images, and the outputs are categories (in this case, "bird" or "cat").

To find all the inputs to our model, run the **_get_image_files_** function. it will return a list of all image files from that path.

The parameter **_valid_pct=0.2_** tells fastai to hold out **20%** of the data and not use it for training the model at all. This 20% of the data is called the **validation set**; the remaining **80%** is called the **training set**. The validation set is used to measurer the accuracy of the model. By default, the 20% is selected randomly.

The parameter **_seed=42_** sets the random seed to the same value every time we run this code, which means we get the same validation set every time we run it. 

Finally, **_resize_** the image to 192x192 pixels.

```
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```
:memo:

This part creates a convolutional neural network (CNN) and specifies what architecture to use, what data we want to train it on, and what metric to use.

**_resnet34_** refers to the number of layers in this variant of the architecture, and in this case we are using 18 layers vision model.

By using **_metrics=error_rate_**, it tells what percentage of images in the validation set are being classified incorrectly.

**_learn.fine_tune(3)_** tells fastAI how many times to look at each image.


















