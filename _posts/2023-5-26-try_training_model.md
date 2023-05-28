# Use our model

After building and training the fastAI model in previous post, we can use this model to test and identify animals from our 10-animals-library. 👽

First of all, as an example, we can search for a random dog photo and seeing what result we can get.

```
urls = search_images('dog photos', max_images=1)
urls[0]

from fastdownload import download_url
dest = 'dog.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)
```

![dog_pic](http://www.publicdomainpictures.net/pictures/40000/velka/cute-dog-1362593345bn4.jpg)

Let's see what our model thinks about that image we downloaded. ❓

```
is_animal,_,probs = learn.predict(PILImage.create('dog.jpg'))
print(f"This is a: {is_animal}.")
print(f"Probability it's a dog: {probs[1].item():.4f}")
```

By running the code, it obtains the result that:

![](/images/10.png)

Similarly, we can try our model with other examples
```
download_url(search_images('bird photos', max_images=1)[0], 'bird.jpg', show_progress=False)
Image.open('bird.jpg').to_thumb(256,256)

download_url(search_images('stone photos', max_images=1)[0], 'stone.jpg', show_progress=False)
Image.open('stone.jpg').to_thumb(256,256)

is_animal,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_animal}.")
print(f"Probability it's a bird: {probs[1].item():.4f}")

is_animal,_,probs = learn.predict(PILImage.create('stone.jpg'))
print(f"Probability it's an animal : {probs[0].item():.4f}")
```

The result shows that a bird image can be identified successfully, while a stone, non-animal, image will be identified with low probability rate.

![](/images/11.png)
