# FastAI and Deep learning

 What is Deep Learning :alien:
  -	Deep learning is a computer technique to extract and transform data –
    with use cases ranging from human speech recognition to animal imagery 
    classification – by using multiple layers of neural networks.
  - Each of these layers takes its inputs from previous layers and progressively refines them.
  -	The layers are trained by algorithms that minimize their errors and improve their accuracy.
  -	In this way, the network learns to perform a specified task.

![Image of neural network logo](images/1.png)

Nearly all of deep learning is based on a single type of model, the neural network. The name and 
structure of neural network are inspired by the human brain, mimicking the way that biological neurons signal to one another.

The artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. 
Each node connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, 
that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.

Neural networks rely on training data to learn and improve their accuracy over time.

# First Model

This model implements fastAI to train to model and make it be able to identify animals (dog/cat) from sample pictures.

The full code and working process are displayed below:

![Image of fisrt model](images/2.png)

:memo:The first line imports all of the **_fastai.vision_** library. This gives all of the functions and classes needed to create a wide variety of computer vision models.

![Image of fisrt model code](images/3.png)

:memo:The second line downloads a standard dataset from the fast.ai database collection to local server, extracts it, and returns a path object with the extracted location.

![Image of fisrt model code](images/4.png)

:memo:
In the third line we define a function, **_is_cat_**, which labels cats based on a filename rule provided by the dataset creators.

![Image of fisrt model code](images/5.png)

:memo:
The function in the fourth line tells fastai what kind of dataset we have and how it is structured. The class for deep learning datasets we are using is **_ImageDataLoaders_**. 
The last line tells fastai to use the **_is_cat_** function just defined to get the labels from the dataset. Finally, we define the Transforms that we need.
A Transform contains code that is applied automatically during training. In this case, each item is resized to a 224-pixel square.

By increasing the size, model with better results can be obtained (since it will be able to focus on more details), but at the price of speed and memory consumption.

The parameter **_valid_pct=0.2_** tells fastai to hold out 20% of the data and not use it for training the model at all. This 20% of the data is called the validation set; the remaining 80% is called the training set.
The validation set is used to measurer the accuracy of the model. By default, the 20% is selected randomly. 
The parameter **_seed=42_** sets the random seed to the same value every time we run this code, which means we get the same validation set every time we run it. 
By doing so, if we change our model and retrain it, we know that any differences are due to the changed to the model, not due tot having a different random validation set.

![Image of fisrt model code](images/6.png)

:memo:
The fifth line of the code training our image recognizer tells fastai to create a **_convolutional neural network (CNN)_** and specifies what architecture to use, what data we want to train it on, and what **_metric_** to use.

The 34 in resnet34 refers to the number of layers in this variant of the architecture (other options are 18, 50, 101, and 152). Models using architectures with more layers take longer to train, and are more prone to overfitting.
On the other hand, when using more data, they can be quite a bit more accurate.

A **_metric_** is a function that measures the quality of the model's predictions using the validation set, and will be printed at the end of each epoch.
In this case, we're using **_error_rate_**, which is a function provided by fastai that does just what it says: tells you what percentage of images in the validation set are being classified incorrectly.
Another common metric for classification is accuracy.3

![Image of fisrt model code](images/7.png)

:memo:
The sixth line of our code tells fastai how to fit the model. In order to fit a model, we have to provide at least one piece of information: how many times to look at each image (known as number of epochs). 
The number of epochs you select will largely depend on how much time you have available, and how long you find it takes in practice to fit your model.

The head of a model is the part that is newly added to be specific to the new dataset. An epoch is one complete pass through the dataset. After calling fit, the results after each epoch are printed, showing the epoch number,
the training and validation set losses (the "measure of performance" used for training the model), and any metrics you've requested (error rate, in this case).

![Image of fisrt model code](images/8.png)

From this experiment we can see some fundamental things about a deep learning model: :alien:
  +	A model cannot be created without data.
  +	A model can only learn to operate on the patterns seen in the input data used to train it.
  +	This learning approach only creates predictions, not recommended actions.
  +	It’s not enough to just have examples of input data; we need labels for that data too.

In general, the primary goal when creating a model is ensuring the model to be useful on data that the model only sees in the future, after it has been trained. The longer you trained the model for, 
the better your accuracy will get on the training se; the validation set accuracy will improve for a while, but eventually it will start getting worse as the model starts to memorize the training set. 
When it happens, we say that the model is overfitting.


