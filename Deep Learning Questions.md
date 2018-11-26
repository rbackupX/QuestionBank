#### Questions on Deep Learning

* Design a network to detect two object classes if you know there is going to be only single instance of each object in the image. How the design changes if multiple, unknown number of instances are present? How the design and strategy changes if the number of object classes to be detected is huge ( > 10K)?
* Let me also share questions from published material which tests if the candidate is well prepared to understand the current literature/existing approaches. This is by no means an exhaustive list:
* Semantic segmentation, Object detection
* Explain max un-pooling operation for increasing the resolution of feature maps.
* What is a Learnable up-sampling or Transpose convolution ?
* Describe the transition between R-CNN, Fast R-CNN and Faster RCNN for object detection.
* Describe how RPNs are trained for prediction of region proposals in Faster R-CNN?
* Describe the approach in SSD and YOLO for object detection. How these approaches differ from Faster-RCNN. When will you use one over the other?
* Difference between Inception v3 and v4. How does Inception Resnet compare with V4.
* Explain main ideas behind ResNet? Why would you try ResNet over other architectures?
* Explain batch gradient descent, stochastic gradient descent and mini-match gradient descent.
* Loss functions: Cross-entropy, L2, L1
* Explain Dropout and Batch Normalization. Why Batch Normalization helps in faster convergence?
* Are neural networks and deep learning overrated?
* Can deep learning and neural networks be patented?
* What leadership questions should I expect from an Amazon on-site interview for a Software Engineering role?
* What is learning in neural network?
* What is the difference between Neural Networks and Deep Learning?
* Interestingly, this question as applied to Deep Learning does have a definitive answer for me, whereas the general form of the question may not.
* It is a relatively new topic in the general software engineering population. It has not yet been taught for years in college by professors who have extracted insightful ways to teach the fundamentals. So a lot knowledge here is gleaned from watching advanced talks and reading research papers. Unfortunately, this also means that many candidates have a strong functional knowledge of the state-of-the-art Whats and Hows, yet not fully mastering the Whys.
* So, I find that there are indeed "toughest NN and Deep Learning" questions, where many otherwise knowledgeable candidates fall down. They might give you technically correct answers, using lots of jargon, but never getting to the heart of the issue. They might give you an answer involving a lot of correct Hows, but that reveal they don't really understand the fundamental Whys. The best answers to these questions cannot (yet) be easily Googled. They are invariably of this pattern:
* Explain the following, so that a colleague new to the field/an eighth grader can understand (in no particular order, not exhaustive):
* What is an auto-encoder? Why do we "auto-encode"? Hint: it's really a misnomer.
* What is a Boltzmann Machine? Why a Boltzmann Machine?
* Why do we use sigmoid for an output function? Why tanh? Why not cosine? Why any function in particular?
* Why are CNNs used primarily in imaging and not so much other tasks?
* Explain backpropagation. Seriously. To the target audience described above.
* Is it OK to connect from a Layer 4 output back to a Layer 2 input?
* A data-scientist person recently put up a YouTube video explaining that the essential difference between a Neural Network and a Deep Learning network is that the former is trained from output back to input, while the latter is trained from input toward output. Do you agree? Explain.

* Try these yourself and see if you do indeed have mastery of the fundamentals. If you do, an eighth grader ought to be able to understand and repeat your explanation.
*
* I had some “deep learning interviews” recently, and I thought I could share some questions. First of all, be aware that most of the time, questions don’t have a single answer, and the interviewer just wants to talk with you to see if you are confident about the notions.
* Usually the first questions are : what do you know about some “pre-deep learning epoch” algorithms, like SVM, KNN, Kmeans, Random Forest…?
* Talking about deep learning, here are the questions I was asked to answer:
* Implement dropout during forward and backward pass?
* Was not very hard, you just have to consider what’s happening during testing vs training phase. In this question, the interviewer can test your knowledge on dropout, and backprop
* Neural network training loss/testing loss stays constant, what do you do?
* Open question (ask if there could be an error in your code, going deeper, going simpler…)
* Why do RNNs have a tendency to suffer from exploding/vanishing gradient?
* And probably you know the next question… How to prevent this? You can talk about LSTM cell which helps the gradient from vanishing, but make sure you know why it does so. I also remember having a nice conversation about gradient clipping, where we wonder whether we should clip the gradient element wise, or clip the norm of the gradient.
* Then I had a lot of question about some modern architecture, such as Do you know GAN, VAE, and memory augmented neural network? Can you talk about it?
* Of course, let me talk about the beauty of variational auto encoder.
* Some maths questions such as: Does using full batch means that the convergence is always better given unlimited power?
* What is the problem with sigmoid during backpropagation?
* Very small, between 0.25 and zero.[2]
* Given a black box machine learning algorithm that you can’t modify, how could you improve its error?
* Open question, you can transform the input for example.
* How to find the best hyper parameters?
* Random search, grid search, Bayesian search (and what it is?)
* What is transfer learning?
* I was also asked to implement some papers idea, but it was more as an assignment, than during an interview. Finally I also get non ML questions, more like algorithmic questions


* Good luck for your interview. If you are enough curious, and have a correct knowledge of the field, they will notice it, and you will pass a good moment with the interviewer.
*
* However, I do have some questions to test whether candidates really understand deep learning.
* Can they derive the back-propagation and weights update?
* Extend the above question to non-trivial layers such as convolutional layers, pooling layers, etc.
* How to implement dropout
* Their intuition when and why some tricks such as max pooling, ReLU, maxout, etc. work. There are no right answers but it helps to understand their thoughts and research experience.
* Can they abstract the forward, backward, update operations as matrix operations, to leverage BLAS and GPU?
* If a candidate shows early signs that he/she is an expert in DL, it's not necessary to ask all those questions in details. We can discuss one of their papers or a recent hot paper or something that is not necessarily DL.
* 86.3k Views · View 220 Upvoters
* Related Questions
* What are the learning algorithm in deep neural network?
* How can a neural network learn itself?
* Do you have to show your face during a Google Hangouts interview?
* What is neural networking?
* How can deep learning networks generate images?
* How do neural networks of neural networks behave?
* What is the best book or resource to learn about Neural Networks and Deep Neural Networks?
* What is the best YouTube channel to learn deep learning and neural networks?
* What topics come under deep learning other than neural networks?
* How do I implement deep neural network?
* What is the difference between neural networks and deep neural networks?
* Why are deep networks characterized by neural networks?
* ELI5: What are neural networks?
* What are neural networks in machine learning?
* What are the deep learning algorithms other than neural networks?
* Are neural networks and deep learning overrated?
* Can deep learning and neural networks be patented?
* What leadership questions should I expect from an Amazon on-site interview for a Software Engineering role?
* What is learning in neural network?
* What is the difference between Neural Networks and Deep Learning?
* What are the learning algorithm in deep neural network?
* How can a neural network learn itself?
* Do you have to show your face during a Google Hangouts interview?
* What is neural networking?
* How can deep learning networks generate images?
'
source    
[1]  https://www.quora.com/What-are-the-toughest-neural-networks-and-deep-learning-interview-questions'
[2] Is full-batch gradient descent, with unlimited computer power, always better than mini-batch gradient descent?
