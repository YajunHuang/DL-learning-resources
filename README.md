# My Deep Learning Resources
### Yajun Huang, 黄亚军

Deep learning resources that I marked here for reading and learning myself.

## Deep Learning for NLP 
### Survey, book, tutorial, class:
Deep learning for NLP 的综述论文: <br/>
Yoav Goldberg. 2015. A primer on neural network models for natural language processing. arXiv preprint arXiv:1510.00726.

Deep Learning Book: <br/>
(Reading) Ian Goodfellow, Yoshua Bengio and Aaron Courville. Deep Learning. MIT Press

Stanford CS231n: Convolutional Neural Networks for Visual Recognition. <br/>
要把notes看完，三份作业也做完，作业设计的很好。

Stanford CS224d: Deep Learning for Natural Language Processing. <br/>

### Word2Vect
T.Mikolov 的训练 word2vect 方法的论文：<br/>
T. Mikolov, K. Chen, G. Corrado, J. Dean, Efficient estimation of word representations in vector space, CoRR abs/1301.3781.

将word2vec和lda联合训练的work，既考虑相邻词之间的关系又考虑document对词的影响，神经网络结合词的 word vector 和文本的 topic model(lda) vector，同时训练。In this work, we describe lda2vec, a model that learns dense word vectors jointly with Dirichlet-distributed latent document-level mixtures of topic vectors: <br/>
Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec. arXiv:1605.02019v1 <br/>
https://github.com/cemoody/lda2vec <br/>
http://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec <br/>

### CNN for NLP model and text classification
CNN 对句子语义建模的理论论文： <br/>
Kalchbrenner N, Grefenstette E, Blunsom P. A Convolutional Neural Network for Modelling Sentences[C]. Proceedings of ACL. Baltimore and USA: Association for Computational Linguistics, 2014: 655-665.

CNN 对句子语义分类的理论论文：<br/>
Kim Y. Convolutional Neural Networks for Sentence Classification[C]. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). Doha, Qatar: Association for Computational Linguistics, 2014: 1746–1751.

CNN 对句子语义分类的实践指导论文：<br/>
Ye Zhang and Byron C. Wallace. 2015. A sensitivity analysis of (and practitioners’ guide to) convolutional neural networks for sentence classification. arXiv preprint arXiv:1510.03820.

CNN 对句子语义分类的另一种思路，主要考虑在one-hot词上直接进行CNN学习，找出具有语义的“word region”：<br/>
Johnson, Rie and Zhang, Tong. Effective use of word order for text categorization with convolutional neural networks. In NAACL HLT, 2015a.

上一篇论文的后续，主要考虑利用更多数据或者额外信息先做unsupervised learning，提高模型效果: <br/>
Johnson, Rie and Zhang, Tong. Semi-supervised convolutional neural networks for text categorization via region embedding. In NIPS, 2015b.

利用 CNN 网络直接在字符集做语义建模和文本分类，使用一个深层的CNN网络（9层, 6 convolutional layers and 3 fully connected layers）: <br/>
Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Advanced in Neural Information Processing Systems (NIPS 2015), volume 28. <br/>
code: https://github.com/zhangxiangxiao/Crepe

上一篇的early version。This technical report is superseded by a paper entitled "Character-level Convolutional Networks for Text Classification", arXiv:1509.01626. It has considerably more experimental results and a rewritten introduction: <br/>
Xiang Zhang, Yann LeCun. Text Understanding from Scratch. arXiv:1502.01710

利用resnet技术，使用更深层次的CNN网络在字符集上做语义建模的，可以达到几十层。参照计算机视觉的工作，Character-level convolutional networks for text classification是DL4NLP的VGG版本，这篇是ResNet版本。
Alexis Conneau, Holger Schwenk, Loïc Barrault, Yann Lecun. Very Deep Convolutional Networks for Natural Language Processing. arXiv:1606.01781

### RNN for NLP model and text classification
RNN 做文本语义分类的理论论文：<br/>
Siwei Lai, Liheng Xu, Kang Liu, and Jun Zhao. Recurrent convolutional neural networks for text classification. In Proc. Conference of the Association for the Advancement of Artificial Intelligence (AAAI), 2015.

Johnson 前面两个工作的后续，用 LSTM 代替 CNN： <br/>
Johnson, Rie and Zhang, Tong. Supervised and Semi-Supervised Text Categorization
using One-Hot LSTM for Region Embeddings. arXiv:1602.02373


### CNN-RNN for NLP model and text classification
CNN + RNN 做文本语义建模的理论论文，现有 CNN 表示一个词的字符组合特征，再用RNN表示词序之间的特征: <br/>
Yoon Kim, Yacine Jernite, David Sontag, and Alexander M Rush. 2015. Character aware neural language models. arXiv preprint arXiv:1508.06615.

CNN + RNN 做文本语义分类的理论论文: <br/>
Yijun Xiao, Kyunghyun Cho. Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers


### Memory NN:
第一个在 sequence-to-sequence 任务中提出 memory 和 attention 的论文，memory是 encoder RNN网络中多层的 hidden states，attention是计算 decoder 当前 state 与 encoder 的 states 的相似度： <br/>
Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate [J]. arXiv, 2014.

用基于 memory 和 attention 的神经网络表示程序代码并判断代码执行，允许更新 memory，attention 使用 content 和 location 两种信息的关联关系（并使用sharpen突出特征）: <br/>
Graves, G. Wayne, and I. Danihelka. Neural turing machines. arXiv preprint 1410.5401, 2014. 

基于 memory 和 attention 的神经网络处理NLP问题，语言模型、问答等场景： <br/>
J. Weston, S. Chopra, and A. Bordes. Memory networks. In International Conference on Learning Representations (ICLR), 2015. 

基于 Memory Network 的改进，端对端网络，易于训练和应用于其他问题： <br/>
Sukhbaatar S, Weston J, Fergus R. End-to-end memory networks[C] Advances in Neural Information Processing Systems. 2015: 2431-2439.

基于 Memory Network 的改进，使用复杂的网络结构，端对端网络，可处理多种问题，包括问答，POS，语言模型，情感分类： <br/>
Kumar A, Irsoy O, Su J, et al. Ask me anything: Dynamic memory networks for natural language processing[J]. arXiv preprint arXiv:1506.07285, 2015.


### Others:
使用 RNN 做 encoder & decoder 模型，训练机器翻译神经网络: <br/>
Sutskever I, Vinyals O, Le Q V V. Sequence to Sequence Learning with Neural Networks[M]. Advances in Neural Information Processing Systems 27. 2014: 3104-3112.

Li J, Luong M T, Jurafsky D. A Hierarchical Neural Autoencoder for Paragraphs and Documents[C]. Proceedings of ACL. 2015.

Zeng D, Liu K, Lai S, et al. Relation Classification via Convolutional Deep Neural Network[C]. Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers. Dublin, Ireland: Association for Computational Linguistics, 2014: 2335–2344.

比较 word2vec-avg, word2vec-weighted, RNN, LSTM 等6种模型在 sentence modeling 和 multi-source sentence modeling 的效果，对比结果可以参考: <br/>
John Wieting, Mohit Bansal, Kevin Gimpel, Karen Livescu. Towards Universal Paraphrastic Sentence Embeddings. Published as a conference paper at ICLR 2016

compositional and attentional 模型处理Q&A。根据问句的句法结构，选出对应的modules构造神经网络，使用RL和<question, world, answer>数据进行训练: <br/>
Learning to compose neural networks for question answering. Jacob Andreas, Marcus Rohrbach, Trevor Darrell and Dan Klein. NAACL 2016. <br/>
https://github.com/jacobandreas/nmn2

Hu B, Chen Q, Zhu F. LCSTS: a large scale chinese short text summarization dataset[J]. arXiv preprint arXiv:1506.05865, 2015.


### Deep Reinforcement Learning
Jiwei Li, Will Monroe, Alan Ritter, Dan Jurafsky. Deep Reinforcement Learning for Dialogue Generation. arXiv:1606.01541


### Name Entity Recognition(NER)
Boosting Named Entity Recognition with Neural Character Embeddings. ACL 2015

Chinese Word Segmentation and Named Entity Recognition: A Pragmatic Approach. ACL, 2005

Chinese Named Entity Recognition using Lexicalized HMMs. SigKDD, 2005


## Very Deep Architecture 
在 CNN 深度网络中使用 Highway 的方法减弱 error vanishing的影响: <br/>
Rupesh Kumar Srivastava, Klaus Greff, and Jurgen Schmidhuber. Highway networks. ¨ arXiv:1505.00387 [cs], May 2015.

Highway networks 的扩展: <br/>
R. K. Srivastava, K. Greff, and J. Schmidhuber. Training very deep networks. 1507.06228, 2015.

Batch normalization: <br/>
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. CoRR, abs/1502.03167, 2015. arxiv.org/abs/1502.03167.

VGG network: <br/>
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In CVPR, 2016.

Residual network: <br/>
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale
image recognition. In ICLR, 2015.

FractalNet: Ultra-Deep Neural Networks without Residuals. arXiv:1605.07648v1.


## Deep Learning Brilliant Blogs
http://www.wildml.com/

Implementing a CNN for Text Classification in TF: <br/>
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

CONTEXT v3: Convolutional neural networks and LSTM for text categorization in C++ on GPU: <br/>
http://riejohnson.com/cnn_download.html

Standford computer vision and deep learning class: <br/>
http://cs224d.stanford.edu/syllabus.html

NLP DL lecture note: <br/>
https://github.com/nyu-dl/NLP_DL_Lecture_Note

LSTM character awareness CNN: <br/>
https://github.com/yoonkim/lstm-char-cnn

neural networks and deep learning: <br/>
http://neuralnetworksanddeeplearning.com

understanding LSTM: <br/>
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

demystifying deep reinforcement learning: <br/>
http://www.nervanasys.com/demystifying-deep-reinforcement-learning/

## demos
http://gitxiv.com/posts/jG46ukGod8R7Rdtud/a-neural-algorithm-of-artistic-style

https://www.youtube.com/watch?v=V1eYniJ0Rnk

https://www.youtube.com/watch?v=p88R2_3yWPA

http://people.eecs.berkeley.edu/~igor.mordatch/policy/index.html

https://www.youtube.com/watch?v=0VTI1BBLydE


