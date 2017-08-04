# Transfer Learning Note

## Surveys
### Blogs
[Transfer Learning - Machine Learning's Next Frontier]
(http://sebastianruder.com/transfer-learning/)

### Papers

Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345–1359

Karl Weiss, Taghi M. Khoshgoftaar and DingDing Wang (2016). A survey of transfer learning. Journal of Big Data. 2016 3:9.


## Applications

### Learning from simulations
Learning from a simulation and applying the acquired knowledge to the real world is an instance of transfer learning scenario 2, as the feature spaces between source and target domain are the same (both generally rely on pixels), but the marginal probability distributions between simulation and reality are different, i.e. objects in the simulation and the source look different, although this difference diminishes as simulations get more realistic. At the same time, the conditional probability distributions between simulation and real wold might be different as the simulation is not able to fully replicate all reactions in the real world, e.g. a physics engine can not completely mimic the complex interactions of real-world objects.
 
Rusu, A. A., Vecerik, M., Rothörl, T., Heess, N., Pascanu, R., & Hadsell, R. (2016). Sim-to-Real Robot Learning from Pixels with Progressive Nets. arXiv Preprint arXiv:1610.04286. Retrieved from [here](http://arxiv.org/abs/1610.04286).

Mikolov, T., Joulin, A., & Baroni, M. (2015). A Roadmap towards Machine Intelligence. arXiv Preprint arXiv:1511.08130. Retrieved from [here](http://arxiv.org/abs/1511.08130)

### Adapting to new domains
learning from simulations is a particular instance of domain adaptation.

In computer vision, often the data where labeled information is easily accessible and the data that we actually care about are different.

Another common domain adaptation scenario pertains to adapting to different text types.

Automatic Speech Recognition (ASR), now more than ever do we need systems that are able to adapt to individual users and minorities to ensure that everyone's voice is heard.

### Transferring knowledge across languages
learning from one language and applying our knowledge to another language is -- in my opinion -- another killer application of transfer learning. [Blog](http://sebastianruder.com/cross-lingual-embeddings/index.html)

Given the current state-of-the-art, this still seems utopian, but recent advances such as zero-shot translation promise rapid progress in this area.

Johnson, M., Schuster, M., Le, Q. V, Krikun, M., Wu, Y., Chen, Z., … Dean, J. (2016). Google’s Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation. arXiv Preprint arXiv:1611.0455.

## Methods

### Using pre-trained CNN features
__Off-the-shelf CNN features__

Razavian, A. S., Azizpour, H., Sullivan, J., & Carlsson, S. (2014). CNN features off-the-shelf: An astounding baseline for recognition. IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, 512–519.

__Learning the underlying structure of images__

Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. ICLR. Retrieved from [here](http://arxiv.org/abs/1511.06434)

__Are pre-trained features useful beyond vision?__

Currently, there are no off-the-shelf features that achieve results for natural language processing that are as astounding as their vision equivalent. 

Jozefowicz, R., Vinyals, O., Schuster, M., Shazeer, N., & Wu, Y. (2016). Exploring the Limits of Language Modeling. arXiv Preprint arXiv:1602.02410. Retrieved from [here](http://arxiv.org/abs/1602.02410)

Ramachandran, P., Liu, P. J., & Le, Q. V. (2016). Unsupervised Pretrainig for Sequence to Sequence Learning. arXiv Preprint arXiv:1611.02683.

Bingel, J., & Søgaard, A. (2017). Identifying beneficial task relations for multi-task learning in deep neural networks. In EACL. Retrieved from [here](http://arxiv.org/abs/1702.08303)

Plank, B., Søgaard, A., & Goldberg, Y. (2016). Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics.

Yu, J., & Jiang, J. (2016). Learning Sentence Embeddings with Auxiliary Tasks for Cross-Domain Sentiment Classification. Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP2016), 236–246. Retrieved from [here](http://www.aclweb.org/anthology/D/D16/D16-1023.pdf)

### Learning domain-invariant representations

Glorot, X., Bordes, A., & Bengio, Y. (2011). Domain Adaptation for Large-Scale Sentiment Classification: A Deep Learning Approach. Proceedings of the 28th International Conference on Machine Learning, 513–520. Retrieved from [here](http://www.icml-2011.org/papers/342_icmlpaper.pdf)

Chen, M., Xu, Z., Weinberger, K. Q., & Sha, F. (2012). Marginalized Denoising Autoencoders for Domain Adaptation. Proceedings of the 29th International Conference on Machine Learning (ICML-12), 767--774. [here](http://doi.org/10.1007/s11222-007-9033-z)

### Making representations more similar

Daumé III, H. (2007). Frustratingly Easy Domain Adaptation. Association for Computational Linguistic (ACL), (June), 256–263. [here](http://doi.org/10.1.1.110.2062)

Sun, B., Feng, J., & Saenko, K. (2016). Return of Frustratingly Easy Domain Adaptation. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16). Retrieved from [here](http://arxiv.org/abs/1511.05547)

Bousmalis, K., Trigeorgis, G., Silberman, N., Krishnan, D., & Erhan, D. (2016). Domain Separation Networks. NIPS

Tzeng, E., Hoffman, J., Zhang, N., Saenko, K., & Darrell, T. (2014). Deep Domain Confusion: Maximizing for Domain Invariance. CoRR. Retrieved from [here](https://arxiv.org/pdf/1412.3474.pdf)

### Confusing domains
Another way to ensure similarity between the representations of both domains that has recently become more popular is to add another objective to an existing model that encourages it to confuse the two domains.

Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. In Proceedings of the 32nd International Conference on Machine Learning. (Vol. 37).

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., … Lempitsky, V. (2016). Domain-Adversarial Training of Neural Networks. Journal of Machine Learning Research, 17, 1–35. [here](http://www.jmlr.org/papers/volume17/15-239/source/15-239.pdf)

## Related Research Areas
Transfer learning is by far not the only area of machine learning that seeks to leverage limited amounts of data, use learned knowledge for new endeavours, and enable models to generalize better to new settings. In the following, we will thus introduce other directions that are related or complementary to the goals of transfer learning.

### Semi-supervised learning

### Using available data more effectively
Plank, B. (2016). What to do about non-standard (or non-canonical) language in NLP. KONVENS 2016. Retrieved from [here](https://arxiv.org/pdf/1608.07836.pdf)

### Improving models' ability to generalize
Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. ICLR 2017.

### Making models more robust
Kurakin, A., Goodfellow, I., & Bengio, S. (2017). Adversarial examples in the physical world. In ICLR 2017. Retrieved from [here](http://arxiv.org/abs/1607.02533)

Huang, S., Papernot, N., Goodfellow, I., Duan, Y., & Abbeel, P. (2017). Adversarial Attacks on Neural Network Policies. In Workshop Track - ICLR 2017.

### Multi-task learning
Bingel, J., & Søgaard, A. (2017). Identifying beneficial task relations for multi-task learning in deep neural networks. In EACL. Retrieved from [here](http://arxiv.org/abs/1702.08303)

### Continuous learning

### Zero-shot learning
Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., & Wierstra, D. (2016). Matching Networks for One Shot Learning. NIPS 2016. Retrieved from [here](http://arxiv.org/abs/1606.04080)

Ravi, S., & Larochelle, H. (2017). Optimization as a Model for Few-Shot Learning. In ICLR 2017.

Xian, Y., Schiele, B., Akata, Z., Campus, S. I., & Machine, A. (2017). Zero-Shot Learning - The Good, the Bad and the Ugly. In CVPR 2017.

Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017). Adversarial Discriminative Domain Adaptation.
