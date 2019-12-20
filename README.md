# 学习fastai而进行的项目学习
+ [数字图像识别](https://www.lintcode.com/ai/digit-recognition/data)

---

+ 手写数字生成任务  
**原始数据**
![](digital_gan/original_examples.png)
**自带wgan生成的数据**
![](digital_gan/bad_examples.png)
**预训练对抗生成结果**
![](digital_gan/maybe_good.png)
表现很差的原因在于，我们预训练的生成器，使用的是上一步wgan作为图片。  
现在改进一下，直接对生成器做一个预训练，然后模型效果如下图所示：
![](digital_gan/good.png)
可以看到生成的效果已经非常不错了~此项目到底结束~

---

