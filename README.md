## RealnessGAN &mdash; Simple TensorFlow Implementation [[Paper]](https://openreview.net/pdf?id=B1lPaCNtPB)
### : Real or Not Real, that is the Question

## Usage
```python

fake_img = generator(noise)

real_logit = discriminator(real_img)
fake_logit = discriminator(fake_img)

g_loss = generator_loss(fake_logit)
d_loss = discriminator_loss(real_logit, fake_logit)

```

## Author
[Junho Kim](http://bit.ly/jhkim_ai)
