## Датасет EuroSAT
Датасет состоит из 27 000 изображений, поделённых на 10 классов. Изображения получены со спутника Sentinel-2

## Примеры изображений
![Примеры изображений](images/EuroSAT.jpg) <br />

## Учитель
Была обучена модель ResNet101: точность на тестовом датасете составила 0.977, <br />
количество обучаемых параметров - 42 520 650 <br />

## Дистилляция
Дистилляция - это один из методов ускорения нейронных сетей <br />

При дистилляции помимо обучающих данных необходима модель-учитель (модель с большим количеством параметров, которая имеет <br />
высокую точность предсказаний), данная модель участвует в обучении модели-студента. <br />
Данный подход применяется, чтобы обучить нейронную сеть с небольшим количеством параметров. <br />
При обучении модели-студента считаются потери не только с истинными значениями, но и с <br />
предсказаниями модели-учителя.

## Функция обучения с дистилляцией
```python
def train_with_distillation(train_loader, test_loader, model, loss_fn, optimizer, teacher_model, max_lr, epochs, alpha = 0.1, temperature = 3):
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.5)
    teacher_model.eval()
    for epoch in range(epochs):
      model.train()
      for data, targets in tqdm(train_loader):
          data = data.to(DEVICE)
          targets = targets.to(DEVICE)

          scores = model(data)
          teacher_model_scores = teacher_model(data)
          teacher_prob = F.softmax(teacher_model_scores/temperature, dim=1)
          student_prob = F.softmax(scores/temperature, dim=1)
          dist_loss = F.kl_div(student_prob, teacher_prob)
          loss = alpha * loss_fn(scores, targets) + (1 - alpha) * dist_loss

          
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
      sched.step()
      check_accuracy(test_loader, model, DEVICE)


```
<br /> <br />

# Модель-студент

```python

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

def conv_block(in_channels, out_channels, pool=False):
    layers = [SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SmallMyNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 16)
        self.conv2 = conv_block(16, 32, pool=True)
        self.res1 = nn.Sequential(conv_block(32, 32), conv_block(32, 32))

        self.transition1 = SqueezeExcitation(32, 16)
        
        self.conv3 = conv_block(32, 64, pool=True)
        self.conv4 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.transition2 = SqueezeExcitation(64, 32)

        self.conv5 = conv_block(64, 128, pool=True)
        self.conv6 = conv_block(128, 128, pool=True)
        self.res3 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(),
                                        nn.Linear(128, num_classes)
                                        )
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out

        out = self.transition1(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out

        out = self.transition2(out)

        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out

        out = self.classifier(out)
        return out
```
<br /><br />
Количество обучаемых параметров модели - 89 397 <br />

При обучении данной модели без использования дистилляции была получена точность на тестовом датасете = 0.908 <br />

При обучении с использованием дистилляции была получена точность  на тестовом датасете = 0.928 <br />

