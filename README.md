# SegPosNer_Bilstm_CRF
基于深度学习bi_lstm_crf的(分词/词性标注/实体识别)实现.

# Data Demo
seg:train.txt
```
预	B
约	E
即	B
可	E
获	B
得	E
考	B
试	E
大	S
礼	B
包	E
```
pos:pos_train.txt
```
迈向	v
充满	v
希望	n
的	u
新	a
世纪	n
```

# Run
> python train.py

```
start to training...
step: 20/20000...  loss: 60.6413...  1.0220 sec/batch
step: 40/20000...  loss: 55.0804...  1.1040 sec/batch
.
.
step: 3760/20000...  loss: 2.1385...  0.9900 sec/batch
step: 3780/20000...  loss: 1.9285...  0.9870 sec/batch
step: 3800/20000...  loss: 1.5843...  1.0100 sec/batch
val len: 5000
accuracy:85.58%. best:87.26%

```

# Test
> python test.py
```
start to testing...

word / tag / pre
上 / B / B
述 / E / E
担 / B / B
保 / E / E
不 / S / B
构 / B / E
成 / E / S
关 / B / B
联 / E / B
交 / B / E
易 / E / S
。 / S / S
```

# References
https://github.com/Franck-Dernoncourt/NeuroNER
https://github.com/rockyzhengwu/FoolNLTK

