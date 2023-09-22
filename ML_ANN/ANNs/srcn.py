


def ResNet50_model(ds):
      input = ds.shape[1:]
      print(input)
      classes = 4
      resnet = ResNet50(include_top=False, input_shape=input, pooling='avg', weights="imagenet")
      for layer in resnet.layers[:-20]:
      layer.trainable = False
      model = Sequential()
      model.add(resnet)
      model.add(Flatten())
      model.add(Dense(classes, activation='softmax'))
      model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      return model
